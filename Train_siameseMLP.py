import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dataset_registry import get_dataset_spec
from feature_cache_utils import build_cache_dir, cache_ready, load_array_cache, read_metadata, write_array_cache
from training_runtime_utils import (
    cleanup_runtime,
    get_ddp_timeout,
    is_main_process,
    maybe_parallelize,
    reduce_sum_and_count,
    setup_runtime,
    wait_for_all_ranks,
    unwrap_state_dict,
)


DEFAULT_DATASET = get_dataset_spec("gaze360")


class Config:
    TRAIN_FILE = DEFAULT_DATASET.train_file
    VALID_FILE = DEFAULT_DATASET.valid_file
    MODEL_SAVE_PATH = "models/gaze360_siameseMLP.pth"

    BRANCH_HIDDEN_WIDTH = 64
    FUSION_HIDDEN_WIDTH = 128
    NUM_BLOCKS = 3
    DROPOUT_RATE = 0.1

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 15
    RANDOM_STATE = 42

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    AUGMENTATION_NOISE_STD = 0.0

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    LEFT_INNER_CORNER = 133
    LEFT_OUTER_CORNER = 33
    LEFT_EYE_CONTOUR = [LEFT_OUTER_CORNER, LEFT_INNER_CORNER, 159, 145]
    RIGHT_INNER_CORNER = 362
    RIGHT_OUTER_CORNER = 263
    RIGHT_EYE_CONTOUR = [RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, 386, 374]
    HEAD_ANCHORS = [1, 168]

    LEFT_EYE_LANDMARKS = LEFT_IRIS + LEFT_EYE_CONTOUR
    RIGHT_EYE_LANDMARKS = RIGHT_IRIS + RIGHT_EYE_CONTOUR
    SCALE_FACTOR = 448.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train landmark SiameseMLP gaze regressor.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset alias from dataset_registry.py (e.g. gaze360, gazegene, xgaze, blender).",
    )
    parser.add_argument("--train-file", default=None, help="Training CSV path.")
    parser.add_argument("--valid-file", default=None, help="Validation CSV path.")
    parser.add_argument("--model-save-path", default=None, help="Output checkpoint path.")
    parser.add_argument("--device", default="auto", help='Device string, e.g. "auto", "cuda", "cuda:0", "cpu".')
    parser.add_argument("--multi-gpu", action="store_true", help="Enable torch.nn.DataParallel on all visible GPUs.")
    parser.add_argument("--distributed", action="store_true", help="Enable torch.distributed DDP (launch with torchrun).")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--branch-hidden-width", type=int, default=None)
    parser.add_argument("--fusion-hidden-width", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--augmentation-noise-std", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(".cache", "feature_tensors"),
        help="Directory for precomputed feature caches shared across runs/ranks.",
    )
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.dataset:
        spec = get_dataset_spec(args.dataset)
        if spec.train_file is None or spec.valid_file is None:
            raise ValueError(f"Dataset '{spec.key}' does not define default TRAIN/VALID files.")
        config.TRAIN_FILE = spec.train_file
        config.VALID_FILE = spec.valid_file
        config.MODEL_SAVE_PATH = f"models/{spec.key}_siameseMLP.pth"

    if args.train_file is not None:
        config.TRAIN_FILE = args.train_file
    if args.valid_file is not None:
        config.VALID_FILE = args.valid_file
    if args.model_save_path is not None:
        config.MODEL_SAVE_PATH = args.model_save_path
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.weight_decay is not None:
        config.WEIGHT_DECAY = args.weight_decay
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    if args.patience is not None:
        config.PATIENCE = args.patience
    if args.branch_hidden_width is not None:
        config.BRANCH_HIDDEN_WIDTH = args.branch_hidden_width
    if args.fusion_hidden_width is not None:
        config.FUSION_HIDDEN_WIDTH = args.fusion_hidden_width
    if args.num_blocks is not None:
        config.NUM_BLOCKS = args.num_blocks
    if args.dropout_rate is not None:
        config.DROPOUT_RATE = args.dropout_rate
    if args.augmentation_noise_std is not None:
        config.AUGMENTATION_NOISE_STD = args.augmentation_noise_std


def build_siamese_arrays(data_frame: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l_cols_x = [f"{idx}_x" for idx in config.LEFT_EYE_LANDMARKS]
    l_cols_y = [f"{idx}_y" for idx in config.LEFT_EYE_LANDMARKS]
    r_cols_x = [f"{idx}_x" for idx in config.RIGHT_EYE_LANDMARKS]
    r_cols_y = [f"{idx}_y" for idx in config.RIGHT_EYE_LANDMARKS]
    h_cols_x = [f"{idx}_x" for idx in config.HEAD_ANCHORS]
    h_cols_y = [f"{idx}_y" for idx in config.HEAD_ANCHORS]

    l_xs = data_frame[l_cols_x].to_numpy(dtype=np.float32)
    l_ys = data_frame[l_cols_y].to_numpy(dtype=np.float32)
    r_xs = data_frame[r_cols_x].to_numpy(dtype=np.float32)
    r_ys = data_frame[r_cols_y].to_numpy(dtype=np.float32)

    l_in = data_frame[[f"{config.LEFT_INNER_CORNER}_x", f"{config.LEFT_INNER_CORNER}_y"]].to_numpy(dtype=np.float32)
    l_out = data_frame[[f"{config.LEFT_OUTER_CORNER}_x", f"{config.LEFT_OUTER_CORNER}_y"]].to_numpy(dtype=np.float32)
    r_in = data_frame[[f"{config.RIGHT_INNER_CORNER}_x", f"{config.RIGHT_INNER_CORNER}_y"]].to_numpy(dtype=np.float32)
    r_out = data_frame[[f"{config.RIGHT_OUTER_CORNER}_x", f"{config.RIGHT_OUTER_CORNER}_y"]].to_numpy(dtype=np.float32)

    l_center_x = (l_in[:, [0]] + l_out[:, [0]]) / 2.0
    l_center_y = (l_in[:, [1]] + l_out[:, [1]]) / 2.0
    r_center_x = (r_in[:, [0]] + r_out[:, [0]]) / 2.0
    r_center_y = (r_in[:, [1]] + r_out[:, [1]]) / 2.0

    feat_left = np.empty((len(data_frame), len(config.LEFT_EYE_LANDMARKS) * 2), dtype=np.float32)
    feat_left[:, 0::2] = (l_xs - l_center_x) / config.SCALE_FACTOR
    feat_left[:, 1::2] = (l_ys - l_center_y) / config.SCALE_FACTOR

    feat_right = np.empty((len(data_frame), len(config.RIGHT_EYE_LANDMARKS) * 2), dtype=np.float32)
    feat_right[:, 0::2] = (r_xs - r_center_x) / config.SCALE_FACTOR
    feat_right[:, 1::2] = (r_ys - r_center_y) / config.SCALE_FACTOR

    relative_pos = np.empty((len(data_frame), 2), dtype=np.float32)
    relative_pos[:, 0] = ((r_center_x - l_center_x)[:, 0]) / config.SCALE_FACTOR
    relative_pos[:, 1] = ((r_center_y - l_center_y)[:, 0]) / config.SCALE_FACTOR

    head_x = data_frame[h_cols_x].to_numpy(dtype=np.float32)
    head_y = data_frame[h_cols_y].to_numpy(dtype=np.float32)
    face_center_x = (l_center_x + r_center_x) / 2.0
    face_center_y = (l_center_y + r_center_y) / 2.0

    feat_head = np.empty((len(data_frame), len(config.HEAD_ANCHORS) * 2), dtype=np.float32)
    feat_head[:, 0::2] = (head_x - face_center_x) / config.SCALE_FACTOR
    feat_head[:, 1::2] = (head_y - face_center_y) / config.SCALE_FACTOR

    targets = data_frame[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float32)
    targets /= np.clip(np.linalg.norm(targets, axis=1, keepdims=True), 1e-8, None)
    return feat_left, feat_right, relative_pos, feat_head, targets


def prepare_siamese_dataset(
    csv_path: str,
    config: Config,
    cache_dir_root: str,
    split_name: str,
    *,
    is_train: bool,
    rank: int,
    distributed: bool,
) -> tuple["SiameseGazeDataset", int, str]:
    cache_dir = build_cache_dir(
        csv_path,
        cache_dir_root,
        "siamese",
        (
            "v1",
            config.LEFT_EYE_LANDMARKS,
            config.RIGHT_EYE_LANDMARKS,
            config.HEAD_ANCHORS,
            config.SCALE_FACTOR,
        ),
    )
    array_names = ("feat_left", "feat_right", "relative_pos", "feat_head", "targets")

    if is_main_process(rank):
        if cache_ready(cache_dir, array_names):
            metadata = read_metadata(cache_dir)
            print(f"Using cached {split_name} tensors from {cache_dir} ({metadata['num_rows']} rows).")
        else:
            print(f"Loading {split_name} CSV: {csv_path}")
            data_frame = pd.read_csv(csv_path, sep=";")
            print(f"Loaded {len(data_frame)} {split_name} rows.")
            print(f"Precomputing {split_name} landmark features...")
            feat_left, feat_right, relative_pos, feat_head, targets = build_siamese_arrays(data_frame, config)
            del data_frame
            write_array_cache(
                cache_dir,
                {
                    "feat_left": feat_left,
                    "feat_right": feat_right,
                    "relative_pos": relative_pos,
                    "feat_head": feat_head,
                    "targets": targets,
                },
                {
                    "csv_path": os.path.abspath(csv_path),
                    "num_rows": int(targets.shape[0]),
                    "split_name": split_name,
                },
            )

    wait_for_all_ranks(distributed)

    if not cache_ready(cache_dir, array_names):
        raise RuntimeError(f"{split_name.capitalize()} cache is missing or incomplete at {cache_dir}.")

    metadata = read_metadata(cache_dir)
    arrays = load_array_cache(cache_dir, array_names)
    dataset = SiameseGazeDataset(
        config,
        is_train=is_train,
        feat_left=arrays["feat_left"],
        feat_right=arrays["feat_right"],
        relative_pos=arrays["relative_pos"],
        feat_head=arrays["feat_head"],
        targets=arrays["targets"],
    )
    return dataset, int(metadata["num_rows"]), cache_dir


class SiameseGazeDataset(Dataset):
    def __init__(
        self,
        config: Config,
        is_train: bool = False,
        *,
        data_frame: Optional[pd.DataFrame] = None,
        feat_left: Optional[np.ndarray] = None,
        feat_right: Optional[np.ndarray] = None,
        relative_pos: Optional[np.ndarray] = None,
        feat_head: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.is_train = is_train

        if any(value is None for value in (feat_left, feat_right, relative_pos, feat_head, targets)):
            if data_frame is None:
                raise ValueError("Either data_frame or all cached feature arrays must be provided.")
            feat_left, feat_right, relative_pos, feat_head, targets = build_siamese_arrays(data_frame, config)

        self.feat_left = feat_left
        self.feat_right = feat_right
        self.relative_pos = relative_pos
        self.feat_head = feat_head
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feat_left = self.feat_left[idx].copy() if self.is_train else self.feat_left[idx]
        feat_right = self.feat_right[idx].copy() if self.is_train else self.feat_right[idx]
        feat_head = self.feat_head[idx].copy() if self.is_train else self.feat_head[idx]
        relative_pos = self.relative_pos[idx]

        if self.is_train and self.config.AUGMENTATION_NOISE_STD > 0:
            noise_l = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, feat_left.shape).astype(np.float32)
            noise_r = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, feat_right.shape).astype(np.float32)
            noise_h = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, feat_head.shape).astype(np.float32)
            feat_left += noise_l
            feat_right += noise_r
            feat_head += noise_h

        target = self.targets[idx]

        return (
            torch.from_numpy(feat_left),
            torch.from_numpy(feat_right),
            torch.from_numpy(relative_pos),
            torch.from_numpy(feat_head),
            torch.from_numpy(target),
        )


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EyeEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int, dropout_rate: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout_rate) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return x


class SiameseGazeNet(nn.Module):
    def __init__(self, input_size_per_eye: int, config: Config):
        super().__init__()

        self.left_branch = EyeEncoder(
            input_size_per_eye,
            config.BRANCH_HIDDEN_WIDTH,
            config.NUM_BLOCKS,
            config.DROPOUT_RATE,
        )
        self.right_branch = EyeEncoder(
            input_size_per_eye,
            config.BRANCH_HIDDEN_WIDTH,
            config.NUM_BLOCKS,
            config.DROPOUT_RATE,
        )

        head_input_dim = len(config.HEAD_ANCHORS) * 2
        fusion_input_dim = config.BRANCH_HIDDEN_WIDTH * 2 + 2 + head_input_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_WIDTH),
            nn.BatchNorm1d(config.FUSION_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_HIDDEN_WIDTH, config.FUSION_HIDDEN_WIDTH // 2),
            nn.GELU(),
            nn.Linear(config.FUSION_HIDDEN_WIDTH // 2, 3),
        )

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        rel_pos: torch.Tensor,
        head_anchors: torch.Tensor,
    ) -> torch.Tensor:
        l_feat = self.left_branch(x_left)
        r_feat = self.right_branch(x_right)
        return self.fusion(torch.cat([l_feat, r_feat, rel_pos, head_anchors], dim=1))


class GazeAngularLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        cos_sim = torch.sum(pred * target, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_sim)
        return torch.mean(torch.rad2deg(theta))


def main() -> None:
    args = parse_args()
    if args.distributed and args.multi_gpu:
        raise ValueError("Use either --distributed (DDP) or --multi-gpu (DataParallel), not both.")

    config = Config()
    apply_overrides(config, args)

    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    device, distributed, rank, world_size, _local_rank = setup_runtime(args.device, args.distributed)
    amp_enabled = args.amp and device.type == "cuda"

    try:
        if is_main_process(rank):
            print(f"Using {device}")
            if distributed:
                print(f"DDP enabled across {world_size} GPUs.")
            if amp_enabled:
                print("AMP enabled.")

        if not os.path.exists(config.TRAIN_FILE) or not os.path.exists(config.VALID_FILE):
            raise FileNotFoundError(
                f"Data files not found at {config.TRAIN_FILE} or {config.VALID_FILE}"
            )

        train_dataset, train_rows, _train_cache_dir = prepare_siamese_dataset(
            config.TRAIN_FILE,
            config,
            args.cache_dir,
            "training",
            is_train=True,
            rank=rank,
            distributed=distributed,
        )
        valid_dataset, valid_rows, _valid_cache_dir = prepare_siamese_dataset(
            config.VALID_FILE,
            config,
            args.cache_dir,
            "validation",
            is_train=False,
            rank=rank,
            distributed=distributed,
        )
        if is_main_process(rank):
            print(f"Prepared {train_rows} training rows and {valid_rows} validation rows.")

        train_sampler = None
        valid_sampler = None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if is_main_process(rank):
            print("Data loaders ready.")
            if distributed:
                print(f"Waiting for all DDP ranks before model setup (timeout={get_ddp_timeout()}).")

        wait_for_all_ranks(distributed)

        input_dim_eye = len(config.LEFT_EYE_LANDMARKS) * 2
        model = SiameseGazeNet(input_dim_eye, config).to(device)
        model = maybe_parallelize(
            model,
            device=device,
            use_dataparallel=args.multi_gpu,
            use_distributed=distributed,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )
        criterion = GazeAngularLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        if is_main_process(rank):
            print("Starting Training (Siamese Vector Regression)...")
            print("\n--- Starting Training ---")

        best_error = float("inf")
        epochs_no_improve = 0
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

        for epoch in range(config.NUM_EPOCHS):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_loss_sum = 0.0
            train_sample_count = 0
            for x_l, x_r, x_rel, x_h, y_batch in train_loader:
                x_l = x_l.to(device, non_blocking=pin_memory)
                x_r = x_r.to(device, non_blocking=pin_memory)
                x_rel = x_rel.to(device, non_blocking=pin_memory)
                x_h = x_h.to(device, non_blocking=pin_memory)
                y_batch = y_batch.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    pred = model(x_l, x_r, x_rel, x_h)
                loss = criterion(pred.float(), y_batch.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = y_batch.size(0)
                train_loss_sum += loss.detach().item() * batch_size
                train_sample_count += batch_size

            train_loss_sum, train_sample_count = reduce_sum_and_count(
                train_loss_sum,
                train_sample_count,
                device=device,
                distributed=distributed,
            )

            model.eval()
            val_loss_sum = 0.0
            val_sample_count = 0
            with torch.no_grad():
                for x_l, x_r, x_rel, x_h, y_batch in valid_loader:
                    x_l = x_l.to(device, non_blocking=pin_memory)
                    x_r = x_r.to(device, non_blocking=pin_memory)
                    x_rel = x_rel.to(device, non_blocking=pin_memory)
                    x_h = x_h.to(device, non_blocking=pin_memory)
                    y_batch = y_batch.to(device, non_blocking=pin_memory)

                    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                        pred = model(x_l, x_r, x_rel, x_h)
                    loss = criterion(pred.float(), y_batch.float())
                    batch_size = y_batch.size(0)
                    val_loss_sum += loss.detach().item() * batch_size
                    val_sample_count += batch_size

            val_loss_sum, val_sample_count = reduce_sum_and_count(
                val_loss_sum,
                val_sample_count,
                device=device,
                distributed=distributed,
            )

            avg_train = train_loss_sum / max(train_sample_count, 1)
            avg_val = val_loss_sum / max(val_sample_count, 1)
            scheduler.step(avg_val)

            if is_main_process(rank):
                print(
                    f"Epoch {epoch + 1:03d} | Train Error: {avg_train:.2f} deg | Val Error: {avg_val:.2f} deg"
                )

            if avg_val < best_error:
                best_error = avg_val
                epochs_no_improve = 0
                if is_main_process(rank):
                    torch.save(unwrap_state_dict(model), config.MODEL_SAVE_PATH)
                    print(f"  > New best model saved! Val Error: {avg_val:.3f} deg")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config.PATIENCE:
                if is_main_process(rank):
                    print(
                        f"\nEarly stopping triggered at epoch {epoch + 1} after {config.PATIENCE} epochs with no improvement."
                    )
                break

        if is_main_process(rank):
            print("\n--- Training Complete ---")
            print(f"Best validation angular error: {best_error:.3f} deg")
            print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
    finally:
        cleanup_runtime(distributed)


if __name__ == "__main__":
    main()
