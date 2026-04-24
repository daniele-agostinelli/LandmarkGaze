import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset_registry import get_dataset_spec


DEFAULT_DATASET = get_dataset_spec("gaze360")


class Config:
    TRAIN_FILE = DEFAULT_DATASET.train_file
    VALID_FILE = DEFAULT_DATASET.valid_file
    MODEL_SAVE_PATH = "models/gaze360_MLP.pth"

    HIDDEN_WIDTH = 256
    NUM_BLOCKS = 3
    DROPOUT_RATE = 0.1

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 15
    RANDOM_STATE = 42

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4
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
    LANDMARK_INDICES = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + HEAD_ANCHORS

    SCALE_FACTOR = 448.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train landmark MLP gaze regressor.")
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hidden-width", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--augmentation-noise-std", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.dataset:
        spec = get_dataset_spec(args.dataset)
        if spec.train_file is None or spec.valid_file is None:
            raise ValueError(f"Dataset '{spec.key}' does not define default TRAIN/VALID files.")
        config.TRAIN_FILE = spec.train_file
        config.VALID_FILE = spec.valid_file
        config.MODEL_SAVE_PATH = f"models/{spec.key}_MLP.pth"

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
    if args.hidden_width is not None:
        config.HIDDEN_WIDTH = args.hidden_width
    if args.num_blocks is not None:
        config.NUM_BLOCKS = args.num_blocks
    if args.dropout_rate is not None:
        config.DROPOUT_RATE = args.dropout_rate
    if args.augmentation_noise_std is not None:
        config.AUGMENTATION_NOISE_STD = args.augmentation_noise_std


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def maybe_dataparallel(model: nn.Module, device: torch.device, enabled: bool) -> nn.Module:
    if enabled and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"DataParallel enabled across {torch.cuda.device_count()} GPUs.")
        return nn.DataParallel(model)
    return model


def unwrap_state_dict(model: nn.Module):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


class GazeDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, config: Config, is_train: bool = False):
        self.config = config
        self.is_train = is_train

        lm_cols_x = [f"{idx}_x" for idx in config.LANDMARK_INDICES]
        lm_cols_y = [f"{idx}_y" for idx in config.LANDMARK_INDICES]
        eye_anchor_ids = [
            config.LEFT_INNER_CORNER,
            config.LEFT_OUTER_CORNER,
            config.RIGHT_INNER_CORNER,
            config.RIGHT_OUTER_CORNER,
        ]
        eye_lm_cols_x = [f"{idx}_x" for idx in eye_anchor_ids]
        eye_lm_cols_y = [f"{idx}_y" for idx in eye_anchor_ids]

        xs = data_frame[lm_cols_x].to_numpy(dtype=np.float32)
        ys = data_frame[lm_cols_y].to_numpy(dtype=np.float32)
        xs_eye = data_frame[eye_lm_cols_x].to_numpy(dtype=np.float32)
        ys_eye = data_frame[eye_lm_cols_y].to_numpy(dtype=np.float32)

        centroid_x = np.mean(xs_eye, axis=1, keepdims=True)
        centroid_y = np.mean(ys_eye, axis=1, keepdims=True)
        xs_norm = (xs - centroid_x) / config.SCALE_FACTOR
        ys_norm = (ys - centroid_y) / config.SCALE_FACTOR

        self.features = np.empty((len(data_frame), len(config.LANDMARK_INDICES) * 2), dtype=np.float32)
        self.features[:, 0::2] = xs_norm
        self.features[:, 1::2] = ys_norm

        self.targets = data_frame[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float32)
        self.targets /= np.clip(np.linalg.norm(self.targets, axis=1, keepdims=True), 1e-8, None)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx].copy() if self.is_train else self.features[idx]

        if self.is_train and self.config.AUGMENTATION_NOISE_STD > 0:
            noise = np.random.normal(
                0,
                self.config.AUGMENTATION_NOISE_STD,
                features.shape,
            ).astype(np.float32)
            features += noise

        target = self.targets[idx]
        return torch.from_numpy(features), torch.from_numpy(target)


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


class GazeNetVector(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        hidden_width: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_width),
            nn.BatchNorm1d(hidden_width),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_width, dropout_rate) for _ in range(num_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Linear(hidden_width // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_head(x)


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
    config = Config()
    apply_overrides(config, args)

    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    device = resolve_device(args.device)
    print(f"Using {device}")

    if not os.path.exists(config.TRAIN_FILE) or not os.path.exists(config.VALID_FILE):
        raise FileNotFoundError(
            f"Data files not found at {config.TRAIN_FILE} or {config.VALID_FILE}"
        )

    print(f"Loading training CSV: {config.TRAIN_FILE}")
    train_df = pd.read_csv(config.TRAIN_FILE, sep=";")
    print(f"Loading validation CSV: {config.VALID_FILE}")
    valid_df = pd.read_csv(config.VALID_FILE, sep=";")
    print(f"Loaded {len(train_df)} training rows and {len(valid_df)} validation rows.")

    print("Precomputing normalized landmark features...")
    train_dataset = GazeDataset(train_df, config, True)
    valid_dataset = GazeDataset(valid_df, config, False)
    del train_df
    del valid_df

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print("Data loaders ready.")

    input_dim = len(config.LANDMARK_INDICES) * 2
    model = GazeNetVector(
        input_dim,
        output_size=3,
        hidden_width=config.HIDDEN_WIDTH,
        num_blocks=config.NUM_BLOCKS,
        dropout_rate=config.DROPOUT_RATE,
    ).to(device)
    model = maybe_dataparallel(model, device, args.multi_gpu)

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

    print("Starting Training (Vector Regression)...")
    best_error = float("inf")
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device, non_blocking=pin_memory)
                y_batch = y_batch.to(device, non_blocking=pin_memory)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch + 1:03d} | Train Error: {avg_train:.2f} deg | Val Error: {avg_val:.2f} deg")
        scheduler.step(avg_val)

        if avg_val < best_error:
            best_error = avg_val
            epochs_no_improve = 0
            torch.save(unwrap_state_dict(model), config.MODEL_SAVE_PATH)
            print(f"  > New best model saved! Val Error: {avg_val:.3f} deg")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1} after {config.PATIENCE} epochs with no improvement."
            )
            break

    print("\n--- Training Complete ---")
    print(f"Best validation angular error: {best_error:.3f} deg")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
