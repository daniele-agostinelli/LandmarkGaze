import numpy as np
import pandas as pd
import torch, os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# --- Configuration -----------------------------------------------------------
class Config:
    VALID_FILE = 'datasets/Gaze360/gaze360_normalized_VAL.csv' #'datasets/XGaze_448/xgaze_normalized_det_conf_0_8_VALID.csv' # 'datasets/GazeGene/gazegene_normalized_det_conf_0_8_VALID.csv'
    TRAIN_FILE = 'datasets/Gaze360/gaze360_normalized_TRAIN.csv' #'datasets/XGaze_448/xgaze_normalized_det_conf_0_8_TRAIN.csv' # 'datasets/GazeGene/gazegene_normalized_det_conf_0_8_TRAIN.csv'
    MODEL_SAVE_PATH = 'models/gaze360_siameseMLP.pth' # 'models/xgaze_siameseMLP.pth' # 'models/gazegene_siameseMLP.pth'

    # Model Hyperparameters
    BRANCH_HIDDEN_WIDTH = 64
    FUSION_HIDDEN_WIDTH = 128
    NUM_BLOCKS = 3  # Residual blocks per branch
    DROPOUT_RATE = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 15 # For early stopping
    RANDOM_STATE = 42

    # Scheduler setting
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Augmentation
    AUGMENTATION_NOISE_STD = 0.0

    # Landmark Indices
    # 1. Iris (Center + 4 circumference)
    LEFT_IRIS  = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    # 2. Contours
    LEFT_INNER_CORNER = 133
    LEFT_OUTER_CORNER = 33
    LEFT_EYE_CONTOUR  = [LEFT_OUTER_CORNER, LEFT_INNER_CORNER, 159, 145] 
    RIGHT_INNER_CORNER = 362
    RIGHT_OUTER_CORNER = 263
    RIGHT_EYE_CONTOUR  = [RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, 386, 374] 
    # 3. Head Anchors
    HEAD_ANCHORS = [1,9]

    LEFT_EYE_LANDMARKS = LEFT_IRIS + LEFT_EYE_CONTOUR
    #[LEFT_OUTER_CORNER, 246, 161, 160, 159, 158, 157, 173, LEFT_INNER_CORNER, 155, 154, 153, 145, 144,
    #163, 7, 468, 469, 470, 471, 472, 27, 190, 243, 233, 232, 230, 31, 25, 110, 113, 247, 225]
    RIGHT_EYE_LANDMARKS = RIGHT_IRIS + RIGHT_EYE_CONTOUR
    #[475, 473, 474, 476, 477, RIGHT_INNER_CORNER, 398, 382, 381, 380, 374, 373, 390, 249,
    #RIGHT_OUTER_CORNER, 466, 388, 387, 386, 385, 384, 257, 445, 342, 467, 255, 339, 450, 452, 453, 463, 414, 261]

    #LANDMARK_INDICES = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + HEAD_ANCHORS

    SCALE_FACTOR = 448.0 # based on normalized camera


# --- Siamese Dataset ---------------------------------------------------------
class SiameseGazeDataset(Dataset):
    def __init__(self, data_frame, config, is_train=False):
        self.data_frame = data_frame
        self.config = config
        self.is_train = is_train

        # Pre-compute column names for fast access
        self.l_cols_x = [f"{idx}_x" for idx in config.LEFT_EYE_LANDMARKS]
        self.l_cols_y = [f"{idx}_y" for idx in config.LEFT_EYE_LANDMARKS]

        self.r_cols_x = [f"{idx}_x" for idx in config.RIGHT_EYE_LANDMARKS]
        self.r_cols_y = [f"{idx}_y" for idx in config.RIGHT_EYE_LANDMARKS]

        self.h_cols_x = [f"{idx}_x" for idx in config.HEAD_ANCHORS]
        self.h_cols_y = [f"{idx}_y" for idx in config.HEAD_ANCHORS]

        # Corner Indices for stable centering
        # We need the column names for the specific corner landmarks
        self.l_in_x = f"{config.LEFT_INNER_CORNER}_x"
        self.l_in_y = f"{config.LEFT_INNER_CORNER}_y"
        self.l_out_x = f"{config.LEFT_OUTER_CORNER}_x"
        self.l_out_y = f"{config.LEFT_OUTER_CORNER}_y"

        self.r_in_x = f"{config.RIGHT_INNER_CORNER}_x"
        self.r_in_y = f"{config.RIGHT_INNER_CORNER}_y"
        self.r_out_x = f"{config.RIGHT_OUTER_CORNER}_x"
        self.r_out_y = f"{config.RIGHT_OUTER_CORNER}_y"

    def __len__(self):
        return len(self.data_frame)

    def _process_eye(self, xs, ys, inner_pt, outer_pt):
        """
        Normalizes a single eye's landmarks relative to eye corners.
        """
        # Calculate local centroid (center of the eye)
        centroid_x = (inner_pt[0] + outer_pt[0]) / 2.0
        centroid_y = (inner_pt[1] + outer_pt[1]) / 2.0

        scale_factor = self.config.SCALE_FACTOR

        # Normalize: Center and scale
        xs_norm = (xs - centroid_x) / scale_factor
        ys_norm = (ys - centroid_y) / scale_factor

        # Interleave features [x1, y1, x2, y2...]
        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm
        features[1::2] = ys_norm

        return features, (centroid_x, centroid_y), scale_factor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]

        # 1. Load Raw Coordinates
        l_xs = row[self.l_cols_x].values.astype(np.float32)
        l_ys = row[self.l_cols_y].values.astype(np.float32)

        r_xs = row[self.r_cols_x].values.astype(np.float32)
        r_ys = row[self.r_cols_y].values.astype(np.float32)

        # --- Get Corner Coordinates for Anchoring ---
        l_in = (row[self.l_in_x], row[self.l_in_y])
        l_out = (row[self.l_out_x], row[self.l_out_y])
        r_in = (row[self.r_in_x], row[self.r_in_y])
        r_out = (row[self.r_out_x], row[self.r_out_y])

        # 2. Process Independently (Siamese Approach)
        # This removes the relative position of the eyes on the face,
        # forcing the model to learn from the shape of the eye itself.
        feat_left, l_center, l_scale = self._process_eye(l_xs, l_ys, l_in, l_out)
        feat_right, r_center, r_scale = self._process_eye(r_xs, r_ys, r_in, r_out)

        # --- Calculate Relative Position Vector ---
        # Vector from Left Eye Center to Right Eye Center
        # Normalized by the average scale of the eyes to keep units consistent
        avg_scale = (l_scale + r_scale) / 2.0
        delta_x = (r_center[0] - l_center[0]) / avg_scale
        delta_y = (r_center[1] - l_center[1]) / avg_scale
        
        relative_pos = np.array([delta_x, delta_y], dtype=np.float32)

        # Head anchors
        H_x = row[self.h_cols_x].values.astype(np.float32)
        H_y = row[self.h_cols_y].values.astype(np.float32)
        Hx_norm = (H_x - (l_center[0]+r_center[0])/2.) / avg_scale
        Hy_norm = (H_y - (l_center[1]+r_center[1])/2.) / avg_scale

        # Interleave features [x1, y1, x2, y2...]
        feat_head = np.empty((len(H_x) * 2,), dtype=np.float32)
        feat_head[0::2] = Hx_norm
        feat_head[1::2] = Hy_norm

        # 3. Add Noise (Augmentation)
        if self.is_train and self.config.AUGMENTATION_NOISE_STD > 0:
            noise_l = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, feat_left.shape).astype(np.float32)
            noise_r = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, feat_right.shape).astype(np.float32)
            noise_h = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, feat_head.shape).astype(np.float32)
            feat_left += noise_l
            feat_right += noise_r
            feat_head += noise_h

        # 4. Target
        target = np.array([row['gaze_x'], row['gaze_y'], row['gaze_z']], dtype=np.float32)
        target /= np.linalg.norm(target)

        return torch.tensor(feat_left), torch.tensor(feat_right), torch.tensor(relative_pos), torch.tensor(feat_head), torch.tensor(target)


# --- Siamese Model -----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


class EyeEncoder(nn.Module):
    """Encodes a single eye's landmarks into a feature vector."""

    def __init__(self, input_size, hidden_size, num_blocks, dropout_rate):
        super(EyeEncoder, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return x


class SiameseGazeNet(nn.Module):
    def __init__(self, input_size_per_eye, config):
        super(SiameseGazeNet, self).__init__()

        # Branch 1: Left Eye
        self.left_branch = EyeEncoder(
            input_size_per_eye,
            config.BRANCH_HIDDEN_WIDTH,
            config.NUM_BLOCKS,
            config.DROPOUT_RATE
        )

        # Branch 2: Right Eye
        # We use a separate encoder (weights not shared) to allow for
        # asymmetries or consistent differences in landmark estimation.
        self.right_branch = EyeEncoder(
            input_size_per_eye,
            config.BRANCH_HIDDEN_WIDTH,
            config.NUM_BLOCKS,
            config.DROPOUT_RATE
        )

        # Fusion Head
        # Concatenates output of both branches: [Left_Features, Right_Features, Relative_Pos_X, Relative_Pos_Y, head_anchors]
        head_input_dim = len(config.HEAD_ANCHORS)*2
        fusion_input_dim = config.BRANCH_HIDDEN_WIDTH * 2 + 2 + head_input_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_WIDTH),
            nn.BatchNorm1d(config.FUSION_HIDDEN_WIDTH),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_HIDDEN_WIDTH, config.FUSION_HIDDEN_WIDTH // 2),
            nn.GELU(),
            nn.Linear(config.FUSION_HIDDEN_WIDTH // 2, 3)  # Output: x, y, z
        )

    def forward(self, x_left, x_right, rel_pos, head_anchors):
        l_feat = self.left_branch(x_left)
        r_feat = self.right_branch(x_right)

        # Fuse
        combined = torch.cat([l_feat, r_feat, rel_pos, head_anchors], dim=1)
        output = self.fusion(combined)
        return output


# --- Loss Function -----------------------------------------------------------
class GazeAngularLoss(nn.Module):
    def __init__(self):
        super(GazeAngularLoss, self).__init__()

    def forward(self, pred, target):
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        cos_sim = torch.sum(pred * target, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_sim)
        return torch.mean(torch.rad2deg(theta))


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    config = Config()

    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Load Data
    if not os.path.exists(config.TRAIN_FILE) or not os.path.exists(config.VALID_FILE):
        print(f"Error: Data files not found at {config.TRAIN_FILE} or {config.VALID_FILE}")
        exit(1)

    train_df = pd.read_csv(config.TRAIN_FILE, sep=';')
    valid_df = pd.read_csv(config.VALID_FILE, sep=';')
    
    # Use the Siamese Dataset
    train_dataset = SiameseGazeDataset(train_df, config, True)
    valid_dataset = SiameseGazeDataset(valid_df, config, False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model Initialization
    # Input dim per eye is number of landmarks * 2 (x,y)
    input_dim_eye = len(config.LEFT_EYE_LANDMARKS) * 2
    model = SiameseGazeNet(input_dim_eye, config).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
    criterion = GazeAngularLoss()

    print("Starting Training (Siamese Vector Regression)...")

    best_error = float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        # TRAIN
        model.train()
        train_losses = []
        for X_l, X_r, X_rel, X_h, y in train_loader:
            X_l, X_r, X_rel, X_h, y = X_l.to(device), X_r.to(device), X_rel.to(device), X_h.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X_l, X_r, X_rel, X_h)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # VAL
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_l, X_r, X_rel, X_h, y in valid_loader:
                X_l, X_r, X_rel, X_h, y = X_l.to(device), X_r.to(device), X_rel.to(device), X_h.to(device), y.to(device)
                pred = model(X_l, X_r, X_rel, X_h)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        print(f"Epoch {epoch + 1:03d} | Train Error: {avg_train:.2f}° | Val Error: {avg_val:.2f}°")

        scheduler.step(avg_val)

        if avg_val < best_error:
            best_error = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  > New best model saved! Val Error (deg): {avg_val:.3f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(
                f"\nEarly stopping triggered at epoch {epoch + 1} after {config.PATIENCE} epochs with no improvement.")
            break

    print("\n--- Training Complete ---")
    print(f"Best validation angular error: {best_error:.3f} degrees")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
