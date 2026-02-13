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
    MODEL_SAVE_PATH = 'models/gaze360_MLP.pth' # 'models/xgaze_MLP.pth' # 'models/gazegene_MLP.pth'

    # Model Hyperparameters
    HIDDEN_WIDTH = 256
    NUM_BLOCKS   = 3
    DROPOUT_RATE = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 15  # For early stopping
    RANDOM_STATE = 42

    # Scheduler setting
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4

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
    LANDMARK_INDICES = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + HEAD_ANCHORS

    SCALE_FACTOR = 448.0 # based on normalized camera


# --- Improved Dataset --------------------------------------------------------
class GazeDataset(Dataset):
    def __init__(self, data_frame, config, is_train=False):
        self.data_frame = data_frame
        self.config = config
        self.is_train = is_train

        # Pre-compute indices for fast access
        self.lm_cols_x = [f"{idx}_x" for idx in config.LANDMARK_INDICES]
        self.lm_cols_y = [f"{idx}_y" for idx in config.LANDMARK_INDICES]
        self.eye_lm_cols_x = [f"{idx}_x" for idx in [config.LEFT_INNER_CORNER, config.LEFT_OUTER_CORNER, config.RIGHT_INNER_CORNER, config. RIGHT_OUTER_CORNER]]
        self.eye_lm_cols_y = [f"{idx}_y" for idx in [config.LEFT_INNER_CORNER, config.LEFT_OUTER_CORNER, config.RIGHT_INNER_CORNER, config. RIGHT_OUTER_CORNER]]


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # 1. Load Raw Coordinates
        xs = row[self.lm_cols_x].values.astype(np.float32)
        ys = row[self.lm_cols_y].values.astype(np.float32)
        xs_eye = row[self.eye_lm_cols_x].values.astype(np.float32)
        ys_eye = row[self.eye_lm_cols_y].values.astype(np.float32)
        
        # 2. Relative Coordinates
        scale_factor = self.config.SCALE_FACTOR
        # Calculate the centroid of the face for this specific frame
        centroid_x = np.mean(xs_eye)
        centroid_y = np.mean(ys_eye)
        xs_norm = (xs - centroid_x) / scale_factor
        ys_norm = (ys - centroid_y) / scale_factor
        #xs_norm = xs / scale_factor
        #ys_norm = ys / scale_factor

        # Interleave x and y: [x1, y1, x2, y2, ...]
        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm
        features[1::2] = ys_norm

        # 3. Add Noise (Augmentation)
        if self.is_train and self.config.AUGMENTATION_NOISE_STD > 0:
            noise = np.random.normal(0, self.config.AUGMENTATION_NOISE_STD, features.shape).astype(np.float32)
            features += noise

        # 4. Target: Use 3D Vector
        target = np.array([row['gaze_x'], row['gaze_y'], row['gaze_z']], dtype=np.float32)

        # Normalize target
        target /= np.linalg.norm(target)

        return torch.tensor(features), torch.tensor(target)


# --- Improved Model ----------------------------------------------------------
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


class GazeNetVector(nn.Module):
    def __init__(self, input_size, output_size=3, hidden_width=256, num_blocks=3, dropout_rate=0.0):
        super(GazeNetVector, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_width),
            nn.BatchNorm1d(hidden_width),
            nn.GELU()
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_width, dropout_rate) for _ in range(num_blocks)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Linear(hidden_width // 2, output_size)  # Output is 3 (x,y,z)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_head(x)
        return x


# --- Loss Function -----------------------------------------------------------
class GazeAngularLoss(nn.Module):
    def __init__(self):
        super(GazeAngularLoss, self).__init__()

    def forward(self, pred, target):
        # Normalize vectors to ensure they are unit vectors
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)

        # Cosine similarity: range [-1, 1]
        cos_sim = torch.sum(pred * target, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_sim)

        # Return mean angle in degrees (easier to interpret logs)
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
    train_df = pd.read_csv(config.TRAIN_FILE, sep=';')
    valid_df = pd.read_csv(config.VALID_FILE, sep=';')

    train_loader = DataLoader(GazeDataset(train_df, config, True), batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(GazeDataset(valid_df, config, False), batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=0)

    # Model
    input_dim = len(config.LANDMARK_INDICES) * 2
    model = GazeNetVector(input_dim, output_size=3, hidden_width=config.HIDDEN_WIDTH, num_blocks=config.NUM_BLOCKS, dropout_rate=config.DROPOUT_RATE).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
    criterion = GazeAngularLoss()

    print("Starting Training (Vector Regression)...")

    best_error = float('inf')
    epochs_no_improve = 0

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(config.NUM_EPOCHS):
        # TRAIN
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)  # Loss is directly the angular error in degrees
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # VAL
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        print(f"Epoch {epoch + 1:03d} | Train Error: {avg_train:.2f}° | Val Error: {avg_val:.2f}°")

        scheduler.step(avg_val)

        if avg_val < best_error:         # Early Stopping Check
            best_error = avg_val
            epochs_no_improve = 0

            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  > New best model saved! Val Error (deg): {avg_val:.3f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1} after {config.PATIENCE} epochs with no improvement.")
            break

    print("\n--- Training Complete ---")
    print(f"Best validation angular error: {best_error:.3f} degrees")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")
