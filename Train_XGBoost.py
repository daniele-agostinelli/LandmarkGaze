import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # Added for saving the sklearn wrapper model

# --- USER CONFIGURATION ------------------------------------------------------
# Import config to get landmark indices and file paths
from Train_MLP import Config # We reuse existing Config

MODEL_NAME = 'gaze360_xgboost' #
VALID_FILE = 'datasets/Gaze360/gaze360_normalized_VAL.csv'  # 'datasets/XGaze_448/xgaze_normalized_det_conf_0_8_VALID.csv' # 'datasets/GazeGene/gazegene_normalized_det_conf_0_8_VALID.csv'
TRAIN_FILE = 'datasets/Gaze360/gaze360_normalized_TRAIN.csv'  # 'datasets/XGaze_448/xgaze_normalized_det_conf_0_8_TRAIN.csv' # 'datasets/GazeGene/gazegene_normalized_det_conf_0_8_TRAIN.csv'

# -----------------------------------------------------------------------------
def load_and_process_data(file_path, config):
    """
    Loads data and performs the exact same preprocessing as the Neural Network
    but returns Numpy arrays for XGBoost.
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, sep=';')

    # Generate column names based on the config indices
    lm_cols_x = [f"{idx}_x" for idx in config.LANDMARK_INDICES]
    lm_cols_y = [f"{idx}_y" for idx in config.LANDMARK_INDICES]
    eye_lm_cols_x = [f"{idx}_x" for idx in [config.LEFT_INNER_CORNER, config.LEFT_OUTER_CORNER, config.RIGHT_INNER_CORNER, config. RIGHT_OUTER_CORNER]]
    eye_lm_cols_y = [f"{idx}_y" for idx in [config.LEFT_INNER_CORNER, config.LEFT_OUTER_CORNER, config.RIGHT_INNER_CORNER, config. RIGHT_OUTER_CORNER]]

    # 1. Extract Raw Coordinates
    xs = df[lm_cols_x].values.astype(np.float32)
    ys = df[lm_cols_y].values.astype(np.float32)

    xs_eye = df[eye_lm_cols_x].values.astype(np.float32)
    ys_eye = df[eye_lm_cols_y].values.astype(np.float32)
        
    # 2. Relative Coordinates
    scale_factor = config.SCALE_FACTOR # Using the same scale factor as the NN for fair comparison
    # Calculate the centroid of the eyes
    centroid_x = np.mean(xs_eye)
    centroid_y = np.mean(ys_eye)
    xs_norm = (xs - centroid_x) / scale_factor
    ys_norm = (ys - centroid_y) / scale_factor
    #xs_norm = xs / scale_factor
    #ys_norm = ys / scale_factor
    
    # 3. Flatten/Interleave features
    num_samples = xs.shape[0]
    num_landmarks = xs.shape[1]

    # Create empty array of shape (N, Features)
    X = np.empty((num_samples, num_landmarks * 2), dtype=np.float32)
    X[:, 0::2] = xs_norm  # Even columns are X
    X[:, 1::2] = ys_norm  # Odd columns are Y

    # 4. Targets (Gaze Vector)
    y = df[['gaze_x', 'gaze_y', 'gaze_z']].values.astype(np.float32)

    # Ensure targets are unit vectors (normalization)
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / norms

    return X, y


def angular_error(pred, target):
    """
    Computes angular error in degrees between two sets of vectors.
    """
    # Normalize predictions (XGBoost doesn't strictly output unit vectors)
    pred_norm = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    target_norm = target / np.linalg.norm(target, axis=1, keepdims=True)

    # Dot product
    dot = np.sum(pred_norm * target_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    angles = np.arccos(dot)
    return np.degrees(angles)


def main():
    config = Config()

    # Define model save path
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"{MODEL_NAME}.pkl")

    print("--- XGBoost Benchmark & Feature Analysis ---")

    # 1. Prepare Data
    # 1.1. Load Training Data
    try:
        print("--- Loading Training Set ---")
        X_train, y_train = load_and_process_data(TRAIN_FILE, config)
        print(f"Training Samples: {X_train.shape[0]}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # 1.2. Load Valid Data
    try:
        print("\n--- Loading Valid Set ---")
        X_valid, y_valid = load_and_process_data(VALID_FILE, config)
        print(f"Testing Samples:  {X_valid.shape[0]}")
    except Exception as e:
        print(f"Error loading valid data: {e}")
        return

    # 2. Train XGBoost
    # We use MultiOutputRegressor because standard XGBoost predicts a single scalar.
    # This wraps 3 separate XGBoost models (one for x, one for y, one for z).
    print("\nTraining XGBoost Model (this may take a moment)...")

    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',  # Optimizing MSE
        'n_jobs': -1,
        'random_state': config.RANDOM_STATE,
        'tree_method': 'hist',  # Use histogram-based algorithm for speed
    }

    estimator = xgb.XGBRegressor(**xgb_params)
    model = MultiOutputRegressor(estimator)

    model.fit(X_train, y_train)

    # Save the model
    print(f"Saving model to {model_save_path}...")
    joblib.dump(model, model_save_path)

    # 3. Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_valid)

    # Compute Angular Error
    errors = angular_error(y_pred, y_valid)
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    print(f"XGBoost Mean Angular Error:   {mean_error:.4f} degrees")
    print(f"XGBoost Median Angular Error: {median_error:.4f} degrees")

    # 4. Feature Importance Analysis
    # MultiOutputRegressor has an 'estimators_' attribute containing the 3 trained models
    # We average the feature importance across X, Y, and Z predictors.

    feature_names = []
    for idx in config.LANDMARK_INDICES:
        feature_names.append(f"LM_{idx}_x")
        feature_names.append(f"LM_{idx}_y")

    # Get importances from each regressor
    # Feature importance type 'gain' is usually most informative
    importances_x = model.estimators_[0].feature_importances_
    importances_y = model.estimators_[1].feature_importances_
    importances_z = model.estimators_[2].feature_importances_

    # Average them
    avg_importance = (importances_x + importances_y + importances_z) / 3.0

    results_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Importance_Score': avg_importance
    })

    results_df = results_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("TOP 15 MOST IMPORTANT FEATURES (XGBoost)")
    print("=" * 50)
    print(results_df.head(15).to_string(index=False))

    # Save Results
    output_csv = "models/stats/xgboost_feature_importance_"+MODEL_NAME+".csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nFull analysis saved to: {output_csv}")

    # Plot
    try:
        plt.figure(figsize=(10, 8))
        top_plot = results_df.head(20)
        sns.barplot(x='Importance_Score', y='Feature_Name', data=top_plot, palette='magma')
        plt.title('Top 20 Features (XGBoost Importance)')
        plt.xlabel('Importance Score (Gain)')
        plt.tight_layout()
        plt.savefig(f'models/stats/xgboost_feature_importance_{MODEL_NAME}.png')
        print("Plot saved to models/stats/xgboost_feature_importance_"+MODEL_NAME+".png")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
