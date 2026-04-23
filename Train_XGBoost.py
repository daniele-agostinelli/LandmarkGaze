import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from Train_MLP import Config
from dataset_registry import get_dataset_spec


def parse_args() -> argparse.Namespace:
    gaze360_spec = get_dataset_spec("gaze360")
    parser = argparse.ArgumentParser(description="Train XGBoost gaze regressor on landmark CSV.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset alias from dataset_registry.py (e.g. gaze360, gazegene, xgaze, blender).",
    )
    parser.add_argument(
        "--train-file",
        default=gaze360_spec.train_file,
        help="Training CSV (semicolon-separated).",
    )
    parser.add_argument(
        "--valid-file",
        default=gaze360_spec.valid_file,
        help="Validation CSV (semicolon-separated).",
    )
    parser.add_argument("--model-name", default="gaze360_xgboost")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--stats-dir", default="models/stats")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _safe_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return vectors / norms


def load_and_process_data(file_path: str, config: Config):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, sep=";")

    lm_cols_x = [f"{idx}_x" for idx in config.LANDMARK_INDICES]
    lm_cols_y = [f"{idx}_y" for idx in config.LANDMARK_INDICES]
    eye_lm_cols_x = [
        f"{idx}_x"
        for idx in [
            config.LEFT_INNER_CORNER,
            config.LEFT_OUTER_CORNER,
            config.RIGHT_INNER_CORNER,
            config.RIGHT_OUTER_CORNER,
        ]
    ]
    eye_lm_cols_y = [
        f"{idx}_y"
        for idx in [
            config.LEFT_INNER_CORNER,
            config.LEFT_OUTER_CORNER,
            config.RIGHT_INNER_CORNER,
            config.RIGHT_OUTER_CORNER,
        ]
    ]

    xs = df[lm_cols_x].values.astype(np.float32)
    ys = df[lm_cols_y].values.astype(np.float32)
    xs_eye = df[eye_lm_cols_x].values.astype(np.float32)
    ys_eye = df[eye_lm_cols_y].values.astype(np.float32)

    centroid_x = np.mean(xs_eye, axis=1, keepdims=True)
    centroid_y = np.mean(ys_eye, axis=1, keepdims=True)
    xs_norm = (xs - centroid_x) / config.SCALE_FACTOR
    ys_norm = (ys - centroid_y) / config.SCALE_FACTOR

    num_samples = xs.shape[0]
    num_landmarks = xs.shape[1]
    x_features = np.empty((num_samples, num_landmarks * 2), dtype=np.float32)
    x_features[:, 0::2] = xs_norm
    x_features[:, 1::2] = ys_norm

    y_targets = df[["gaze_x", "gaze_y", "gaze_z"]].values.astype(np.float32)
    y_targets = _safe_normalize(y_targets)
    return x_features, y_targets


def angular_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_norm = _safe_normalize(pred)
    target_norm = _safe_normalize(target)
    dot = np.sum(pred_norm * target_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _resolve_xgb_device(arg_device: str) -> str:
    if arg_device == "auto":
        return "cuda" if xgb.build_info().get("USE_CUDA", False) else "cpu"
    return arg_device


def main() -> None:
    args = parse_args()
    config = Config()

    train_file = args.train_file
    valid_file = args.valid_file
    model_name = args.model_name

    if args.dataset:
        spec = get_dataset_spec(args.dataset)
        if spec.train_file is None or spec.valid_file is None:
            raise ValueError(f"Dataset '{spec.key}' does not define default TRAIN/VALID files.")
        train_file = spec.train_file
        valid_file = spec.valid_file
        model_name = f"{spec.key}_xgboost"

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_dir, f"{model_name}.pkl")

    print("--- XGBoost Benchmark & Feature Analysis ---")

    try:
        print("--- Loading Training Set ---")
        x_train, y_train = load_and_process_data(train_file, config)
        print(f"Training Samples: {x_train.shape[0]}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    try:
        print("\n--- Loading Valid Set ---")
        x_valid, y_valid = load_and_process_data(valid_file, config)
        print(f"Validation Samples: {x_valid.shape[0]}")
    except Exception as e:
        print(f"Error loading valid data: {e}")
        return

    print("\nTraining XGBoost Model (this may take a while)...")

    selected_device = _resolve_xgb_device(args.xgb_device)
    xgb_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": "reg:squarederror",
        "n_jobs": args.n_jobs,
        "random_state": config.RANDOM_STATE,
        "tree_method": "hist",
    }
    if selected_device == "cuda":
        xgb_params["device"] = "cuda"

    model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
    try:
        model.fit(x_train, y_train)
    except Exception as e:
        if selected_device != "cuda":
            raise
        print(f"CUDA training failed ({e}), retrying on CPU.")
        xgb_params.pop("device", None)
        model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
        model.fit(x_train, y_train)

    print(f"Saving model to {model_save_path}...")
    joblib.dump(model, model_save_path)

    print("\nEvaluating...")
    y_pred = model.predict(x_valid)
    errors = angular_error(y_pred, y_valid)
    print(f"XGBoost Mean Angular Error:   {np.mean(errors):.4f} degrees")
    print(f"XGBoost Median Angular Error: {np.median(errors):.4f} degrees")

    feature_names = []
    for idx in config.LANDMARK_INDICES:
        feature_names.append(f"LM_{idx}_x")
        feature_names.append(f"LM_{idx}_y")

    importances_x = model.estimators_[0].feature_importances_
    importances_y = model.estimators_[1].feature_importances_
    importances_z = model.estimators_[2].feature_importances_
    avg_importance = (importances_x + importances_y + importances_z) / 3.0

    results_df = pd.DataFrame(
        {
            "Feature_Name": feature_names,
            "Importance_Score": avg_importance,
        }
    ).sort_values(by="Importance_Score", ascending=False, ignore_index=True)

    print("\n" + "=" * 50)
    print("TOP 15 MOST IMPORTANT FEATURES (XGBoost)")
    print("=" * 50)
    print(results_df.head(15).to_string(index=False))

    output_csv = os.path.join(args.stats_dir, f"xgboost_feature_importance_{model_name}.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\nFull analysis saved to: {output_csv}")

    try:
        plt.figure(figsize=(10, 8))
        top_plot = results_df.head(20)
        sns.barplot(x="Importance_Score", y="Feature_Name", data=top_plot, palette="magma")
        plt.title("Top 20 Features (XGBoost Importance)")
        plt.xlabel("Importance Score (Gain)")
        plt.tight_layout()
        plot_path = os.path.join(args.stats_dir, f"xgboost_feature_importance_{model_name}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
