import os
from pathlib import Path
from typing import Iterable, List, Optional

from omegaconf import OmegaConf

from Test_on_Gaze360 import GazeBenchmarkFromCSV
from Test_on_GazeGene import GazeGeneBenchmarkFromCSV
from Test_on_XGaze import CSVBenchmarkRunner
from dataset_registry import default_cross_domain_targets, get_dataset_spec, normalize_dataset_key


def _validate_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _make_config(config_path: str, device: str, model_path: str):
    cfg = OmegaConf.load(config_path)
    cfg.device = device
    cfg.gaze_estimator.model_path = model_path

    cam_params = cfg.gaze_estimator.get("camera_params", None)
    if cam_params is None:
        norm_cam = cfg.gaze_estimator.get("normalized_camera_params", None)
        if norm_cam is None:
            raise ValueError(
                "Both gaze_estimator.camera_params and normalized_camera_params are missing."
            )
        cfg.gaze_estimator.camera_params = norm_cam
    return cfg


def resolve_targets(source_dataset: Optional[str], targets: Optional[Iterable[str]]) -> List[str]:
    if targets:
        return [normalize_dataset_key(name) for name in targets]
    return default_cross_domain_targets(source_dataset)


def run_cross_domain_validation(
    *,
    model_path: str,
    model_type: str,
    config_path: str,
    device: str,
    output_dir: str,
    tag: str,
    source_dataset: Optional[str] = None,
    targets: Optional[Iterable[str]] = None,
) -> None:
    model_path = str(Path(model_path).expanduser().resolve())
    config_path = str(Path(config_path).expanduser().resolve())
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    _validate_file(model_path, "Model file")
    _validate_file(config_path, "Config file")

    resolved_targets = resolve_targets(source_dataset, targets)
    if not resolved_targets:
        raise ValueError("No target datasets selected for cross-domain validation.")

    summary_csv = str(output_dir_path / f"summary_cross_domain_{tag}.csv")

    print("=== Cross-Domain Validation ===")
    print(f"Model:  {model_path}")
    print(f"Type:   {model_type}")
    print(f"Device: {device}")
    print(f"Targets: {', '.join(resolved_targets)}")
    print(f"Summary CSV: {summary_csv}")

    for index, dataset_key in enumerate(resolved_targets, start=1):
        spec = get_dataset_spec(dataset_key)
        _validate_file(spec.all_file, f"{spec.label} CSV")
        cfg = _make_config(config_path, device, model_path)
        row_output = str(output_dir_path / f"{dataset_key}_{tag}_rows.csv")

        print(f"\n[{index}/{len(resolved_targets)}] Running {spec.label} -> {row_output}")

        if dataset_key == "gaze360":
            bench = GazeBenchmarkFromCSV(
                config=cfg,
                model_type=model_type,
                output_csv_path=row_output,
                summary_csv_path=summary_csv,
            )
            bench.run_benchmark(spec.all_file)
        elif dataset_key == "gazegene":
            bench = GazeGeneBenchmarkFromCSV(
                config=cfg,
                model_type=model_type,
                output_csv_path=row_output,
                summary_csv_path=summary_csv,
            )
            bench.run_benchmark(spec.all_file)
        elif dataset_key == "xgaze":
            runner = CSVBenchmarkRunner(
                config=cfg,
                model_def={
                    "name": Path(model_path).stem,
                    "path": model_path,
                    "type": model_type,
                },
                output_csv_path=row_output,
                summary_csv_path=summary_csv,
            )
            runner.run_benchmark(spec.all_file)
        else:
            raise ValueError(f"Unsupported target dataset: {dataset_key}")

    print("\nDone.")
    print(f"Summary: {summary_csv}")
