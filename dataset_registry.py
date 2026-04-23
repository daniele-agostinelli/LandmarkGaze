from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    train_file: Optional[str]
    valid_file: Optional[str]
    test_file: Optional[str]
    all_file: str


_DATASETS: Dict[str, DatasetSpec] = {
    "gaze360": DatasetSpec(
        key="gaze360",
        label="Gaze360",
        train_file="datasets/Gaze360/gaze360_normalized_with_lm168_TRAIN.csv",
        valid_file="datasets/Gaze360/gaze360_normalized_with_lm168_VAL.csv",
        test_file="datasets/Gaze360/gaze360_normalized_with_lm168_TEST.csv",
        all_file="datasets/Gaze360/gaze360_normalized_with_lm168_ALL_USED.csv",
    ),
    "gazegene": DatasetSpec(
        key="gazegene",
        label="GazeGene",
        train_file="datasets/GazeGene/training_gazegene_dataset_normalized_det_conf_0_8_with_lm168_TRAIN.csv",
        valid_file="datasets/GazeGene/training_gazegene_dataset_normalized_det_conf_0_8_with_lm168_VALID.csv",
        test_file="datasets/GazeGene/training_gazegene_dataset_normalized_det_conf_0_8_with_lm168_TEST.csv",
        all_file="datasets/GazeGene/training_gazegene_dataset_normalized_det_conf_0_8_with_lm168.csv",
    ),
    "xgaze": DatasetSpec(
        key="xgaze",
        label="XGaze",
        train_file="datasets/XGaze_448/training_xgaze_dataset_normalized_det_conf_0_8_with_lm168_TRAIN.csv",
        valid_file="datasets/XGaze_448/training_xgaze_dataset_normalized_det_conf_0_8_with_lm168_VALID.csv",
        test_file="datasets/XGaze_448/training_xgaze_dataset_normalized_det_conf_0_8_with_lm168_TEST.csv",
        all_file="datasets/XGaze_448/training_xgaze_dataset_normalized_det_conf_0_8_with_lm168.csv",
    ),
    "blender": DatasetSpec(
        key="blender",
        label="Blender - synthetic",
        train_file="datasets/Blender - synthetic/normalized_dataset_TRAIN.csv",
        valid_file="datasets/Blender - synthetic/normalized_dataset_VALID.csv",
        test_file="datasets/Blender - synthetic/normalized_dataset_TEST.csv",
        all_file="datasets/Blender - synthetic/normalized_dataset.csv",
    ),
}

_ALIASES = {
    "gaze360": "gaze360",
    "g360": "gaze360",
    "gazegene": "gazegene",
    "gaze gene": "gazegene",
    "xgaze": "xgaze",
    "xgaze_448": "xgaze",
    "eth-xgaze": "xgaze",
    "eth xgaze": "xgaze",
    "blender": "blender",
    "blender synthetic": "blender",
    "blender-synthetic": "blender",
    "blender_synthetic": "blender",
}

_REAL_DATASET_KEYS = ("gaze360", "gazegene", "xgaze")


def normalize_dataset_key(name: str) -> str:
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    if key not in _ALIASES:
        supported = ", ".join(sorted(_DATASETS.keys()))
        raise KeyError(f"Unsupported dataset '{name}'. Supported: {supported}")
    return _ALIASES[key]


def get_dataset_spec(name: str) -> DatasetSpec:
    return _DATASETS[normalize_dataset_key(name)]


def list_dataset_keys() -> List[str]:
    return list(_DATASETS.keys())


def default_cross_domain_targets(source_dataset: Optional[str]) -> List[str]:
    if not source_dataset:
        return list(_REAL_DATASET_KEYS)

    source_key = normalize_dataset_key(source_dataset)
    return [key for key in _REAL_DATASET_KEYS if key != source_key]
