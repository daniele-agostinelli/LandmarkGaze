import hashlib
import json
import os
from typing import Iterable

import numpy as np


def build_cache_dir(csv_path: str, cache_root: str, namespace: str, signature_parts: Iterable[object]) -> str:
    abs_csv = os.path.abspath(csv_path)
    stat = os.stat(abs_csv)
    payload = {
        "csv_path": abs_csv,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "signature_parts": list(signature_parts),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    csv_stem = os.path.splitext(os.path.basename(abs_csv))[0]
    return os.path.join(cache_root, namespace, f"{csv_stem}_{digest}")


def cache_ready(cache_dir: str, array_names: Iterable[str]) -> bool:
    if not os.path.exists(os.path.join(cache_dir, "metadata.json")):
        return False
    return all(os.path.exists(os.path.join(cache_dir, f"{name}.npy")) for name in array_names)


def write_array_cache(cache_dir: str, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    for name, array in arrays.items():
        final_path = os.path.join(cache_dir, f"{name}.npy")
        tmp_path = f"{final_path}.tmp.npy"
        np.save(tmp_path, array)
        os.replace(tmp_path, final_path)

    metadata_path = os.path.join(cache_dir, "metadata.json")
    metadata_tmp = f"{metadata_path}.tmp"
    with open(metadata_tmp, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(metadata_tmp, metadata_path)


def read_metadata(cache_dir: str) -> dict[str, object]:
    with open(os.path.join(cache_dir, "metadata.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_array_cache(cache_dir: str, array_names: Iterable[str]) -> dict[str, np.ndarray]:
    return {
        name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r+")
        for name in array_names
    }
