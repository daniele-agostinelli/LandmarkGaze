import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from dataset_registry import get_dataset_spec


def parse_args() -> argparse.Namespace:
    blender_spec = get_dataset_spec("blender")
    parser = argparse.ArgumentParser(
        description="Create subject-wise TRAIN/VALID/TEST CSV splits without loading the full dataset in memory."
    )
    parser.add_argument(
        "--input-csv",
        default=blender_spec.all_file,
        help="Input semicolon-separated CSV.",
    )
    parser.add_argument(
        "--train-out",
        default=blender_spec.train_file,
        help="Output TRAIN CSV path.",
    )
    parser.add_argument(
        "--valid-out",
        default=blender_spec.valid_file,
        help="Output VALID CSV path.",
    )
    parser.add_argument(
        "--test-out",
        default=blender_spec.test_file,
        help="Output TEST CSV path.",
    )
    parser.add_argument("--sep", default=";", help="CSV separator.")
    parser.add_argument(
        "--split-mode",
        choices=["ratio", "random", "last_n", "specific"],
        default="ratio",
        help="How to choose validation/test subjects.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation subject ratio for split-mode=ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test subject ratio for split-mode=ratio.",
    )
    parser.add_argument(
        "--last-n-count",
        type=int,
        default=10,
        help="Total number of trailing subjects reserved for VALID+TEST when split-mode=last_n.",
    )
    parser.add_argument(
        "--train-subjects",
        nargs="*",
        default=[],
        help="Explicit training subjects when split-mode=specific. If omitted, training subjects are inferred as the remaining ones.",
    )
    parser.add_argument(
        "--valid-subjects",
        nargs="*",
        default=[],
        help="Explicit validation subjects when split-mode=specific.",
    )
    parser.add_argument(
        "--test-subjects",
        nargs="*",
        default=[],
        help="Explicit test subjects when split-mode=specific.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split-mode=ratio.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Rows per chunk while scanning/writing large CSV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned split, without writing output CSV files.",
    )
    return parser.parse_args()


def _collect_subject_counts(input_csv: Path, sep: str, chunksize: int) -> Counter:
    counts: Counter = Counter()
    for chunk in pd.read_csv(input_csv, sep=sep, usecols=["subject"], chunksize=chunksize):
        counts.update(chunk["subject"].astype(str).tolist())
    return counts


def _split_by_ratio(
    all_subjects: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    total_subjects = len(all_subjects)
    if total_subjects < 3:
        raise ValueError("Need at least 3 subjects to build train/valid/test splits.")

    rng = np.random.default_rng(seed)
    shuffled = list(all_subjects)
    rng.shuffle(shuffled)

    valid_count = max(1, int(round(total_subjects * val_ratio)))
    test_count = max(1, int(round(total_subjects * test_ratio)))
    if valid_count + test_count >= total_subjects:
        raise ValueError(
            "Invalid val/test ratios: they leave no subjects for training."
        )

    valid_subjects = set(shuffled[:valid_count])
    test_subjects = set(shuffled[valid_count : valid_count + test_count])
    return valid_subjects, test_subjects


def _split_by_last_n(all_subjects: Sequence[str], last_n_count: int) -> Tuple[Set[str], Set[str]]:
    if last_n_count < 2:
        raise ValueError("--last-n-count must be at least 2.")
    if last_n_count >= len(all_subjects):
        raise ValueError("--last-n-count must be smaller than the total subject count.")

    selected = list(all_subjects[-last_n_count:])
    half = len(selected) // 2
    valid_subjects = set(selected[:half])
    test_subjects = set(selected[half:])
    return valid_subjects, test_subjects


def _split_by_specific(
    all_subjects: Iterable[str],
    train_subjects: Sequence[str],
    valid_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> Tuple[Set[str], Set[str], Set[str]]:
    known_subjects = set(all_subjects)
    train_set = {subject for subject in train_subjects if subject in known_subjects}
    valid_set = {subject for subject in valid_subjects if subject in known_subjects}
    test_set = {subject for subject in test_subjects if subject in known_subjects}

    missing = (set(train_subjects) | set(valid_subjects) | set(test_subjects)) - known_subjects
    if missing:
        print(f"Warning: configured subjects not found in CSV: {sorted(missing)}")

    if not valid_set or not test_set:
        raise ValueError(
            "split-mode=specific requires at least one valid subject and one test subject."
        )

    overlap = (train_set & valid_set) | (train_set & test_set) | (valid_set & test_set)
    if overlap:
        raise ValueError(f"Explicit subject lists overlap: {sorted(overlap)}")

    if train_set:
        return train_set, valid_set, test_set

    inferred_train = known_subjects - valid_set - test_set
    if not inferred_train:
        raise ValueError("No training subjects remain after excluding explicit VALID/TEST subjects.")
    return inferred_train, valid_set, test_set


def choose_subject_splits(args: argparse.Namespace, all_subjects: Sequence[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    if args.split_mode in ("ratio", "random"):
        valid_subjects, test_subjects = _split_by_ratio(
            all_subjects=all_subjects,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_subjects = set(all_subjects) - valid_subjects - test_subjects
    elif args.split_mode == "last_n":
        valid_subjects, test_subjects = _split_by_last_n(all_subjects, args.last_n_count)
        train_subjects = set(all_subjects) - valid_subjects - test_subjects
    else:
        train_subjects, valid_subjects, test_subjects = _split_by_specific(
            all_subjects=all_subjects,
            train_subjects=args.train_subjects,
            valid_subjects=args.valid_subjects,
            test_subjects=args.test_subjects,
        )

    overlap = valid_subjects & test_subjects
    if overlap:
        raise ValueError(f"Validation and test subjects overlap: {sorted(overlap)}")

    if not train_subjects:
        raise ValueError("No subjects left for training after the split.")

    return train_subjects, valid_subjects, test_subjects


def _print_summary(
    subject_counts: Counter,
    train_subjects: Set[str],
    valid_subjects: Set[str],
    test_subjects: Set[str],
) -> None:
    train_samples = sum(subject_counts[subject] for subject in train_subjects)
    valid_samples = sum(subject_counts[subject] for subject in valid_subjects)
    test_samples = sum(subject_counts[subject] for subject in test_subjects)

    print("\n--- Split Summary ---")
    print(f"Train Subjects: {len(train_subjects)}")
    print(f"Valid Subjects: {len(valid_subjects)}")
    print(f"Test Subjects:  {len(test_subjects)}")
    print(f"Train Samples:  {train_samples}")
    print(f"Valid Samples:  {valid_samples}")
    print(f"Test Samples:   {test_samples}")


def _write_split(
    input_csv: Path,
    output_path: Path,
    allowed_subjects: Set[str],
    sep: str,
    chunksize: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    for chunk in pd.read_csv(input_csv, sep=sep, chunksize=chunksize):
        filtered = chunk[chunk["subject"].astype(str).isin(allowed_subjects)]
        if filtered.empty:
            continue

        mode = "w" if not wrote_header else "a"
        filtered.to_csv(output_path, sep=sep, index=False, mode=mode, header=not wrote_header)
        wrote_header = True

    if not wrote_header:
        raise RuntimeError(f"No rows written to {output_path}. Check the subject split configuration.")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    train_out = Path(args.train_out).expanduser().resolve()
    valid_out = Path(args.valid_out).expanduser().resolve()
    test_out = Path(args.test_out).expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Reading subjects from: {input_csv}")
    subject_counts = _collect_subject_counts(input_csv, args.sep, args.chunksize)
    if not subject_counts:
        raise ValueError("No subjects found in the input CSV.")

    all_subjects = sorted(subject_counts.keys())
    print(f"Total unique subjects found: {len(all_subjects)}")

    train_subjects, valid_subjects, test_subjects = choose_subject_splits(args, all_subjects)
    _print_summary(subject_counts, train_subjects, valid_subjects, test_subjects)

    leakage = (
        (train_subjects & valid_subjects)
        | (train_subjects & test_subjects)
        | (valid_subjects & test_subjects)
    )
    if leakage:
        raise RuntimeError(f"Data leakage detected in subject split: {sorted(leakage)}")
    print("Verification passed: no subject overlap between TRAIN, VALID and TEST.")

    print("\n--- Output Paths ---")
    print(f"TRAIN: {train_out}")
    print(f"VALID: {valid_out}")
    print(f"TEST:  {test_out}")

    if args.dry_run:
        print("\nDry run completed. No files were written.")
        return

    print("\nWriting split CSV files...")
    _write_split(input_csv, train_out, train_subjects, args.sep, args.chunksize)
    _write_split(input_csv, valid_out, valid_subjects, args.sep, args.chunksize)
    _write_split(input_csv, test_out, test_subjects, args.sep, args.chunksize)
    print("Done.")


if __name__ == "__main__":
    main()
