"""Create a conventional row-oriented CSV from the bundled UCI raw data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from breast_cancer_classifier import DEFAULT_DATA_PATH, FEATURE_NAMES, load_dataset  # noqa: E402


def write_prepared_csv(source: Path, destination: Path, force: bool = False) -> None:
    """Write a documented derivative without modifying the raw source file."""

    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force to replace it.")
    dataset = load_dataset(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", *FEATURE_NAMES, "diagnosis"))
        for sample_id, features, label in zip(
            dataset.sample_ids, dataset.features, dataset.labels
        ):
            writer.writerow(
                (sample_id, *(int(value) for value in features), "malignant" if label else "benign")
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed" / "breast_cancer_wisconsin_8_feature.csv",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_prepared_csv(args.source, args.destination, force=args.force)
    print(f"Prepared dataset written to {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
