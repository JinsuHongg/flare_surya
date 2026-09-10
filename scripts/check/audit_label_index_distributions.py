#!/usr/bin/env python3
"""Audit label distributions against an SDO availability index.

Example:
    python3 scripts/check/audit_label_index_distributions.py \
        --index-root /mnt/storage/surya/index_data \
        --availability-index /mnt/storage/surya/index_data/surya-bench_sdo.csv \
        --output results/audits/label_index_distributions.csv
"""

import argparse
import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from loguru import logger


OUTPUT_FIELDS = (
    "label_directory",
    "label_file",
    "forecasting_window_hours",
    "flare_threshold",
    "sampling_interval_hours",
    "source_label_rows",
    "overlapping_present_rows",
    "negative_label_max",
    "positive_label_max",
    "other_or_missing_label_max",
    "positive_percent",
    "excluded_nonoverlapping_rows",
)


def parse_timestamp(value: str, source: Path, row_number: int) -> datetime:
    """Parse one ISO-like timestamp and add midnight for date-only labels.

    Args:
        value: Timestamp text from a CSV row.
        source: CSV containing the timestamp.
        row_number: One-based data row number for error reporting.

    Returns:
        A timezone-naive datetime.

    Raises:
        ValueError: If the timestamp is empty or cannot be parsed.
    """
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(
            f"Invalid timestamp in {source} at data row {row_number}: {value!r}"
        ) from error


def read_available_timestamps(index_path: Path) -> set[datetime]:
    """Return SDO timesteps whose files are available.

    Args:
        index_path: CSV with ``timestep`` and ``present`` columns.

    Returns:
        Unique available SDO timestamps.
    """
    available: set[datetime] = set()
    with index_path.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"timestep", "present"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Availability index must contain {required_columns}: {index_path}")
        for row_number, row in enumerate(reader, start=1):
            if row["present"] == "1":
                available.add(parse_timestamp(row["timestep"], index_path, row_number))
    return available


def audit_label_file(
    label_path: Path, available_timestamps: set[datetime]
) -> tuple[int, int, int, int, int]:
    """Count available labels and their binary ``label_max`` distribution.

    Args:
        label_path: Label CSV with ``timestamp`` and ``label_max`` columns.
        available_timestamps: Available SDO timestamps.

    Returns:
        Source rows, available rows, negative rows, positive rows, and rows with
        another or missing label among available rows.
    """
    source_rows = 0
    available_rows = 0
    negative_rows = 0
    positive_rows = 0
    other_or_missing_rows = 0
    with label_path.open(newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        required_columns = {"timestamp", "label_max"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Label file must contain {required_columns}: {label_path}")
        for row_number, row in enumerate(reader, start=1):
            source_rows += 1
            timestamp = parse_timestamp(row["timestamp"], label_path, row_number)
            if timestamp not in available_timestamps:
                continue
            available_rows += 1
            if row["label_max"] == "0":
                negative_rows += 1
            elif row["label_max"] == "1":
                positive_rows += 1
            else:
                other_or_missing_rows += 1
    return (
        source_rows,
        available_rows,
        negative_rows,
        positive_rows,
        other_or_missing_rows,
    )


def iter_audit_rows(
    prior_rows: Iterable[dict[str, str]], index_root: Path, available_timestamps: set[datetime]
) -> Iterable[dict[str, str | int]]:
    """Recalculate audit rows while retaining their dataset metadata.

    Args:
        prior_rows: Existing audit rows that identify label files and metadata.
        index_root: Directory containing the label-directory subdirectories.
        available_timestamps: Available SDO timestamps.

    Yields:
        Recalculated audit rows in their original order.
    """
    for prior_row in prior_rows:
        label_path = index_root / prior_row["label_directory"] / prior_row["label_file"]
        (
            source_rows,
            overlapping_rows,
            negative_rows,
            positive_rows,
            other_or_missing_rows,
        ) = audit_label_file(label_path, available_timestamps)
        classified_rows = negative_rows + positive_rows
        positive_percent = (
            f"{(positive_rows / classified_rows) * 100:.6f}" if classified_rows else ""
        )
        yield {
            "label_directory": prior_row["label_directory"],
            "label_file": prior_row["label_file"],
            "forecasting_window_hours": prior_row["forecasting_window_hours"],
            "flare_threshold": prior_row["flare_threshold"],
            "sampling_interval_hours": prior_row["sampling_interval_hours"],
            "source_label_rows": source_rows,
            "overlapping_present_rows": overlapping_rows,
            "negative_label_max": negative_rows,
            "positive_label_max": positive_rows,
            "other_or_missing_label_max": other_or_missing_rows,
            "positive_percent": positive_percent,
            "excluded_nonoverlapping_rows": source_rows - overlapping_rows,
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--availability-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Rebuild the requested distribution audit."""
    args = parse_arguments()
    with args.output.open(newline="") as file_handle:
        prior_rows = list(csv.DictReader(file_handle))
    available_timestamps = read_available_timestamps(args.availability_index)
    audit_rows = list(iter_audit_rows(prior_rows, args.index_root, available_timestamps))
    with args.output.open("w", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    logger.info(
        "Wrote {} label-distribution audit rows using {} available SDO timestamps.",
        len(audit_rows),
        len(available_timestamps),
    )


if __name__ == "__main__":
    main()
