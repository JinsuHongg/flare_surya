"""Retrieve and organize test metrics from a W&B project.

The output preserves each cloud run ID and computes false alarm ratio (FAR)
and probability of detection (POD) from W&B's logged confusion-matrix counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import wandb
from loguru import logger


SUMMARY_KEYS = ("test/hss", "test/tss", "test/tp", "test/fp", "test/tn", "test/fn")
FIELDNAMES = (
    "run_id",
    "run_name",
    "run_url",
    "run_state",
    "architecture",
    "test_hss",
    "test_tss",
    "tp",
    "fp",
    "tn",
    "fn",
    "far",
    "pod",
    "status",
    "detail",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="gsu-dmlab/flareforecasting-nosampling-test",
        help="W&B entity/project path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def architecture_from_run_id(run_id: str) -> str:
    """Map the run-ID prefix to its explicit architecture name.

    Args:
        run_id: Full W&B cloud run ID.

    Returns:
        A display architecture name, or ``Unknown`` when the prefix is not
        represented by the manuscript's model taxonomy.
    """
    if run_id.startswith("alexnet_"):
        return "AlexNet"
    if run_id.startswith("resnet18_"):
        return "ResNet18"
    if run_id.startswith("mlp_"):
        return "Surya"
    return "Unknown"


def numeric_summary(summary: Any, key: str) -> float | None:
    """Return one finite numeric W&B summary value, if present."""
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def build_record(run: wandb.apis.public.Run) -> dict[str, str | float | int]:
    """Create one test-result audit row from a W&B run summary.

    Args:
        run: W&B public API run.

    Returns:
        Run metadata, raw test metrics, derived FAR/POD, and an audit status.
    """
    summary = run.summary
    values = {key: numeric_summary(summary, key) for key in SUMMARY_KEYS}
    record: dict[str, str | float | int] = {
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "run_state": run.state,
        "architecture": architecture_from_run_id(run.id),
        "test_hss": "",
        "test_tss": "",
        "tp": "",
        "fp": "",
        "tn": "",
        "fn": "",
        "far": "",
        "pod": "",
        "status": "OK",
        "detail": "",
    }
    if any(values[key] is None for key in SUMMARY_KEYS):
        record.update(
            {
                "status": "MISSING_TEST_METRICS",
                "detail": "One or more required test metrics are absent from the W&B summary.",
            }
        )
        return record

    tp = values["test/tp"]
    fp = values["test/fp"]
    tn = values["test/tn"]
    fn = values["test/fn"]
    assert tp is not None and fp is not None and tn is not None and fn is not None
    predicted_positive = tp + fp
    actual_positive = tp + fn
    record.update(
        {
            "test_hss": values["test/hss"],
            "test_tss": values["test/tss"],
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "far": fp / predicted_positive if predicted_positive else "",
            "pod": tp / actual_positive if actual_positive else "",
        }
    )
    if not predicted_positive or not actual_positive:
        record.update(
            {
                "status": "UNDEFINED_RATE",
                "detail": "FAR or POD denominator is zero.",
            }
        )
    return record


def write_records(records: list[dict[str, str | float | int]], output_path: Path) -> None:
    """Write the test-result audit CSV.

    Args:
        records: Test-result audit records.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Retrieve project runs and write raw plus derived test metrics."""
    args = parse_args()
    api = wandb.Api(timeout=30)
    records = []
    for run in api.runs(args.project):
        records.append(build_record(run))
    records.sort(key=lambda record: str(record["run_id"]))
    write_records(records, args.output)
    successful = sum(record["status"] == "OK" for record in records)
    logger.info("Retrieved {}/{} complete test summaries.", successful, len(records))


if __name__ == "__main__":
    main()
