"""Plot 24-hour test HSS and TSS after validation-CSS model selection.

For each flare threshold and architecture, the script selects the sampling
interval with the highest validation CSS, then retrieves the corresponding
test-run metrics.  This keeps validation and test roles separate while making
the figure source auditable from the two result-ledger CSV files.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS = ("c", "m", "x")
ARCHITECTURES = {"mlp": "Surya", "alexnet": "AlexNet"}
METRICS = ("test_hss", "test_tss")


@dataclass(frozen=True)
class SelectedResult:
    """One validation-selected test result used in the figure."""

    architecture: str
    threshold: str
    sampling_interval: int
    validation_css: float
    test_hss: float
    test_tss: float
    test_run_id: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--test-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--forecasting-window",
        type=int,
        default=24,
        help="Forecasting window (hours) used for validation selection.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV audit ledger into records.

    Args:
        path: Source CSV path.

    Returns:
        Parsed CSV rows.
    """
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_sampling_interval(run_id: str) -> int | None:
    """Extract the sampling interval in hours from a test run ID."""
    if "_daily_" in run_id:
        return 24
    if "_hourly_" in run_id:
        return 1
    match = re.search(r"_(\d+)hour_", run_id)
    return int(match.group(1)) if match else None


def validation_best_rows(
    rows: list[dict[str, str]], forecasting_window: int
) -> dict[tuple[str, str], dict[str, str]]:
    """Select maximum validation CSS rows by architecture and threshold."""
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for prefix in ARCHITECTURES:
        for threshold in THRESHOLDS:
            candidates = [
                row
                for row in rows
                if row["status"] == "OK"
                and row["forecasting_window_size"] == str(forecasting_window)
                and row["threshold"] == threshold
                and row["experiment_id"].startswith(f"{prefix}_")
                and row["val_css"]
            ]
            if not candidates:
                raise ValueError(
                    f"No validation row for {prefix}, threshold {threshold}, "
                    f"window {forecasting_window}h."
                )
            selected[(prefix, threshold)] = max(
                candidates, key=lambda row: float(row["val_css"])
            )
    return selected


def matching_test_row(
    rows: list[dict[str, str]], prefix: str, threshold: str, interval: int
) -> dict[str, str]:
    """Return the matching complete test row, rejecting metric conflicts."""
    candidates = [
        row
        for row in rows
        if row["status"] == "OK"
        and row["run_id"].startswith(f"{prefix}_")
        and f"_{threshold}_" in row["run_id"]
        and test_sampling_interval(row["run_id"]) == interval
        and all(row[metric] for metric in METRICS)
    ]
    if not candidates:
        raise ValueError(
            f"No complete test row for {prefix}, threshold {threshold}, "
            f"sampling {interval}h."
        )

    metric_pairs = {
        (float(row["test_hss"]), float(row["test_tss"])) for row in candidates
    }
    if len(metric_pairs) != 1:
        raise ValueError(
            f"Conflicting test metrics for {prefix}, threshold {threshold}, "
            f"sampling {interval}h: {[row['run_id'] for row in candidates]}"
        )
    return min(candidates, key=lambda row: row["run_id"])


def select_results(
    validation_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    forecasting_window: int,
) -> list[SelectedResult]:
    """Join validation-selected intervals to matching test metrics."""
    validation_selected = validation_best_rows(validation_rows, forecasting_window)
    selected: list[SelectedResult] = []
    for prefix, architecture in ARCHITECTURES.items():
        for threshold in THRESHOLDS:
            validation_row = validation_selected[(prefix, threshold)]
            interval = int(validation_row["sampling_interval"])
            test_row = matching_test_row(test_rows, prefix, threshold, interval)
            selected.append(
                SelectedResult(
                    architecture=architecture,
                    threshold=threshold.upper(),
                    sampling_interval=interval,
                    validation_css=float(validation_row["val_css"]),
                    test_hss=float(test_row["test_hss"]),
                    test_tss=float(test_row["test_tss"]),
                    test_run_id=test_row["run_id"],
                )
            )
    return selected


def write_selection_ledger(rows: list[SelectedResult], output_path: Path) -> None:
    """Write exact plotted data and provenance to a CSV ledger."""
    fieldnames = [
        "architecture",
        "threshold",
        "sampling_interval_hours",
        "validation_css",
        "test_hss",
        "test_tss",
        "test_run_id",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "architecture": row.architecture,
                    "threshold": row.threshold,
                    "sampling_interval_hours": row.sampling_interval,
                    "validation_css": row.validation_css,
                    "test_hss": row.test_hss,
                    "test_tss": row.test_tss,
                    "test_run_id": row.test_run_id,
                }
            )


def plot_metric(rows: list[SelectedResult], metric: str, output_path: Path) -> None:
    """Create one grouped-bar test-score figure for HSS or TSS."""
    metric_name = metric.removeprefix("test_").upper()
    values = {
        architecture: [
            getattr(next(row for row in rows if row.architecture == architecture and row.threshold == threshold), metric)
            for threshold in ("C", "M", "X")
        ]
        for architecture in ARCHITECTURES.values()
    }
    intervals = {
        architecture: [
            next(row for row in rows if row.architecture == architecture and row.threshold == threshold).sampling_interval
            for threshold in ("C", "M", "X")
        ]
        for architecture in ARCHITECTURES.values()
    }
    x = np.arange(len(THRESHOLDS))
    width = 0.34
    colors = {"Surya": "#1f4e79", "AlexNet": "#c97a24"}

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.22)
    for offset, architecture in [(-width / 2, "Surya"), (width / 2, "AlexNet")]:
        bars = axis.bar(
            x + offset,
            values[architecture],
            width,
            label=architecture,
            color=colors[architecture],
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values[architecture]], padding=3, fontsize=10)

    axis.set_ylim(0, 1.0)
    axis.set_ylabel(metric_name)
    axis.set_xticks(x, ["≥C", "≥M", "≥X"])
    axis.set_title(f"24-hour test {metric_name} by flare threshold", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    interval_note = "; ".join(
        f"{threshold}: S/A {surya}/{alexnet}h"
        for threshold, surya, alexnet in zip(
            ("≥C", "≥M", "≥X"), intervals["Surya"], intervals["AlexNet"], strict=True
        )
    )
    figure.text(
        0.125,
        0.04,
        "Sampling interval selected independently by maximum validation CSS. " + interval_note,
        fontsize=9,
        color="#404040",
    )
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    """Create test-score figures and their plotting-data ledger."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_results(
        read_csv(args.validation_audit),
        read_csv(args.test_audit),
        args.forecasting_window,
    )
    write_selection_ledger(selected, args.output_dir / "test_24h_validation_selected_scores.csv")
    plot_metric(selected, "test_tss", args.output_dir / "test_24h_validation_selected_tss.png")
    plot_metric(selected, "test_hss", args.output_dir / "test_24h_validation_selected_hss.png")


if __name__ == "__main__":
    main()
