"""Plot test TSS/HSS sweeps recorded in W&B threshold tables.

The default runs are the M1.0, 24-hour forecasting-window, 8-hour sampling
interval Surya and AlexNet experiments.  Each curve is read from the exact
run's ``test/threshold_df`` table; it does not reconstruct scores from a
generic local prediction export.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from loguru import logger


DEFAULT_PROJECT = "gsu-dmlab/flareforecasting-nosampling-test"
DEFAULT_SURYA_RUN = "mlp_8hour_m_ce_nosample_nasgh_test_lr1e-05_wd0.05"
DEFAULT_ALEXNET_RUN = "alexnet_8hour_m_nasgh_test_lr1e-05_wd0.05"
DEFAULT_OUTPUT_STEM = "test_24h_m_8h"
DEFAULT_CONDITION_LABEL = (
    "Solar Cycle 25 test set | ≥M1.0 | 24-hour forecasting window | "
    "8-hour sampling interval"
)
TABLE_KEY = "test/threshold_df"


def read_threshold_table(path: Path) -> list[tuple[float, float, float]]:
    """Read probability threshold, TSS, and HSS rows from a W&B table JSON.

    Args:
        path: Downloaded ``test/threshold_df`` JSON file.

    Returns:
        Rows sorted by probability threshold as ``(threshold, tss, hss)``.

    Raises:
        ValueError: If the W&B table lacks required columns or valid rows.
    """
    payload = json.loads(path.read_text())
    columns = payload.get("columns")
    data = payload.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        raise ValueError(f"{path} is not a W&B table JSON.")
    required = ("threshold", "TSS", "HSS")
    if any(column not in columns for column in required):
        raise ValueError(f"{path} lacks required columns {required}.")
    indices = {column: columns.index(column) for column in required}
    rows: list[tuple[float, float, float]] = []
    for row in data:
        if not isinstance(row, list):
            raise ValueError(f"{path} contains a non-list table row.")
        try:
            rows.append(
                (
                    float(row[indices["threshold"]]),
                    float(row[indices["TSS"]]),
                    float(row[indices["HSS"]]),
                )
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(f"{path} contains an invalid threshold-score row.") from error
    if not rows:
        raise ValueError(f"{path} contains no threshold-score rows.")
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate probability thresholds.")
    return rows


def download_threshold_table(
    api: wandb.Api, project: str, run_id: str, download_directory: Path
) -> Path:
    """Download one run's W&B threshold table and return its local path."""
    run = api.run(f"{project}/{run_id}")
    table_metadata = run.summary.get(TABLE_KEY)
    if not hasattr(table_metadata, "get") or not isinstance(table_metadata.get("path"), str):
        raise ValueError(f"{run_id} does not contain {TABLE_KEY} in its W&B summary.")
    file = run.file(table_metadata["path"]).download(root=str(download_directory), replace=True)
    return Path(file.name)


def validate_matching_thresholds(
    curves: dict[str, Sequence[tuple[float, float, float]]]
) -> None:
    """Ensure compared models use the same probability-threshold grid."""
    threshold_sets = {name: tuple(row[0] for row in rows) for name, rows in curves.items()}
    if len(set(threshold_sets.values())) != 1:
        raise ValueError(f"Models have different threshold grids: {threshold_sets}.")


def output_path_for_metric(output_directory: Path, output_stem: str, metric: str) -> Path:
    """Return the condition-specific file path for one threshold-sweep metric."""
    return output_directory / f"{output_stem}_{metric.lower()}_threshold_sweep.png"


def plot_metric(
    curves: dict[str, Sequence[tuple[float, float, float]]],
    metric: str,
    output_path: Path,
    condition_label: str,
) -> None:
    """Create one two-model line plot for TSS or HSS."""
    metric_index = {"TSS": 1, "HSS": 2}[metric]
    colors = {"Surya": "#2070b4", "AlexNet": "#e07a28"}
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for name, rows in curves.items():
        thresholds = [row[0] for row in rows]
        values = [row[metric_index] for row in rows]
        axis.plot(thresholds, values, label=name, color=colors[name], linewidth=2.4)
        fixed_index = thresholds.index(0.5)
        axis.scatter(0.5, values[fixed_index], color=colors[name], s=38, zorder=3)
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="Fixed threshold = 0.50")
    axis.set(
        title=f"Test {metric} versus probability threshold",
        xlabel="Probability threshold (positive if probability > threshold)",
        ylabel=metric,
        xlim=(0.01, 0.99),
        ylim=(0.0, 0.65),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3, loc="lower left")
    figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.22)
    figure.text(
        0.5,
        0.04,
        condition_label,
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_threshold_grid(
    curves_by_condition: dict[str, dict[str, Sequence[tuple[float, float, float]]]],
    output_path: Path,
) -> None:
    """Create a two-condition grid of TSS and HSS probability-threshold curves.

    Args:
        curves_by_condition: Ordered condition labels and their Surya/AlexNet
            threshold-score curves.
        output_path: Destination PNG path.

    Raises:
        ValueError: If the grid does not contain exactly two conditions.
    """
    if len(curves_by_condition) != 2:
        raise ValueError("Threshold grid requires exactly two forecast conditions.")
    colors = {"Surya": "#2070b4", "AlexNet": "#e07a28"}
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.0, 6.4))
    for row_index, (condition, curves) in enumerate(curves_by_condition.items()):
        validate_matching_thresholds(curves)
        for column_index, metric in enumerate(("TSS", "HSS")):
            axis = axes[row_index, column_index]
            metric_index = {"TSS": 1, "HSS": 2}[metric]
            for name, rows in curves.items():
                thresholds = [row[0] for row in rows]
                values = [row[metric_index] for row in rows]
                axis.plot(thresholds, values, label=name, color=colors[name], linewidth=2.0)
                fixed_index = thresholds.index(0.5)
                axis.scatter(0.5, values[fixed_index], color=colors[name], s=26, zorder=3)
            axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1.0)
            axis.set(
                title=f"{condition} — {metric}",
                xlim=(0.01, 0.99),
                ylim=(0.0, 0.65),
                ylabel=metric,
            )
            axis.grid(axis="y", alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Probability threshold")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, ncols=2, loc="lower left", fontsize=9)
    figure.suptitle("≥M1.0 test: probability-threshold sweeps", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "Solar Cycle 25 test set | Positive prediction if probability > threshold | Dashed line: 0.50",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.12, hspace=0.38, wspace=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--surya-run", default=DEFAULT_SURYA_RUN)
    parser.add_argument("--alexnet-run", default=DEFAULT_ALEXNET_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--condition-label", default=DEFAULT_CONDITION_LABEL)
    return parser.parse_args()


def main() -> None:
    """Download the two recorded sweeps and render TSS and HSS figures."""
    args = parse_args()
    download_directory = args.output_dir / "wandb_tables"
    api = wandb.Api(timeout=60)
    table_paths = {
        "Surya": download_threshold_table(api, args.project, args.surya_run, download_directory),
        "AlexNet": download_threshold_table(api, args.project, args.alexnet_run, download_directory),
    }
    curves = {name: read_threshold_table(path) for name, path in table_paths.items()}
    validate_matching_thresholds(curves)
    for metric in ("TSS", "HSS"):
        output_path = output_path_for_metric(args.output_dir, args.output_stem, metric)
        plot_metric(curves, metric, output_path, args.condition_label)
        logger.info("Wrote {}.", output_path)


if __name__ == "__main__":
    main()
