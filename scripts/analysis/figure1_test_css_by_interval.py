"""Plot test CSS across forecast horizons and sampling intervals."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
from loguru import logger


class ResultRow(TypedDict):
    """A test CSS estimate and its optional temporal bootstrap interval."""

    forecast_hours: int
    threshold: str
    model: str
    strategy: str
    interval_hours: int
    css: float
    ci_lower: float | None
    ci_upper: float | None


MODEL_COLORS = {"Surya": "#0072B2", "AlexNet": "#D55E00", "ResNet18": "#009E73"}
STRATEGY_STYLES = {"full": "-", "undersampled": "--"}
THRESHOLDS = ("C", "M", "X")
FORECAST_HOURS = (24, 2)


def parse_args() -> argparse.Namespace:
    """Parse optional data and output paths."""
    repository_root = Path(__file__).resolve().parents[2]
    figure_dir = repository_root / "manuscript" / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=figure_dir / "figure1_test_css_data.csv")
    parser.add_argument("--output", type=Path, default=figure_dir / "figure1_test_css_by_interval.png")
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Vector PDF destination (defaults to the PNG path with a .pdf suffix).",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[ResultRow]:
    """Load test CSS values and optional confidence bounds from CSV."""
    rows: list[ResultRow] = []
    with path.open(encoding="utf-8", newline="") as source:
        for item in csv.DictReader(source):
            rows.append(
                {
                    "forecast_hours": int(required_field(item, "forecast_hours")),
                    "threshold": required_field(item, "threshold"),
                    "model": required_field(item, "model"),
                    "strategy": required_field(item, "strategy"),
                    "interval_hours": int(required_field(item, "interval_hours")),
                    "css": float(required_field(item, "css")),
                    "ci_lower": optional_float(item.get("ci_lower")),
                    "ci_upper": optional_float(item.get("ci_upper")),
                }
            )
    return rows


def required_field(row: dict[str, str | None], key: str) -> str:
    """Return a required CSV cell or raise an informative error."""
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required CSV field: {key}")
    return value


def optional_float(value: str | None) -> float | None:
    """Parse an optional numeric cell."""
    return float(value) if value else None


def error_bounds(row: ResultRow) -> tuple[float, float]:
    """Return lower and upper errors for a row with complete bounds."""
    lower = row["ci_lower"]
    upper = row["ci_upper"]
    if lower is None or upper is None:
        raise ValueError("Cannot calculate an error bar without both confidence bounds")
    return row["css"] - lower, upper - row["css"]


def plot_panel(
    axis: plt.Axes,
    rows: list[ResultRow],
    horizon: int,
    threshold: str,
    show_x_label: bool,
    show_y_label: bool,
) -> None:
    """Plot one forecast horizon and flare threshold."""
    panel_rows = [
        row for row in rows
        if row["forecast_hours"] == horizon and row["threshold"] == threshold
    ]
    if not panel_rows:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"$\\geq${threshold} class")
        for spine in ("top", "right", "bottom", "left"):
            axis.spines[spine].set_visible(False)
        return

    intervals = sorted({row["interval_hours"] for row in panel_rows})
    interval_positions = {interval: position for position, interval in enumerate(intervals)}
    grouped: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
    for row in panel_rows:
        grouped[(row["model"], row["strategy"])].append(row)

    for (model, strategy), series in grouped.items():
        series.sort(key=lambda row: row["interval_hours"])
        positions = [interval_positions[row["interval_hours"]] for row in series]
        estimates = [row["css"] for row in series]
        axis.plot(
            positions,
            estimates,
            marker="o",
            linestyle=STRATEGY_STYLES[strategy],
            color=MODEL_COLORS[model],
            label=f"{model} ({strategy})" if horizon == 24 and threshold == "M" else model,
        )
        bounded = [row for row in series if row["ci_lower"] is not None and row["ci_upper"] is not None]
        if bounded:
            axis.errorbar(
                [interval_positions[row["interval_hours"]] for row in bounded],
                [row["css"] for row in bounded],
                yerr=[
                    [error_bounds(row)[0] for row in bounded],
                    [error_bounds(row)[1] for row in bounded],
                ],
                fmt="none",
                ecolor=MODEL_COLORS[model],
                capsize=2,
                linewidth=0.9,
            )

    axis.set_xticks(range(len(intervals)), labels=[str(interval) for interval in intervals])
    axis.set_title(f"$\\geq${threshold} class", fontsize=18)
    axis.set_xlabel("Sampling interval (h)" if show_x_label else "", fontsize=16)
    axis.set_ylabel("CSS" if show_y_label else "", fontsize=16)
    axis.tick_params(axis="both", labelsize=14)
    axis.set_ylim(0, 1 if threshold != "X" else 0.4)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.6)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def create_figure(rows: list[ResultRow], output: Path, output_pdf: Path) -> None:
    """Render the six-panel test CSS figure as PNG and vector PDF."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row_index, horizon in enumerate(FORECAST_HOURS):
        for column_index, threshold in enumerate(THRESHOLDS):
            axis = axes[row_index, column_index]
            plot_panel(
                axis,
                rows,
                horizon,
                threshold,
                show_x_label=row_index == len(FORECAST_HOURS) - 1,
                show_y_label=column_index == 0,
            )
            if column_index == 0:
                axis.annotate(
                    f"{horizon}-hour forecast",
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )
            if axis.lines:
                legend_fontsize = 11 if horizon == 24 and threshold == "M" else 13
                axis.legend(frameon=False, fontsize=legend_fontsize)

    figure.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.91, hspace=0.38, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        figure.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved 300 dpi test CSS PNG to {}", output)
    logger.info("Saved vector test CSS PDF to {}", output_pdf)


def main() -> None:
    """Load experiment results and save the figure."""
    args = parse_args()
    rows = read_rows(args.data)
    output_pdf = args.output_pdf or args.output.with_suffix(".pdf")
    create_figure(rows, args.output, output_pdf)


if __name__ == "__main__":
    main()
