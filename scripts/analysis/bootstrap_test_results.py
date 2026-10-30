"""Compute fixed-threshold temporal moving-block bootstrap summaries.

Example:
    python scripts/analysis/bootstrap_test_results.py \
        --input-directory /home/jh/Data/surya/test_results \
        --output /tmp/flare-surya-bootstrap/test_24h_2h.csv \
        --block-days 7 --n-replicates 1000 --seed 1004

Files are scored at the fixed probability threshold of 0.5. The final value in
each row's timestamp sequence is its anchor. A timestamp reset starts a new
saved run, which is analyzed separately. Each moving-block replicate contains
the same number of observations as that run. Intervals are unpaired.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

NANOSECONDS_PER_DAY = 86_400_000_000_000
DEFAULT_THRESHOLD = 0.5
DEFAULT_BLOCK_DAYS = 7
DEFAULT_REPLICATES = 1_000
DEFAULT_SEED = 1_004
METRICS = ("CSS", "TSS", "HSS", "FAR", "FPR", "Recall", "F1macro", "Positive_F1")


@dataclass(frozen=True)
class TestPredictions:
    """Per-row forecasts and their timestamp-group structure.

    Args:
        path: Input CSV path.
        model: Model architecture inferred from the filename.
        flare_class: Flare threshold inferred from the filename.
        forecast_window: Forecast window inferred from the filename.
        sampling_interval: Training sampling interval inferred from the filename.
        timestamps_ns: Sorted unique anchor timestamps.
        probabilities: Positive-class probabilities, sorted by timestamp.
        targets: Binary targets, sorted by timestamp.
    """

    path: Path
    run_number: int
    model: str
    flare_class: str
    forecast_window: str
    sampling_interval: str
    timestamps_ns: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray


def parse_timestamp(value: str) -> int:
    """Return the latest nanosecond timestamp in a result row."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid timestamps value: {value!r}") from error
    if isinstance(parsed, (list, tuple)):
        if not parsed:
            raise ValueError("Timestamp sequence is empty.")
        parsed = parsed[-1]
    return int(float(parsed))


def describe_file(path: Path) -> tuple[str, str, str, str]:
    """Infer model, class, forecast window, and sampling interval."""
    stem = path.stem.lower()
    model = next((key for key in ("alexnet", "resnet18", "surya") if key in stem), "unknown")
    window_match = re.search(r"(?:c|m)2w", stem)
    forecast_window = "2hour" if window_match else "24hour"
    if "c2w" in stem:
        flare_class = ">=C"
    elif "m2w" in stem:
        flare_class = ">=M"
    else:
        class_match = re.search(r"(?:^|_)([cmx])(?:_|$)", stem)
        if not class_match:
            raise ValueError(f"Cannot infer flare class from filename {path.name!r}.")
        flare_class = f">={class_match.group(1).upper()}"
    interval_match = re.search(r"(?:^|_)(daily|hourly|\d+hour)(?:_|$)", stem)
    if not interval_match:
        raise ValueError(f"Cannot infer sampling interval from filename {path.name!r}.")
    return model, flare_class, forecast_window, interval_match.group(1)


def load_predictions(path: Path) -> tuple[TestPredictions, ...]:
    """Load per-row forecasts and separate concatenated test runs."""
    model, flare_class, forecast_window, sampling_interval = describe_file(path)
    timestamps: list[int] = []
    probabilities: list[float] = []
    targets: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamps", "predictions", "targets"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} lacks required columns {sorted(required)}.")
        for row in reader:
            timestamps.append(parse_timestamp(row["timestamps"]))
            probabilities.append(float(row["predictions"]))
            targets.append(int(float(row["targets"])))
    all_timestamps = np.asarray(timestamps, dtype=np.int64)
    all_probabilities = np.asarray(probabilities, dtype=float)
    all_targets = np.asarray(targets, dtype=np.int8)
    if not len(all_timestamps):
        raise ValueError(f"{path} contains no test rows.")
    if not np.isfinite(all_probabilities).all() or not np.isin(all_targets, (0, 1)).all():
        raise ValueError(f"{path} contains invalid probabilities or non-binary targets.")
    run_starts = np.r_[0, np.flatnonzero(np.diff(all_timestamps) <= 0) + 1]
    run_ends = np.r_[run_starts[1:], len(all_timestamps)]
    runs: list[TestPredictions] = []
    for run_number, (start, end) in enumerate(zip(run_starts, run_ends, strict=True), start=1):
        run_timestamps = all_timestamps[start:end]
        if np.any(np.diff(run_timestamps) <= 0):
            raise ValueError(f"{path} has duplicate or unordered timestamps inside run {run_number}.")
        runs.append(
            TestPredictions(
                path=path,
                run_number=run_number,
                model=model,
                flare_class=flare_class,
                forecast_window=forecast_window,
                sampling_interval=sampling_interval,
                timestamps_ns=run_timestamps,
                probabilities=all_probabilities[start:end],
                targets=all_targets[start:end],
            )
        )
    return tuple(runs)


def scores(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute paper metrics using the model's strict ``probability > 0.5`` rule."""
    predicted = probabilities > DEFAULT_THRESHOLD
    tp = float(np.sum(predicted & (targets == 1)))
    tn = float(np.sum(~predicted & (targets == 0)))
    fp = float(np.sum(predicted & (targets == 0)))
    fn = float(np.sum(~predicted & (targets == 1)))
    pod = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    tss = pod + specificity - 1
    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)
    hss = 2 * (tp * tn - fp * fn) / hss_denominator if hss_denominator else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else np.nan
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    f1_macro = (positive_f1 + negative_f1) / 2
    css = float(np.sqrt(max(hss, 0.0) * max(tss, 0.0))) if np.isfinite(hss + tss) else np.nan
    return {
        "CSS": css,
        "TSS": tss,
        "HSS": hss,
        "FAR": far,
        "FPR": fpr,
        "Recall": pod,
        "F1macro": f1_macro,
        "Positive_F1": positive_f1,
    }


def moving_blocks(data: TestPredictions, block_days: int) -> tuple[tuple[np.ndarray, ...], int]:
    """Build overlapping time blocks over unique timestamp anchors."""
    if len(data.timestamps_ns) < 2:
        raise ValueError(f"{data.path} has fewer than two unique timestamps.")
    steps = np.diff(data.timestamps_ns)
    cadence = int(np.median(steps))
    block_width = block_days * NANOSECONDS_PER_DAY
    block_length = round(block_width / cadence) + 1
    if block_length > len(data.timestamps_ns):
        raise ValueError(f"{data.path} is shorter than one {block_days}-day block.")
    blocks = tuple(
        np.arange(start, start + block_length, dtype=np.int64)
        for start in range(len(data.timestamps_ns) - block_length + 1)
    )
    return blocks, block_length


def bootstrap_scores(
    data: TestPredictions, block_days: int, n_replicates: int, seed: int
) -> tuple[dict[str, np.ndarray], int, int]:
    """Resample overlapping temporal blocks to the source row count."""
    blocks, block_length = moving_blocks(data, block_days)
    rng = np.random.default_rng(seed)
    results = {metric: np.empty(n_replicates, dtype=float) for metric in METRICS}
    n_timestamps = len(data.timestamps_ns)
    blocks_per_replicate = int(np.ceil(n_timestamps / block_length))
    offsets = np.arange(block_length, dtype=np.int64)
    for replicate in range(n_replicates):
        starts = rng.integers(0, len(blocks), size=blocks_per_replicate)
        sampled_groups = (starts[:, np.newaxis] + offsets).reshape(-1)[:n_timestamps]
        replicate_scores = scores(
            data.probabilities[sampled_groups], data.targets[sampled_groups]
        )
        for metric in METRICS:
            results[metric][replicate] = replicate_scores[metric]
    return results, len(blocks), block_length


def summarize_run(
    data: TestPredictions, n_replicates: int, block_days: int, seed: int
) -> list[dict[str, str | float | int]]:
    """Compute full-data scores and bootstrap summaries for one run."""
    bootstrap, available_blocks, block_length = bootstrap_scores(data, block_days, n_replicates, seed)
    point = scores(data.probabilities, data.targets)
    calendar_blocks = len(
        np.unique(
            (data.timestamps_ns - data.timestamps_ns.min())
            // (block_days * NANOSECONDS_PER_DAY)
        )
    )
    records: list[dict[str, str | float | int]] = []
    for metric in METRICS:
        values = bootstrap[metric]
        lower, upper = np.nanpercentile(values, (2.5, 97.5))
        records.append(
            {
                "dataset_split": "test",
                "source_file": data.path.name,
                "run_number": data.run_number,
                "model": data.model,
                "flare_class": data.flare_class,
                "forecast_window": data.forecast_window,
                "sampling_interval": data.sampling_interval,
                "metric": metric,
                "point_estimate": point[metric],
                "bootstrap_mean": float(np.nanmean(values)),
                "bootstrap_sd": float(np.nanstd(values, ddof=1)),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "n_examples": len(data.targets),
                "n_unique_timestamps": len(data.timestamps_ns),
                "n_positive_targets": int(data.targets.sum()),
                "n_calendar_blocks": calendar_blocks,
                "n_available_moving_blocks": available_blocks,
                "block_days": block_days,
                "block_timestamp_count": block_length,
                "n_replicates": n_replicates,
                "n_valid_replicates": int(np.isfinite(values).sum()),
                "threshold": DEFAULT_THRESHOLD,
                "decision_rule": "probability > threshold",
                "random_seed": seed,
            }
        )
    logger.info(
        "Summarized run {}: {} examples over {} timestamps from {}.",
        data.run_number,
        len(data.targets),
        len(data.timestamps_ns),
        data.path.name,
    )
    return records


def summarize_file(path: Path, n_replicates: int, block_days: int, seed: int) -> list[dict[str, str | float | int]]:
    """Compute separate bootstrap summaries for every saved run in a file."""
    return [
        record
        for data in load_predictions(path)
        for record in summarize_run(data, n_replicates, block_days, seed)
    ]


def main() -> None:
    """Run reproducible bootstrap summaries over a test-results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    parser.add_argument("--n-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.block_days < 1 or args.n_replicates < 2:
        raise ValueError("Block days must be positive and replicates must be at least two.")
    paths = sorted(args.input_directory.glob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV inputs found in {args.input_directory}.")
    records = [
        record
        for path in paths
        for record in summarize_file(path, args.n_replicates, args.block_days, args.seed)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote {} bootstrap summaries to {}.", len(records), args.output)


if __name__ == "__main__":
    main()
