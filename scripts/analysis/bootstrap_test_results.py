"""Compute temporal block-bootstrap confidence intervals for test predictions.

Example:
    python scripts/analysis/bootstrap_test_results.py \
        --input Surya=/mnt/storage/surya/test_results/surya_8hour_c_ce_nosample.csv \
        --input AlexNet=/mnt/storage/surya/test_results/alexnet_c_8hour_test_result.csv \
        --output results/audits/bootstrap_24h_c_8h.csv \
        --block-days 7 --n-replicates 2000 --seed 1004

Each input file must contain ``timestamps``, ``predictions``, and ``targets``.
The latest timestamp in each input sequence anchors temporal blocks. Inputs are
bootstrapped independently; do not compare confidence intervals as a paired
model comparison. Paired comparisons require a separately verified alignment
of timestamps, targets, thresholds, and experiment conditions.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger


NANOSECONDS_PER_DAY = 86_400_000_000_000
METRICS = ("hss", "tss", "far", "pod")


@dataclass(frozen=True)
class TestPredictions:
    """Parsed predictions for one fixed test configuration.

    Args:
        name: User-provided configuration label.
        timestamps_ns: Latest input timestamp for each prediction in ns.
        probabilities: Positive-class probabilities.
        targets: Binary targets.
    """

    name: str
    timestamps_ns: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray


def parse_input(value: str) -> tuple[str, Path]:
    """Parse a ``NAME=PATH`` command-line input specification."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Inputs must use NAME=PATH format.")
    return name, Path(path)


def parse_timestamp(value: str) -> int:
    """Return the latest nanosecond timestamp in one saved result row."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid timestamps value: {value!r}") from error
    if isinstance(parsed, (list, tuple)):
        if not parsed:
            raise ValueError("Timestamp sequence is empty.")
        parsed = parsed[-1]
    return int(float(parsed))


def load_predictions(name: str, path: Path) -> TestPredictions:
    """Load and validate one per-example test-result CSV.

    Raises:
        ValueError: If required columns, binary targets, or unique timestamps
            are not present.
    """
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
    timestamp_array = np.asarray(timestamps, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int8)
    if not len(timestamp_array):
        raise ValueError(f"{path} contains no test rows.")
    if not np.isin(target_array, (0, 1)).all():
        raise ValueError(f"{path} contains non-binary targets.")
    if len(np.unique(timestamp_array)) != len(timestamp_array):
        raise ValueError(f"{path} contains duplicate anchor timestamps.")
    order = np.argsort(timestamp_array)
    return TestPredictions(
        name=name,
        timestamps_ns=timestamp_array[order],
        probabilities=np.asarray(probabilities, dtype=float)[order],
        targets=target_array[order],
    )


def scores(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute thresholded HSS, TSS, FAR, and POD from one sample."""
    predicted = probabilities > threshold
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
    return {"hss": hss, "tss": tss, "far": far, "pod": pod}


def bootstrap_scores(
    data: TestPredictions, threshold: float, block_days: int, n_replicates: int, seed: int
) -> dict[str, np.ndarray]:
    """Draw non-overlapping temporal blocks with replacement and score them."""
    block_width = block_days * NANOSECONDS_PER_DAY
    block_ids = (data.timestamps_ns - data.timestamps_ns.min()) // block_width
    blocks = [np.flatnonzero(block_ids == block) for block in np.unique(block_ids)]
    if len(blocks) < 2:
        raise ValueError("At least two temporal blocks are required for bootstrap.")
    rng = np.random.default_rng(seed)
    results = {metric: np.empty(n_replicates) for metric in METRICS}
    for replicate in range(n_replicates):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in selected])
        values = scores(data.probabilities[indices], data.targets[indices], threshold)
        for metric, value in values.items():
            results[metric][replicate] = value
    return results


def main() -> None:
    """Run bootstrap analyses and write point estimates with percentile CIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1004)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.block_days < 1 or args.n_replicates < 2:
        raise ValueError("Threshold must be in (0, 1); block days and replicates must be positive.")
    if len({name for name, _ in args.input}) != len(args.input):
        raise ValueError("Each --input name must be unique.")

    records: list[dict[str, str | float | int]] = []
    for name, path in args.input:
        data = load_predictions(name, path)
        bootstrap = bootstrap_scores(data, args.threshold, args.block_days, args.n_replicates, args.seed)
        point = scores(data.probabilities, data.targets, args.threshold)
        n_blocks = len(np.unique((data.timestamps_ns - data.timestamps_ns.min()) // (args.block_days * NANOSECONDS_PER_DAY)))
        for metric in METRICS:
            valid_replicates = int(np.isfinite(bootstrap[metric]).sum())
            if not valid_replicates:
                raise ValueError(f"No valid bootstrap replicates for {name} {metric}.")
            lower, upper = np.nanpercentile(bootstrap[metric], (2.5, 97.5))
            records.append({"configuration": name, "source_file": str(path), "metric": metric, "point_estimate": point[metric], "ci_95_lower": lower, "ci_95_upper": upper, "n_examples": len(data.targets), "n_temporal_blocks": n_blocks, "block_days": args.block_days, "n_replicates": args.n_replicates, "n_valid_replicates": valid_replicates, "threshold": args.threshold, "seed": args.seed})
        logger.info("Bootstrapped {} examples across {} blocks for {}.", len(data.targets), n_blocks, name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote {} metric intervals to {}.", len(records), args.output)


if __name__ == "__main__":
    main()
