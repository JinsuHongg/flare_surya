"""Retrieve validation checkpoint metrics for W&B experiments.

The script resolves every full cloud run ID from ``experiment_ids/exp_ids.csv``
using the matching local W&B configuration, then writes the validation epoch
with the highest combined skill score (CSS) to a CSV audit file.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import wandb
from loguru import logger
from wandb.apis.public import Run
from wandb.errors import CommError


WANDB_ROOT = Path("/mnt/storage/surya/wandb")
METRIC_KEYS = ("epoch", "val/css", "val/tss", "val/hss")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        default=Path("experiment_ids/exp_ids.csv"),
        help="CSV containing experiment IDs and conditions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for the cloud-validation audit.",
    )
    parser.add_argument(
        "--prior-audit",
        type=Path,
        default=None,
        help=(
            "Existing audit CSV whose entity/project paths can supplement local "
            "W&B configurations. Defaults to --output when it already exists."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based row offset after optional window filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of experiment rows to retrieve.",
    )
    parser.add_argument(
        "--forecast-windows",
        nargs="+",
        default=["2", "24"],
        help="Forecast-window sizes to retrieve (default: 2 24).",
    )
    return parser.parse_args()


def load_experiments(
    csv_path: Path,
    forecast_windows: set[str],
) -> list[dict[str, str]]:
    """Load requested experiment conditions from the experiment ledger.

    Args:
        csv_path: Experiment ledger path.
        forecast_windows: Forecast-window sizes to retain.

    Returns:
        Ledger rows matching the requested forecast windows.
    """
    with csv_path.open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["forecasting_window_size"] in forecast_windows
        ]


def find_cloud_paths(experiment_id: str) -> list[tuple[str, str]]:
    """Return unique entity/project pairs from local W&B run configurations.

    Args:
        experiment_id: Full cloud W&B run ID from the experiment ledger.

    Returns:
        Candidate entity/project pairs, ordered by local run timestamp.
    """
    paths: list[tuple[str, str]] = []
    for run_dir in sorted(WANDB_ROOT.glob(f"run-*-{experiment_id}")):
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        config_text = config_path.read_text()
        entity_match = re.search(r"^        entity: (.+)$", config_text, re.MULTILINE)
        project_match = re.search(
            r"^        project: (.+)$", config_text, re.MULTILINE
        )
        if entity_match is None or project_match is None:
            continue
        candidate = (entity_match.group(1).strip(), project_match.group(1).strip())
        if candidate not in paths:
            paths.append(candidate)
    return paths


def load_audit_paths(audit_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load valid entity/project paths keyed by cloud run ID from an audit CSV.

    Args:
        audit_path: Existing cloud-validation audit CSV.

    Returns:
        Candidate W&B paths for each cloud run ID. Missing audit files and rows
        without a complete path are ignored.
    """
    if not audit_path.exists():
        return {}

    paths_by_run: dict[str, list[tuple[str, str]]] = {}
    with audit_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = row.get("cloud_run_id", "")
            entity = row.get("entity", "")
            project = row.get("project", "")
            if not all((run_id, entity, project)):
                continue
            candidate = (entity, project)
            paths_by_run.setdefault(run_id, [])
            if candidate not in paths_by_run[run_id]:
                paths_by_run[run_id].append(candidate)
    return paths_by_run


def infer_candidate_paths(experiment: dict[str, str]) -> list[tuple[str, str]]:
    """Infer candidate W&B entity/project paths from experiment conditions.

    Args:
        experiment: One experiment-ledger row.

    Returns:
        Likely entity/project pairs for the given forecasting conditions.
    """
    window = experiment.get("forecasting_window_size")
    sampling = experiment.get("sampling")
    if window == "2" and sampling == "no":
        return [("gsu-dmlab", "surya-flare-2hwindow-nosample")]
    if window == "2" and sampling == "under":
        return [("gsu-dmlab", "surya-flare-2hwindow-undersample")]
    if window == "24" and sampling == "no":
        return [("gsu-dmlab", "flareforecasting-nosampling")]
    if window == "24" and sampling == "under":
        return [("gsu-dmlab", "flareforecasting-undersampling")]
    return []


def resolve_run(
    api: wandb.Api,
    entity: str,
    project: str,
    experiment_id: str,
) -> tuple[Run | None, str]:
    """Resolve a W&B run by cloud run ID or fallback to display name.

    Args:
        api: Authenticated W&B public API client.
        entity: W&B entity name.
        project: W&B project name.
        experiment_id: Target experiment identifier from the ledger.

    Returns:
        A tuple of (Run object or None, error detail string).
    """
    try:
        return api.run(f"{entity}/{project}/{experiment_id}"), ""
    except CommError as error:
        direct_error = str(error)

    # Fallback: search by display name (Run Name) in target project
    try:
        runs = list(
            api.runs(f"{entity}/{project}", filters={"display_name": experiment_id})
        )
        if runs:
            runs.sort(key=lambda item: str(item.created_at), reverse=True)
            return runs[0], ""
    except CommError as error:
        return None, str(error)

    return None, direct_error


def valid_validation_rows(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter history rows that contain all required validation metrics."""
    return [
        row
        for row in history
        if all(row.get(key) is not None for key in METRIC_KEYS)
    ]


def latest_validation_rows_by_epoch(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the most recent complete validation record for each epoch.

    W&B runs can be resumed under one immutable run ID, resulting in multiple
    validation records for an epoch. More recent records are determined by
    W&B step, then timestamp, then their response order.

    Args:
        rows: Complete validation records from W&B history.

    Returns:
        One latest validation record per epoch.
    """
    latest_rows: dict[int, tuple[tuple[float, float, int], dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        step = row.get("_step")
        timestamp = row.get("_timestamp")
        recency = (
            float(step) if isinstance(step, int | float) else float("-inf"),
            float(timestamp)
            if isinstance(timestamp, int | float)
            else float("-inf"),
            index,
        )
        epoch = int(row["epoch"])
        current = latest_rows.get(epoch)
        if current is None or recency > current[0]:
            latest_rows[epoch] = (recency, row)
    return [latest_rows[epoch][1] for epoch in sorted(latest_rows)]


def retrieve_experiment(
    api: wandb.Api,
    experiment: dict[str, str],
    audit_paths: Iterable[tuple[str, str]],
) -> dict[str, str | int | float]:
    """Retrieve the max-CSS validation record for one experiment.

    Args:
        api: Authenticated W&B public API client.
        experiment: One experiment-ledger row.

    Returns:
        A provenance-rich audit record. Retrieval errors are represented in the
        ``status`` field rather than raising, so the full ledger is audited.
    """
    experiment_id = experiment["ids"]
    base_record: dict[str, str | int | float] = {
        "experiment_id": experiment_id,
        "threshold": experiment["threshold"],
        "sampling": experiment["sampling"],
        "forecasting_window_size": experiment["forecasting_window_size"],
        "sampling_interval": experiment["sampling_interval"],
        "cloud_run_id": experiment_id,
        "entity": "",
        "project": "",
        "run_state": "",
        "best_epoch": "",
        "val_css": "",
        "val_tss": "",
        "val_hss": "",
        "validation_record_count": 0,
        "status": "CLOUD_MISSING",
        "detail": "No local W&B configuration supplied an entity/project path.",
    }
    candidates = find_cloud_paths(experiment_id)
    for candidate in audit_paths:
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in infer_candidate_paths(experiment):
        if candidate not in candidates:
            candidates.append(candidate)
    for entity, project in candidates:
        run, error_detail = resolve_run(api, entity, project, experiment_id)
        if run is None:
            base_record["detail"] = error_detail
            continue

        # Validation is logged once per epoch. Requesting a dense sampled history
        # avoids scanning the high-frequency training-step records remotely while
        # retaining all expected validation records (at most 50 epochs here).
        history_records = run.history(
            keys=list(METRIC_KEYS), samples=1_000, pandas=False
        )
        rows = valid_validation_rows(history_records)
        canonical_rows = latest_validation_rows_by_epoch(rows)
        base_record.update(
            {
                "cloud_run_id": str(run.id),
                "entity": entity,
                "project": project,
                "run_state": run.state,
                "validation_record_count": len(rows),
            }
        )
        if not rows:
            base_record.update(
                {
                    "status": "NO_VALIDATION_HISTORY",
                    "detail": "Cloud run has no complete validation metric rows.",
                }
            )
            return base_record

        best = max(
            canonical_rows,
            key=lambda row: (float(row["val/css"]), int(row["epoch"])),
        )
        base_record.update(
            {
                "best_epoch": int(best["epoch"]),
                "val_css": float(best["val/css"]),
                "val_tss": float(best["val/tss"]),
                "val_hss": float(best["val/hss"]),
                "status": "OK",
                "detail": "",
            }
        )
        return base_record
    return base_record


def write_audit(records: list[dict[str, str | int | float]], output_path: Path) -> None:
    """Write cloud-validation audit records to CSV.

    Args:
        records: Retrieval results for the requested ledger rows.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else [
        "experiment_id",
        "threshold",
        "sampling",
        "forecasting_window_size",
        "sampling_interval",
        "cloud_run_id",
        "entity",
        "project",
        "run_state",
        "best_epoch",
        "val_css",
        "val_tss",
        "val_hss",
        "validation_record_count",
        "status",
        "detail",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Retrieve requested W&B validation histories and write their audit."""
    args = parse_args()
    experiments = load_experiments(args.experiments_csv, set(args.forecast_windows))
    selected = experiments[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        logger.warning("No experiment rows selected for retrieval.")
        write_audit([], args.output)
        return

    api = wandb.Api(timeout=30)
    prior_audit = args.prior_audit or args.output
    audit_paths = load_audit_paths(prior_audit)
    records = []
    for index, experiment in enumerate(selected, start=args.offset + 1):
        logger.info("Retrieving {}/{}: {}", index, len(experiments), experiment["ids"])
        records.append(
            retrieve_experiment(api, experiment, audit_paths.get(experiment["ids"], []))
        )
    write_audit(records, args.output)
    successful = sum(record["status"] == "OK" for record in records)
    logger.info("Retrieved {}/{} cloud validation histories.", successful, len(records))


if __name__ == "__main__":
    main()
