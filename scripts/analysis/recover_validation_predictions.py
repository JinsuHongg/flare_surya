"""Audit and recover per-example validation predictions from explicit checkpoints.

Examples:
    # Produce a no-write command suitable for an HPC/NASA login node.
    python scripts/analysis/recover_validation_predictions.py generate-command \
        --model-kind surya --base-config configs/nas/exp_surya.yaml \
        --experiment-config configs/nas/surya/c2w_exp.yaml \
        --checkpoint-path /mnt/checkpoints/model.ckpt --output-dir /mnt/results \
        --run-id mlp_8hour_c2w_example --event-threshold c

    # On the compute node, verify all inputs without running inference.
    python scripts/analysis/recover_validation_predictions.py export ... --dry-run

    # Recover validation probabilities from a known checkpoint.
    python scripts/analysis/recover_validation_predictions.py export ...

The exporter deliberately uses an explicit checkpoint path. A W&B run ID is
provenance metadata only; this tool never downloads a checkpoint artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


AUDIT_FIELDS = (
    "run_id",
    "threshold",
    "sampling",
    "forecasting_window_size",
    "sampling_interval",
    "best_epoch",
    "val_css",
    "val_tss",
    "val_hss",
    "entity",
    "project",
    "project_source",
    "audit_status",
    "wandb_status",
    "wandb_run_state",
    "wandb_detail",
    "cached_wandb_status",
    "cached_wandb_config_paths",
    "checkpoint_metadata_status",
    "checkpoint_metadata_detail",
    "model",
    "wandb_recorded_checkpoint_path",
    "checkpoint_environment",
    "remote_checkpoint_status",
    "local_checkpoint_path",
    "local_checkpoint_exists",
    "checkpoint_filename",
    "expected_selected_checkpoint_path",
    "expected_selected_checkpoint_filename",
    "checkpoint_resolution_status",
    "local_checkpoint_status",
    "checkpoint_candidates",
)
PREDICTION_FIELDS = (
    "timestamps",
    "reference_timestamp_ns",
    "sample_index",
    "model",
    "forecasting_window_hours",
    "sampling_strategy",
    "sampling_interval_hours",
    "threshold",
    "logit",
    "probability",
    "prediction_at_0_5",
    "predictions",
    "targets",
    "run_id",
    "checkpoint_sha256",
)
OUTPUT_TOKEN = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-runs", help="Audit ledger run provenance.")
    audit.add_argument("--experiments-csv", type=Path, required=True)
    audit.add_argument("--validation-audit", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Optional root to search read-only for files containing each run ID.",
    )
    audit.add_argument(
        "--query-wandb",
        action="store_true",
        help="Read W&B run state for rows with known entity/project provenance.",
    )
    audit.add_argument(
        "--wandb-root",
        type=Path,
        default=None,
        help="Optional local W&B cache root used to inspect run metadata only.",
    )

    command = subparsers.add_parser(
        "generate-command", help="Print an export command for a remote compute node."
    )
    add_export_arguments(command, include_dry_run=True)
    command.add_argument(
        "--execute",
        action="store_true",
        help="Generate an inference command instead of the safe default --dry-run command.",
    )

    export = subparsers.add_parser(
        "export", help="Recover validation probabilities from an explicit checkpoint."
    )
    add_export_arguments(export, include_dry_run=True)
    return parser.parse_args()


def add_export_arguments(parser: argparse.ArgumentParser, include_dry_run: bool) -> None:
    """Add arguments shared by command generation and prediction export."""
    parser.add_argument("--model-kind", choices=("surya", "baseline"), required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument(
        "--run-config",
        type=Path,
        help="Original W&B config.yaml snapshot; preferred when available.",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="", help="Optional immutable W&B/run provenance ID.")
    parser.add_argument(
        "--event-threshold",
        required=True,
        help="Event threshold label used in the output name, for example c, m, or x.",
    )
    parser.add_argument("--forecasting-window-hours", type=int, required=True)
    parser.add_argument("--sampling-strategy", required=True)
    parser.add_argument("--sampling-interval-hours", type=int, required=True)
    parser.add_argument(
        "--device", default="cuda", help="Torch device for export (for example cuda or cpu)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Permit replacing an existing output file."
    )
    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate paths/configuration/checkpoint without inference or output writes.",
        )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into records and require a header."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        return list(reader)


def audit_runs(args: argparse.Namespace) -> None:
    """Write a provenance audit from an experiment ledger and prior validation audit."""
    ledger_rows = read_rows(args.experiments_csv)
    required = {
        "ids",
        "threshold",
        "sampling",
        "forecasting_window_size",
        "sampling_interval",
    }
    if ledger_rows and not required.issubset(ledger_rows[0]):
        raise ValueError(f"{args.experiments_csv} lacks required columns {sorted(required)}.")
    validation = {row.get("experiment_id", ""): row for row in read_rows(args.validation_audit)}
    records: list[dict[str, str]] = []
    for ledger in ledger_rows:
        run_id = ledger["ids"]
        prior = validation.get(run_id, {})
        entity = prior.get("entity", "")
        project = prior.get("project", "")
        record = {
            "run_id": run_id,
            "threshold": ledger["threshold"],
            "sampling": ledger["sampling"],
            "forecasting_window_size": ledger["forecasting_window_size"],
            "sampling_interval": ledger["sampling_interval"],
            "best_epoch": prior.get("best_epoch", ""),
            "val_css": prior.get("val_css", ""),
            "val_tss": prior.get("val_tss", ""),
            "val_hss": prior.get("val_hss", ""),
            "entity": entity,
            "project": project,
            "project_source": "validation_audit" if entity and project else "unresolved",
            "audit_status": prior.get("status", "MISSING_VALIDATION_AUDIT"),
            "wandb_status": "NOT_QUERIED",
            "wandb_run_state": "",
            "wandb_detail": "",
            "cached_wandb_status": "NOT_SEARCHED",
            "cached_wandb_config_paths": "[]",
            "checkpoint_metadata_status": "NOT_SEARCHED",
            "checkpoint_metadata_detail": "",
            "local_checkpoint_status": "NOT_SEARCHED",
            "checkpoint_candidates": "[]",
        }
        if args.wandb_root is not None:
            inspect_cached_wandb_metadata(record, args.wandb_root)
        if args.query_wandb:
            query_wandb_run(record)
        if args.checkpoint_root is not None:
            find_local_checkpoints(record, args.checkpoint_root)
        records.append(record)
    write_csv(args.output, records, AUDIT_FIELDS)
    logger.info("Wrote provenance audit for {} ledger rows to {}.", len(records), args.output)


def query_wandb_run(record: dict[str, str]) -> None:
    """Read W&B run state when previous audit provenance resolves a project."""
    if not record["entity"] or not record["project"]:
        record["wandb_status"] = "PROJECT_UNRESOLVED"
        return
    try:
        import wandb

        run = wandb.Api(timeout=30).run(
            f"{record['entity']}/{record['project']}/{record['run_id']}"
        )
        record["wandb_status"] = "OK"
        record["wandb_run_state"] = str(run.state)
        artifacts = list(run.logged_artifacts())
        checkpoint_artifacts = [
            artifact.name
            for artifact in artifacts
            if "checkpoint" in artifact.name.lower()
            or "model" in artifact.name.lower()
            or artifact.type.lower() in {"model", "checkpoint"}
        ]
        if checkpoint_artifacts:
            record["checkpoint_metadata_status"] = "WAND_B_CHECKPOINT_ARTIFACT_FOUND"
            record["checkpoint_metadata_detail"] = json.dumps(checkpoint_artifacts)
        elif record["checkpoint_metadata_status"] == "NOT_SEARCHED":
            record["checkpoint_metadata_status"] = "WAND_B_CHECKPOINT_ARTIFACT_MISSING"
            record["checkpoint_metadata_detail"] = (
                "No model/checkpoint artifact was reported by the queried W&B run."
            )
    except Exception as error:  # W&B reports several transport/API exception classes.
        record["wandb_status"] = "QUERY_FAILED"
        record["wandb_detail"] = str(error)


def inspect_cached_wandb_metadata(record: dict[str, str], root: Path) -> None:
    """Inspect an explicit local W&B cache without treating it as a checkpoint store."""
    if not root.is_dir():
        record["cached_wandb_status"] = "ROOT_UNAVAILABLE"
        return
    run_dirs = sorted(root.glob(f"run-*-{record['run_id']}"))
    config_paths = [directory / "files" / "config.yaml" for directory in run_dirs]
    config_paths = [path for path in config_paths if path.is_file()]
    record["cached_wandb_config_paths"] = json.dumps([str(path) for path in config_paths])
    if not config_paths:
        record["cached_wandb_status"] = "RUN_CONFIG_MISSING"
        record["checkpoint_metadata_status"] = "CHECKPOINT_METADATA_MISSING"
        record["checkpoint_metadata_detail"] = (
            "No cached W&B run configuration was available for checkpoint provenance."
        )
        return
    record["cached_wandb_status"] = "RUN_CONFIG_FOUND"
    latest_config = config_paths[-1].read_text(errors="replace")
    record["model"] = infer_model_name(record["run_id"])
    checkpoint_path, checkpoint_filename = cached_checkpoint_reference(
        latest_config, run_dirs[-1] / "files" / "wandb-metadata.json"
    )
    record["wandb_recorded_checkpoint_path"] = checkpoint_path
    record["checkpoint_filename"] = checkpoint_filename
    record["checkpoint_environment"] = "NASA" if checkpoint_path else ""
    record["remote_checkpoint_status"] = (
        "REMOTE_PATH_RECORDED_UNVERIFIED"
        if checkpoint_path
        else "CHECKPOINT_METADATA_MISSING"
    )
    record["local_checkpoint_path"] = ""
    record["local_checkpoint_exists"] = "false"
    record["checkpoint_resolution_status"] = "CHECKPOINT_METADATA_MISSING"
    selected_path, selected_filename = derive_selected_checkpoint_path(
        latest_config,
        run_dirs[-1] / "files" / "wandb-metadata.json",
        record["best_epoch"],
    )
    record["expected_selected_checkpoint_path"] = selected_path
    record["expected_selected_checkpoint_filename"] = selected_filename
    if selected_path:
        record["checkpoint_resolution_status"] = "DERIVED_NASA_PATH_UNVERIFIED"
        record["remote_checkpoint_status"] = "DERIVED_NASA_PATH_UNVERIFIED"
    checkpoint_references: list[str] = []
    for directory in run_dirs:
        for path in directory.rglob("*"):
            if path.is_file() and (path.suffix == ".ckpt" or "checkpoint" in path.name.lower()):
                checkpoint_references.append(str(path))
        for path in (directory / "files").glob("*.json"):
            text = path.read_text(errors="replace")
            if ".ckpt" in text or "checkpoint" in text.lower():
                checkpoint_references.append(str(path))
    if checkpoint_references:
        record["checkpoint_metadata_status"] = "CACHED_CHECKPOINT_METADATA_FOUND"
        record["checkpoint_metadata_detail"] = json.dumps(sorted(set(checkpoint_references)))
    else:
        record["checkpoint_metadata_status"] = "CHECKPOINT_METADATA_MISSING"
        record["checkpoint_metadata_detail"] = (
            "Cached W&B files contain no checkpoint file or checkpoint metadata reference."
        )


def find_local_checkpoints(record: dict[str, str], root: Path) -> None:
    """Search an explicit checkpoint root without changing its contents."""
    if not root.is_dir():
        record["local_checkpoint_status"] = "ROOT_UNAVAILABLE"
        return
    candidates = sorted(root.rglob(f"*{record['run_id']}*.ckpt"))
    record["checkpoint_candidates"] = json.dumps([str(path) for path in candidates])
    if len(candidates) == 1:
        record["local_checkpoint_status"] = "ONE_MATCH"
        record["local_checkpoint_path"] = str(candidates[0])
        record["local_checkpoint_exists"] = "true"
        record["checkpoint_resolution_status"] = "LOCAL_COPY_FOUND"
    elif candidates:
        record["local_checkpoint_status"] = "MULTIPLE_MATCHES"
    else:
        record["local_checkpoint_status"] = "NO_MATCH"


def infer_model_name(run_id: str) -> str:
    """Infer the architecture label from the stable run-ID prefix only."""
    for prefix, model in (("alexnet", "AlexNet"), ("resnet18", "ResNet18"), ("mlp", "Surya")):
        if run_id.startswith(prefix):
            return model
    return ""


def cached_checkpoint_reference(config_text: str, metadata_path: Path) -> tuple[str, str]:
    """Recover a NASA-side configured checkpoint reference without resolving it.

    The cache stores W&B's flattened YAML, where ``ckpt_dir`` and ``ckpt_file``
    retain the configuration values.  The metadata ``program`` value establishes
    the NASA repository root only when it is an absolute server path.
    """
    directory_match = re.search(r"^\s+ckpt_dir: (.+)$", config_text, re.MULTILINE)
    filename_match = re.search(r"^\s+ckpt_file: (.+)$", config_text, re.MULTILINE)
    if directory_match is None or filename_match is None:
        return "", ""
    directory = directory_match.group(1).strip().strip('"\'')
    filename = filename_match.group(1).strip().strip('"\'')
    if Path(directory).is_absolute():
        return str(Path(directory) / filename), filename
    if not metadata_path.is_file():
        return str(Path(directory) / filename), filename
    program_match = re.search(
        r'"program"\s*:\s*"([^"]+)"', metadata_path.read_text(errors="replace")
    )
    if program_match is None:
        return str(Path(directory) / filename), filename
    program = Path(program_match.group(1))
    if not program.is_absolute():
        return str(Path(directory) / filename), filename
    return str(program.parents[2] / directory / filename), filename


def derive_selected_checkpoint_path(
    config_text: str, metadata_path: Path, best_epoch: str
) -> tuple[str, str]:
    """Derive the selected checkpoint using the repository callback convention.

    ``build_callbacks`` and ``build_baseline_callbacks`` use the filename
    ``{ckpt_name_tag}_lr{lr}_wd{weight_decay}_{epoch}``. Lightning expands the
    epoch token as ``epoch=<N>`` and appends ``.ckpt``. This does not prove that
    the NASA-side file remains available; it provides the exact path to verify
    there, without substituting ``last.ckpt``.
    """
    if not best_epoch:
        return "", ""
    try:
        cached = OmegaConf.to_container(OmegaConf.create(config_text), resolve=True)
        etc = cached["etc"]["value"]
        optimizer = cached["optimizer"]["value"]
        directory = str(etc["ckpt_dir"])
        tag = str(etc["ckpt_name_tag"])
        learning_rate = str(optimizer["lr"])
        weight_decay = str(optimizer["weight_decay"])
        epoch = int(float(best_epoch))
    except (KeyError, TypeError, ValueError):
        return "", ""
    filename = f"{tag}_lr{learning_rate}_wd{weight_decay}_epoch={epoch}.ckpt"
    if Path(directory).is_absolute():
        return str(Path(directory) / filename), filename
    if not metadata_path.is_file():
        return str(Path(directory) / filename), filename
    program_match = re.search(
        r'"program"\s*:\s*"([^"]+)"', metadata_path.read_text(errors="replace")
    )
    if program_match is None:
        return str(Path(directory) / filename), filename
    program = Path(program_match.group(1))
    if not program.is_absolute():
        return str(Path(directory) / filename), filename
    return str(program.parents[2] / directory / filename), filename


def compose_config(args: argparse.Namespace) -> DictConfig:
    """Load either an original W&B snapshot or a base/experiment config pair."""
    if getattr(args, "resolved_run_config", None) is not None:
        return OmegaConf.create(args.resolved_run_config)
    if args.run_config is not None:
        if not args.run_config.is_file():
            raise FileNotFoundError(f"Run configuration does not exist: {args.run_config}")
        raw = OmegaConf.to_container(OmegaConf.load(args.run_config), resolve=True)
        return OmegaConf.create(unwrap_wandb_values(raw))
    if args.base_config is None or args.experiment_config is None:
        raise ValueError("Provide --run-config or both --base-config and --experiment-config.")
    base_path = args.base_config
    experiment_path = args.experiment_config
    for path in (base_path, experiment_path):
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {path}")
    config = OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.load(experiment_path))
    OmegaConf.resolve(config)
    return config


def unwrap_wandb_values(value: Any) -> Any:
    """Convert W&B's ``{value: ...}`` config serialization back to plain data."""
    if isinstance(value, dict):
        if set(value) == {"value"}:
            return unwrap_wandb_values(value["value"])
        return {key: unwrap_wandb_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unwrap_wandb_values(item) for item in value]
    return value


def validate_export_inputs(args: argparse.Namespace, cfg: DictConfig) -> None:
    """Check recovery prerequisites without touching data batches."""
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_path}")
    if not args.output_dir.parent.exists() and not args.dry_run:
        raise FileNotFoundError(f"Output parent does not exist: {args.output_dir.parent}")
    required_data = ("valid_data_path", "valid_flare_data_path", "scalers_path")
    for key in required_data:
        path = Path(str(cfg.data[key]))
        if not path.is_file():
            raise FileNotFoundError(f"Validation input {key} does not exist: {path}")
    if args.model_kind == "surya" and not hasattr(cfg, "head"):
        raise ValueError("Surya recovery requires a config with a head section.")
    if args.model_kind == "baseline" and not hasattr(cfg, "backbone"):
        raise ValueError("Baseline recovery requires a config with a backbone section.")


def checkpoint_sha256(path: Path) -> str:
    """Return the SHA-256 of an explicit checkpoint for output provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(kind: str, cfg: DictConfig) -> Any:
    """Construct the same model types and arguments used by NAS launch scripts."""
    import torch
    from flare_surya.models import BaseLineModel, FlareSurya

    if kind == "baseline":
        return BaseLineModel(
            model_name=cfg.backbone.model_name,
            in_channels=cfg.backbone.in_channels,
            time_steps=cfg.backbone.time_steps,
            num_classes=cfg.backbone.num_classes,
            p_drop=cfg.backbone.p_drop,
            threshold=cfg.backbone.threshold,
            group_norm_groups=cfg.backbone.group_norm_groups,
            optimizer_dict=cfg.optimizer,
            loss_dict=cfg.loss,
            save_test_results_path=None,
        )
    return FlareSurya(
        img_size=cfg.backbone.img_size,
        patch_size=cfg.backbone.patch_size,
        in_chans=len(cfg.data.channels),
        embed_dim=cfg.backbone.embed_dim,
        time_embedding=cfg.backbone.time_embedding,
        depth=cfg.backbone.depth,
        num_heads=cfg.backbone.num_heads,
        mlp_ratio=cfg.backbone.mlp_ratio,
        drop_rate=cfg.backbone.drop_rate,
        dtype=torch.bfloat16,
        window_size=cfg.backbone.window_size,
        dp_rank=cfg.backbone.dp_rank,
        learned_flow=cfg.backbone.learned_flow,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
        init_weights=cfg.backbone.init_weights,
        checkpoint_layers=cfg.backbone.checkpoint_layers,
        n_spectral_blocks=cfg.backbone.n_spectral_blocks,
        rpe=cfg.backbone.rpe,
        ensemble=cfg.backbone.ensemble,
        finetune=cfg.backbone.finetune,
        nglo=cfg.backbone.nglo,
        path_weights=cfg.backbone.path_weights,
        pooling_type=cfg.backbone.pooling_type,
        head_type=cfg.head.type,
        head_layer_dict=cfg.head.hyper_parameters,
        freeze_backbone=cfg.backbone.freeze_backbone,
        lora_dict=cfg.lora,
        optimizer_dict=cfg.optimizer,
        loss_dict=cfg.loss,
        threshold=cfg.head.threshold,
        save_test_results_path=None,
        save_test_embeddings=False,
    )


def load_checkpoint(model: Any, path: Path) -> None:
    """Load a Lightning checkpoint strictly, matching trainer checkpoint semantics."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a Lightning checkpoint with state_dict.")
    model.load_state_dict(checkpoint["state_dict"], strict=True)


def move_to_device(value: Any, device: Any) -> Any:
    """Move nested batch tensors without invoking Lightning training hooks."""
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def timestamp_rows(metadata: Mapping[str, Any], batch_size: int) -> list[list[int]]:
    """Preserve full input timestamp sequences and derive scalar latest timestamps."""
    values = metadata.get("timestamps_input")
    if values is None:
        raise ValueError("Validation metadata lacks timestamps_input.")
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy().tolist()
    if not isinstance(values, Sequence) or len(values) != batch_size:
        raise ValueError("timestamps_input does not match validation batch size.")
    sequences: list[list[int]] = []
    for value in values:
        if not isinstance(value, Sequence) or not value:
            raise ValueError("Encountered an empty input timestamp sequence.")
        sequences.append([int(item) for item in value])
    return sequences


def compute_metrics(probabilities: list[float], targets: list[int], threshold: float) -> dict[str, float | int | None]:
    """Compute binary metrics with the same strict threshold convention as models."""
    tp = sum(probability > threshold and target == 1 for probability, target in zip(probabilities, targets))
    fp = sum(probability > threshold and target == 0 for probability, target in zip(probabilities, targets))
    tn = sum(probability <= threshold and target == 0 for probability, target in zip(probabilities, targets))
    fn = sum(probability <= threshold and target == 1 for probability, target in zip(probabilities, targets))
    pod = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    far = fp / (tp + fp) if tp + fp else None
    tss = pod - fpr if pod is not None and fpr is not None else None
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)
    hss = 2 * (tp * tn - fn * fp) / denominator if denominator else None
    css = math.sqrt(max(0.0, hss) * max(0.0, tss)) if hss is not None and tss is not None else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "hss": hss, "tss": tss, "css": css, "far": far, "pod": pod, "fpr": fpr}


def export_predictions(args: argparse.Namespace) -> None:
    """Run the configured validation loader once and save probability-level records."""
    cfg = compose_config(args)
    validate_export_inputs(args, cfg)
    output_path = validation_output_path(args, cfg)
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if args.dry_run:
        logger.info("Dry-run OK: checkpoint={}, validation_labels={}, output={}", args.checkpoint_path, cfg.data.valid_flare_data_path, output_path)
        return

    import torch
    from flare_surya.datamodule import FlareDataModule

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu deliberately.")
    device = torch.device(args.device)
    datamodule = FlareDataModule(cfg=cfg)
    datamodule.setup("validate")
    loader = datamodule.val_dataloader()
    model = build_model(args.model_kind, cfg)
    load_checkpoint(model, args.checkpoint_path)
    model.to(device).eval()
    checkpoint_digest = checkpoint_sha256(args.checkpoint_path)
    rows: list[dict[str, str | float | int]] = []
    probabilities: list[float] = []
    targets: list[int] = []
    with torch.no_grad():
        for data, metadata in loader:
            target_tensor = data["label"].float().view(-1)
            sequences = timestamp_rows(metadata, len(target_tensor))
            data = move_to_device(data, device)
            if args.model_kind == "baseline":
                logits = model.backbone(data).view(-1)
            else:
                tokens = model.forward_features(data)
                from flare_surya.models.criterions import FlareSSMLoss

                if isinstance(model.criterion, FlareSSMLoss):
                    logits, _ = model.head.forward_with_hidden(tokens)
                else:
                    logits = model.head(tokens)
                logits = logits.view(-1)
            batch_probabilities = torch.sigmoid(logits).float().cpu().tolist()
            batch_logits = logits.float().cpu().tolist()
            batch_targets = target_tensor.cpu().tolist()
            for sequence, logit, probability, target in zip(
                sequences, batch_logits, batch_probabilities, batch_targets
            ):
                target_int = int(target)
                if target_int not in (0, 1) or target != target_int:
                    raise ValueError(f"Encountered non-binary validation target {target!r}.")
                probability_float = float(probability)
                if not math.isfinite(probability_float) or not 0.0 <= probability_float <= 1.0:
                    raise ValueError(f"Encountered invalid probability {probability_float!r}.")
                rows.append(
                    {
                        "timestamps": json.dumps(sequence),
                        "reference_timestamp_ns": sequence[-1],
                        "sample_index": len(rows),
                        "model": "Surya" if args.model_kind == "surya" else str(cfg.backbone.model_name),
                        "forecasting_window_hours": args.forecasting_window_hours,
                        "sampling_strategy": args.sampling_strategy,
                        "sampling_interval_hours": args.sampling_interval_hours,
                        "threshold": threshold_for_kind(args.model_kind, cfg),
                        "logit": float(logit),
                        "probability": probability_float,
                        # The implementation uses `probs > threshold`, not >=.
                        "prediction_at_0_5": int(probability_float > 0.5),
                        "predictions": probability_float,
                        "targets": target_int,
                        "run_id": args.run_id,
                        "checkpoint_sha256": checkpoint_digest,
                    }
                )
                probabilities.append(probability_float)
                targets.append(target_int)
    reference_timestamps = [int(row["reference_timestamp_ns"]) for row in rows]
    if len(reference_timestamps) != len(set(reference_timestamps)):
        raise ValueError("Validation output has duplicate reference timestamps; refusing ambiguous export.")
    threshold = threshold_for_kind(args.model_kind, cfg)
    metrics = compute_metrics(probabilities, targets, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rows, PREDICTION_FIELDS)
    resolved_config_path = output_path.with_suffix(".resolved_config.yaml")
    resolved_config_text = OmegaConf.to_yaml(cfg, resolve=True)
    resolved_config_path.write_text(resolved_config_text)
    summary = {
        "run_id": args.run_id,
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "event_threshold": str(args.event_threshold).lower(),
        "forecast_target": (
            f"{str(args.event_threshold).lower()}{args.forecasting_window_hours}w"
        ),
        "base_config": str(args.base_config) if args.base_config else None,
        "experiment_config": (
            str(args.experiment_config) if args.experiment_config else None
        ),
        "run_config": str(args.run_config) if args.run_config else None,
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_sha256": hashlib.sha256(
            resolved_config_text.encode()
        ).hexdigest(),
        "split": "validation",
        "threshold": threshold,
        "n_examples": len(rows),
        "metrics": metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote {} validation predictions to {}.", len(rows), output_path)


def threshold_for_kind(kind: str, cfg: DictConfig) -> float:
    """Return the configured binary decision threshold for one model kind."""
    return float(cfg.backbone.threshold if kind == "baseline" else cfg.head.threshold)


def validation_output_path(args: argparse.Namespace, cfg: DictConfig) -> Path:
    """Build a stable validation-prediction filename from experiment conditions."""
    event_threshold = str(args.event_threshold).lower()
    forecast_target = f"{event_threshold}{args.forecasting_window_hours}w"
    model = "surya" if args.model_kind == "surya" else str(cfg.backbone.model_name).lower()
    for label, value in (("event_threshold", event_threshold), ("model", model)):
        if OUTPUT_TOKEN.fullmatch(value) is None:
            raise ValueError(f"{label} must contain only lowercase letters and digits.")
    filename = (
        f"validation_predictions_{forecast_target}_"
        f"{args.sampling_interval_hours}hour_{model}.csv"
    )
    return args.output_dir / filename


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a CSV after creating only its explicit output parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_command(args: argparse.Namespace) -> None:
    """Print a shell-safe export command without contacting remote services."""
    command = ["python", "scripts/analysis/recover_validation_predictions.py", "export"]
    for option, value in (
        ("--model-kind", args.model_kind),
        ("--checkpoint-path", args.checkpoint_path),
        ("--output-dir", args.output_dir),
        ("--event-threshold", args.event_threshold),
        ("--forecasting-window-hours", args.forecasting_window_hours),
        ("--sampling-strategy", args.sampling_strategy),
        ("--sampling-interval-hours", args.sampling_interval_hours),
        ("--device", args.device),
    ):
        command.extend((option, str(value)))
    if args.run_config is not None:
        command.extend(("--run-config", str(args.run_config)))
    else:
        if args.base_config is None or args.experiment_config is None:
            raise ValueError("Provide --run-config or both --base-config and --experiment-config.")
        command.extend(("--base-config", str(args.base_config)))
        command.extend(("--experiment-config", str(args.experiment_config)))
    if args.run_id:
        command.extend(("--run-id", args.run_id))
    if args.overwrite:
        command.append("--overwrite")
    if not args.execute:
        command.append("--dry-run")
    logger.info("Remote recovery command: {}", shlex.join(command))


def main() -> None:
    """Dispatch one explicit, non-overlapping recovery operation."""
    args = parse_args()
    if args.command == "audit-runs":
        audit_runs(args)
    elif args.command == "generate-command":
        generate_command(args)
    elif args.command == "export":
        export_predictions(args)
    else:  # argparse prevents this, but keeps the dispatcher exhaustive.
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
