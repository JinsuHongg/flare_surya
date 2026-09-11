"""Launch validation prediction recovery from one self-contained YAML manifest.

The manifest names one explicit checkpoint and either embeds a resolved
configuration or composes manually maintained base and experiment configs. It
neither selects a checkpoint nor queries W&B.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

try:
    from scripts.analysis.recover_validation_predictions import compose_config, export_predictions
except ModuleNotFoundError:  # Direct execution adds scripts/analysis, not repo root.
    from recover_validation_predictions import compose_config, export_predictions


REQUIRED_KEYS = (
    "model_kind",
    "architecture",
    "run_id",
    "checkpoint_path",
    "output_dir",
    "event_threshold",
    "forecasting_window_hours",
    "sampling_strategy",
    "sampling_interval_hours",
    "validation_image_index_path",
    "validation_label_index_path",
)


def has_value(config: DictConfig, key: str) -> bool:
    """Return whether a manifest key has a concrete, non-null value."""
    return (
        key in config
        and not OmegaConf.is_missing(config, key)
        and config[key] is not None
    )


def parse_args() -> argparse.Namespace:
    """Parse the single manifest argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> DictConfig:
    """Load and validate a recovery manifest with no implicit defaults."""
    if not path.is_file():
        raise FileNotFoundError(f"Recovery manifest does not exist: {path}")
    config = OmegaConf.load(path)
    missing = [key for key in REQUIRED_KEYS if not has_value(config, key)]
    if missing:
        raise ValueError(f"Recovery manifest has unresolved required values: {missing}")
    if config.model_kind not in {"surya", "baseline"}:
        raise ValueError("model_kind must be 'surya' or 'baseline'.")
    has_resolved = has_value(config, "resolved_run_config")
    has_manual_configs = has_value(config, "base_config") and has_value(
        config, "experiment_config"
    )
    if has_resolved == has_manual_configs:
        raise ValueError(
            "Provide exactly one configuration source: resolved_run_config or "
            "both base_config and experiment_config."
        )
    return config


def export_args(config: DictConfig, args: argparse.Namespace) -> Namespace:
    """Translate a validated manifest into the shared exporter arguments."""
    checkpoint_path = Path(str(config.checkpoint_path))
    if checkpoint_path.suffix != ".ckpt":
        raise ValueError("checkpoint_path must name a .ckpt file.")
    logger.info("Recovering one explicit checkpoint: {}", checkpoint_path)
    has_resolved = has_value(config, "resolved_run_config")
    return Namespace(
        model_kind=str(config.model_kind),
        base_config=None if has_resolved else Path(str(config.base_config)),
        experiment_config=None
        if has_resolved
        else Path(str(config.experiment_config)),
        run_config=None,
        resolved_run_config=config.resolved_run_config if has_resolved else None,
        checkpoint_path=checkpoint_path,
        output_dir=Path(str(config.output_dir)),
        run_id=str(config.run_id),
        event_threshold=str(config.event_threshold),
        forecasting_window_hours=int(config.forecasting_window_hours),
        sampling_strategy=str(config.sampling_strategy),
        sampling_interval_hours=int(config.sampling_interval_hours),
        device=str(config.get("device", "cuda")),
        overwrite=bool(args.overwrite or config.get("overwrite", False)),
        dry_run=bool(args.dry_run or config.get("dry_run", False)),
    )


def validate_architecture(config: DictConfig, arguments: Namespace) -> None:
    """Ensure a baseline manifest agrees with its selected configuration."""
    architecture = str(config.architecture)
    if architecture == "Surya":
        if arguments.model_kind != "surya":
            raise ValueError("A Surya manifest must use model_kind=surya.")
        return
    resolved = compose_config(arguments)
    actual = str(resolved.backbone.model_name).lower()
    expected = architecture.lower()
    if actual != expected:
        raise ValueError(
            f"Manifest architecture {architecture!r} does not match "
            f"the archived run configuration backbone {actual!r}."
        )


def validate_validation_indices(config: DictConfig, arguments: Namespace) -> None:
    """Require manifest validation indexes to match the composed config."""
    resolved = compose_config(arguments)
    expected_image = str(config.validation_image_index_path)
    expected_label = str(config.validation_label_index_path)
    if str(resolved.data.valid_data_path) != expected_image:
        raise ValueError("validation_image_index_path does not match data.valid_data_path.")
    if str(resolved.data.valid_flare_data_path) != expected_label:
        raise ValueError(
            "validation_label_index_path does not match data.valid_flare_data_path."
        )


def main() -> None:
    """Validate one manifest and invoke the shared exporter once."""
    args = parse_args()
    config = load_manifest(args.config)
    arguments = export_args(config, args)
    validate_architecture(config, arguments)
    validate_validation_indices(config, arguments)
    export_predictions(arguments)


if __name__ == "__main__":
    main()
