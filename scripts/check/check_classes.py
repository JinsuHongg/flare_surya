#!/usr/bin/env python3
# pyright: reportMissingTypeArgument=false
# pyright: reportArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportIgnoreCommentWithoutRule=false
"""Script to check the number of classes and class distribution in the datasets.

Usage:
    python scripts/check/check_classes.py
"""

import os
import sys
from typing import Any

import hydra
from loguru import logger as lgr_logger
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
import lightning as L

# Add the project root to the Python path to allow imports from flare_surya
project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from flare_surya.datamodule.datamodule import (  # noqa: E402
    FlareDataModule,
    FlareDataModuleAWS,
    FlareDataModuleZarr,
    SuryaFluxDataModule,
)


def count_dataset_classes(dataset: Dataset) -> dict[int, int]:
    """Counts the number of samples per class in a dataset.

    This function accesses dataset metadata directly to avoid loading heavy
    NetCDF or Zarr imagery files, making the count operation very fast.

    Args:
        dataset: The PyTorch Dataset object to inspect.

    Returns:
        A dictionary mapping class labels (integers) to their sample count.
    """
    counts: dict[int, int] = {}

    if isinstance(dataset, ConcatDataset):
        for sub_ds in dataset.datasets:
            sub_counts = count_dataset_classes(sub_ds)
            for k, v in sub_counts.items():
                counts[k] = counts.get(k, 0) + v
    elif (
        hasattr(dataset, "valid_indices")
        and hasattr(dataset, "flare_index")
        and hasattr(dataset, "label_type")
    ):
        valid_indices = getattr(dataset, "valid_indices")
        flare_index = getattr(dataset, "flare_index")
        label_type = getattr(dataset, "label_type")
        for t in valid_indices:
            label = flare_index.loc[t, label_type]
            label_int = int(label)
            counts[label_int] = counts.get(label_int, 0) + 1
    else:
        # Fallback: iterate over the dataset if indices/flare_index are not present.
        # Warn the user since this might trigger actual data file reads and be slow.
        lgr_logger.warning(
            "Dataset does not have 'valid_indices' or 'flare_index' attributes. "
            "Falling back to iterating over the dataset. This may be slow."
        )
        for i in range(len(dataset)):
            sample: Any = dataset[i]
            if isinstance(sample, tuple):
                data_dict: Any = sample[0]
            else:
                data_dict = sample
            if isinstance(data_dict, dict):
                label: Any = data_dict.get("label")
                if label is not None:
                    label_int = int(label)
                    counts[label_int] = counts.get(label_int, 0) + 1

    return counts


@hydra.main(  # type: ignore
    version_base=None,
    config_path="../../configs/nas/",
    config_name="exp_surya",
)
def main(cfg: DictConfig) -> None:
    """Main function to initialize datamodule and count classes across splits.

    Args:
        cfg: The Hydra configuration dictionary.
    """
    lgr_logger.info("Initializing checking script...")

    # Determine the datamodule class to use based on configuration
    datamodule_type: str = str(cfg.get("datamodule_type", "auto"))
    datamodule_class: type[L.LightningDataModule]

    data_cfg: Any = cfg.get("data")
    if data_cfg is None:
        raise ValueError("Configuration must contain a 'data' section.")

    if datamodule_type == "auto":
        if "train_zarr_path" in data_cfg and data_cfg.get("train_zarr_path"):
            datamodule_class = FlareDataModuleZarr
            lgr_logger.info("Auto-detected datamodule type: FlareDataModuleZarr")
        elif "xrs_zarr_path" in data_cfg and data_cfg.get("xrs_zarr_path"):
            datamodule_class = SuryaFluxDataModule
            lgr_logger.info("Auto-detected datamodule type: SuryaFluxDataModule")
        elif data_cfg.get("use_aws", False):
            datamodule_class = FlareDataModuleAWS
            lgr_logger.info("Auto-detected datamodule type: FlareDataModuleAWS")
        else:
            datamodule_class = FlareDataModule
            lgr_logger.info("Auto-detected datamodule type: FlareDataModule")
    elif datamodule_type == "zarr":
        datamodule_class = FlareDataModuleZarr
        lgr_logger.info("Selected datamodule type: FlareDataModuleZarr")
    elif datamodule_type == "flux":
        datamodule_class = SuryaFluxDataModule
        lgr_logger.info("Selected datamodule type: SuryaFluxDataModule")
    elif datamodule_type == "aws":
        datamodule_class = FlareDataModuleAWS
        lgr_logger.info("Selected datamodule type: FlareDataModuleAWS")
    elif datamodule_type == "base":
        datamodule_class = FlareDataModule
        lgr_logger.info("Selected datamodule type: FlareDataModule")
    else:
        raise ValueError(f"Unknown datamodule_type: {datamodule_type}")

    # Initialize Datamodule
    datamodule = datamodule_class(cfg=cfg)
    lgr_logger.info("Setting up datasets...")
    # Setup fit and test stages to get all datasets
    datamodule.setup(stage="fit")
    try:
        datamodule.setup(stage="test")
    except Exception as e:
        lgr_logger.warning(
            f"Could not setup test stage (perhaps test config missing): {e}"
        )

    # Inspect all datasets that have been initialized
    datasets: dict[str, Dataset] = {}
    if hasattr(datamodule, "train_ds"):
        train_ds = getattr(datamodule, "train_ds")
        if train_ds is not None:
            datasets["train"] = train_ds
    if hasattr(datamodule, "val_ds"):
        val_ds = getattr(datamodule, "val_ds")
        if val_ds is not None:
            datasets["val"] = val_ds
    if hasattr(datamodule, "test_ds"):
        test_ds = getattr(datamodule, "test_ds")
        if test_ds is not None:
            datasets["test"] = test_ds

    if not datasets:
        lgr_logger.error("No datasets were initialized by the datamodule.")
        return

    # Print class counts and distributions for each split
    lgr_logger.info("=== Class Distribution Summary ===")
    for split_name, dataset in datasets.items():
        total_samples = len(dataset)
        lgr_logger.info(f"Split: {split_name} (Total samples: {total_samples})")

        if total_samples == 0:
            lgr_logger.info(f"  No samples found in {split_name} split.")
            continue

        counts = count_dataset_classes(dataset)
        sorted_classes = sorted(counts.keys())

        for cls_label in sorted_classes:
            count = counts[cls_label]
            percentage = (count / total_samples) * 100
            lgr_logger.info(f"  Class {cls_label}: {count} samples ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
