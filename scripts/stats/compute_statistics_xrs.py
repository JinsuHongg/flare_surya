import argparse
import os

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf
from xarray.coders import CFDatetimeCoder


def compute_statistics(
    zarr_path: str,
    index_path: str | None = None,
    output_path: str | None = None,
):
    """
    Compute statistics for XRS data in a Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        index_path (str, optional): Path to the index CSV file (containing timestamps) to filter data. Defaults to None.
        output_path (str, optional): Path to save the statistics as a YAML file. Defaults to None.
    """
    logger.info(f"Opening Zarr dataset at {zarr_path}")

    time_coder = CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_times=time_coder)

    if "xray" not in ds.data_vars:
        logger.error("Variable 'xray' not found in the Zarr dataset.")
        return

    xray = ds["xray"]
    logger.info(f"Found variable 'xray' with shape {xray.shape} and dims {xray.dims}")

    if index_path is not None:
        logger.info(f"Loading index from {index_path} to filter data")
        try:
            index_df = pd.read_csv(index_path)
            index_df["timestamp"] = pd.to_datetime(index_df["timestamp"]).values.astype(
                "datetime64[ns]"
            )
            index_df.set_index("timestamp", inplace=True)
            index_df.sort_index(inplace=True)

            # Drop duplicate timestamps from CSV
            if index_df.index.has_duplicates:
                num_dups = index_df.index.duplicated().sum()
                logger.warning(
                    f"Index CSV has {num_dups} duplicate timestamps, keeping first occurrence"
                )
                index_df = index_df[~index_df.index.duplicated(keep="first")]
            selected_timestamps = index_df.index

            # Handle duplicates in the dataset by keeping first occurrence
            timestep_index = ds.timestep.to_index()
            if timestep_index.has_duplicates:
                num_dups = pd.Series(timestep_index.values).duplicated().sum()
                logger.warning(
                    f"Dataset 'timestep' has {num_dups} duplicate values, keeping first occurrence"
                )
                _, index = np.unique(timestep_index.values, return_index=True)
                ds = ds.isel(timestep=index)

            # Convert dataset timesteps to pandas datetime64 for intersection
            # cftime objects need to be converted to POSIX timestamps first
            ds_timesteps_raw = ds.timestep.values
            if hasattr(ds_timesteps_raw[0], "strftime"):
                ds_timesteps = pd.to_datetime(
                    [t.strftime("%Y-%m-%d %H:%M:%S") for t in ds_timesteps_raw]
                )
            else:
                ds_timesteps = pd.to_datetime(ds_timesteps_raw)
            index_timesteps = pd.to_datetime(selected_timestamps)

            # Debug: print sample timestamps
            logger.info(f"Index sample: {index_timesteps[:3].tolist()}")
            logger.info(f"Dataset sample: {ds_timesteps[:3].tolist()}")
            logger.info(f"Index range: {index_timesteps.min()} to {index_timesteps.max()}")
            logger.info(f"Dataset range: {ds_timesteps.min()} to {ds_timesteps.max()}")

            # Find intersection
            ds_set = set(ds_timesteps)
            index_set = set(index_timesteps)
            common_timesteps = ds_set.intersection(index_set)

            if len(common_timesteps) == 0:
                logger.error("No common timestamps found between index and dataset")
                return

            logger.info(
                f"Found {len(common_timesteps)} common timestamps out of "
                f"{len(index_timesteps)} index and {len(ds_timesteps)} dataset"
            )

            # Filter dataset to common timesteps
            common_mask = np.array([t in common_timesteps for t in ds_timesteps])
            ds = ds.isel(timestep=common_mask)
            xray = ds["xray"]

            logger.info(f"After filtering, data shape: {xray.shape}")
        except Exception as e:
            logger.error(f"Failed to load or apply index filter: {e}")
            return

    eps = 1e-10

    stats = {}
    for channel in ["soft", "hard"]:
        arr = xray.sel(channel=channel).data

        arr_log = da.log10(da.clip(arr, eps, None))

        logger.info(f"Computing stats for {channel}...")
        mean = float(arr_log.mean().compute())
        std = float(arr_log.std().compute())
        mn = float(arr_log.min().compute())
        mx = float(arr_log.max().compute())

        stats[channel] = {
            "mean": mean,
            "std": std,
            "min": mn,
            "max": mx,
        }
        logger.info(
            f"  {channel}: mean={mean:.4f}, std={std:.4f}, min={mn:.4f}, max={mx:.4f}"
        )

    output_path = output_path or "xrs_stat.yaml"
    logger.info(f"Saving statistics to {output_path}")
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(OmegaConf.create(stats), output_path)
        logger.success(f"Successfully saved statistics to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save statistics to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statistics of XRS Zarr dataset."
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="./data/xrs_24hour_slices_v2.zarr",
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="./data/pretrain/train.csv",
        help="Path to the index CSV file (containing timestamps) to filter data for statistics computation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/xrs_stat_train.yaml",
        help="Path to save the computed statistics in YAML format.",
    )
    args = parser.parse_args()

    compute_statistics(args.zarr_path, args.index_path, args.output_path)

