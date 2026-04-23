import argparse
import os

import dask.array as da
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import OmegaConf


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

    ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")

    if "xray" not in ds.data_vars:
        logger.error(f"Variable 'xray' not found in the Zarr dataset.")
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

            selected_timestamps = index_df.index

            logger.info(
                f"Filtering data by {len(selected_timestamps)} timestamps using dimension 'timestep'"
            )
            ds = ds.sel(timestep=selected_timestamps)
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
        logger.info(f"  {channel}: mean={mean:.4f}, std={std:.4f}, min={mn:.4f}, max={mx:.4f}")

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
        default="./xrs_24hour_slices.zarr",
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path to the index CSV file (containing timestamps) to filter data for statistics computation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./xrs_stat.yaml",
        help="Path to save the computed statistics in YAML format.",
    )
    args = parser.parse_args()

    compute_statistics(args.zarr_path, args.index_path, args.output_path)