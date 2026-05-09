import argparse
import os

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from loguru import logger
from omegaconf import OmegaConf
from xarray.coding.times import CFDatetimeCoder


def compute_statistics(
    zarr_path: str,
    variable_name: str,
    index_path: str | None = None,
    output_path: str | None = None,
):
    """
    Compute statistics for a given variable in a Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        variable_name (str): Name of the variable to compute statistics on.
        index_path (str, optional): Path to the index CSV file (containing timestamps) to filter data. Defaults to None.
        output_path (str, optional): Path to save the statistics as a YAML file. Defaults to None.
    """
    logger.info(f"Opening Zarr store at {zarr_path}")

    store = zarr.open(zarr_path, mode="r")

    if not isinstance(store, zarr.hierarchy.Group):
        logger.error(
            f"Zarr store at {zarr_path} is not a Group. This script is designed to work with Zarr groups."
        )
        return

    available_arrays = list(store.array_keys())
    if variable_name not in available_arrays:
        logger.error(f"Variable '{variable_name}' not found in the Zarr group.")
        logger.info(f"Available arrays are: {available_arrays}")
        return

    zarr_array = store[variable_name]
    logger.info(
        f"Found variable '{variable_name}' with shape {zarr_array.shape} and {zarr_array.ndim} dimensions."
    )

    dims = None
    if variable_name == "images":
        if zarr_array.ndim == 4:
            dims = ["timestep", "y", "x", "channel"]
        elif zarr_array.ndim == 3:
            dims = ["timestep", "y", "x"]
    elif variable_name == "timestep":
        if zarr_array.ndim == 1:
            dims = ["timestep"]

    if dims is None:
        logger.error(
            f"Could not infer dimension names for variable '{variable_name}' with {zarr_array.ndim} dimensions."
        )
        return

    logger.info(f"Assuming dimensions: {dims}")

    use_xarray = False
    ts_coords = None

    if "timestep" in available_arrays:
        try:
            time_coder = CFDatetimeCoder(use_cftime=True)
            ds = xr.open_dataset(
                zarr_path, engine="zarr", chunks="auto", decode_times=time_coder
            )
            ts_coords = ds["timestep"].values
            use_xarray = True
            logger.info("Loaded timestep with xarray (time metadata detected)")
        except Exception:
            logger.info("Time metadata not found, using raw zarr timestep")
            ts_array = store["timestep"]
            ts_coords = pd.to_datetime(ts_array[:], unit="s")

    dask_array = da.from_array(zarr_array, chunks=zarr_array.chunks)
    data_var = xr.DataArray(dask_array, dims=dims, name=variable_name)

    if index_path is not None:
        logger.info(f"Loading index from {index_path} to filter data")
        try:
            index_df = pd.read_csv(index_path)
            index_df["timestamp"] = pd.to_datetime(index_df["timestamp"]).values.astype(
                "datetime64[ns]"
            )
            index_df.set_index("timestamp", inplace=True)
            index_df.sort_index(inplace=True)

            index_df = index_df[~index_df.index.duplicated(keep="first")]
            selected_timestamps = index_df.index

            time_dim = dims[0]
            logger.info(
                f"Filtering data by {len(selected_timestamps)} timestamps using dimension '{time_dim}'"
            )

            # Convert both sets of timestamps to strings for robust matching
            # (handles cftime vs datetime64 and calendar differences)
            logger.info("Converting Zarr timestamps to strings...")
            zarr_time_strings = []
            for t in ts_coords:
                if hasattr(t, "strftime"):
                    zarr_time_strings.append(t.strftime("%Y-%m-%d %H:%M"))
                else:
                    zarr_time_strings.append(pd.to_datetime(t).strftime("%Y-%m-%d %H:%M"))

            logger.info("Converting index timestamps to strings...")
            index_time_strings = selected_timestamps.strftime("%Y-%m-%d %H:%M")

            # Build a mapping for fast lookup
            zarr_time_to_idx = {ts: i for i, ts in enumerate(zarr_time_strings)}
            
            matched_indices = []
            for ts in index_time_strings:
                if ts in zarr_time_to_idx:
                    matched_indices.append(zarr_time_to_idx[ts])

            if not matched_indices:
                logger.error("No common timestamps found after matching.")
                return

            matched_indices = np.array(matched_indices)
            logger.info(
                f"Found {len(matched_indices)} common timestamps after matching."
            )

            data_var = data_var.isel({time_dim: list(matched_indices)})

            logger.info(f"After filtering, data shape: {data_var.shape}")
        except Exception as e:
            logger.error(f"Failed to load or apply index filter: {e}")
            return

    logger.info(f"Computing log-statistics for variable '{variable_name}'")

    dask_array = data_var.data
    if not isinstance(dask_array, da.Array):
        dask_array = da.from_array(dask_array, chunks="auto")

    # Use a more stable Dask-native approach for masked statistics
    # 1. Cast to float64 to prevent overflow during sum/square operations
    dask_array = dask_array.astype(np.float64)

    # 2. Map background (0 or less) to NaN and solar disk to log10 values
    # This preserves the original array shape and chunking, which is more stable in Dask
    eps = 1e-10
    dask_array_log = da.log10(da.where(dask_array > 0, dask_array, np.nan))

    logger.info("Computing mean and std (this may take a moment for large datasets)...")
    with dask.config.set(scheduler="threads"):
        # 3. Use nan-aware reductions
        mean_val = da.nanmean(dask_array_log).compute()
        std_val = da.nanstd(dask_array_log).compute()
        
        # Min/Max of the raw data for reference
        min_val = dask_array.min().compute()
        max_val = dask_array.max().compute()

    stats = {
        "variable": variable_name,
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min_val),
        "max": float(max_val),
    }

    logger.info(f"Statistics for '{variable_name}':")
    for key, value in stats.items():
        logger.info(f"  {key.capitalize()}: {value}")

    if output_path:
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
        description="Compute statistics of a Zarr dataset."
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="/media/jhong90/storage/surya/gong_halpha_2015_2025.zarr",
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--variable_name",
        type=str,
        default="images",
        help="Name of the variable to compute statistics on.",
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
        default="./data/halpha_stat_train.yaml",
        help="Path to save the computed statistics in YAML format.",
    )
    args = parser.parse_args()

    compute_statistics(
        args.zarr_path, args.variable_name, args.index_path, args.output_path
    )
