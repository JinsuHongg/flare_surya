import argparse
import os

import dask
import dask.array as da
import pandas as pd
import xarray as xr
import yaml
import zarr
from loguru import logger


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

    try:
        store = zarr.open(zarr_path, mode="r")
    except Exception as e:
        logger.error(f"Failed to open Zarr store using zarr library: {e}")
        return

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
            dims = ["time", "y", "x", "channel"]
        elif zarr_array.ndim == 3:
            dims = ["time", "y", "x"]
    elif variable_name == "timestamps":
        if zarr_array.ndim == 1:
            dims = ["time"]

    if dims is None:
        logger.error(
            f"Could not infer dimension names for variable '{variable_name}' with {zarr_array.ndim} dimensions."
        )
        return

    logger.info(f"Assuming dimensions: {dims}")

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

            selected_timestamps = index_df.index

            time_dim = dims[0]
            logger.info(
                f"Filtering data by {len(selected_timestamps)} timestamps using dimension '{time_dim}'"
            )
            data_var = data_var.sel({time_dim: selected_timestamps})

            logger.info(f"After filtering, data shape: {data_var.shape}")
        except Exception as e:
            logger.error(f"Failed to load or apply index filter: {e}")
            return

    logger.info(f"Computing statistics for variable '{variable_name}'")

    with dask.config.set(scheduler="threads"):
        mean_val = data_var.mean().compute()
        std_val = data_var.std().compute()
        min_val = data_var.min().compute()
        max_val = data_var.max().compute()

    stats = {
        "variable": variable_name,
        "mean": float(mean_val),
        "std_dev": float(std_val),
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
            with open(output_path, "w") as f:
                yaml.dump(stats, f, default_flow_style=False)
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
        default="/media/jhong90/storage/surya/gong_halpha_2016-2024.zarr",
        help="Path to the Zarr dataset.",
    )
    parser.add_argument(
        "--variable_name",
        type=str,
        default="flux",
        help="Name of the variable to compute statistics on. Common alternatives could be 'xrsb_flux' or similar.",
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
        default="./halpha_stats.yaml",
        help="Path to save the computed statistics in YAML format.",
    )
    args = parser.parse_args()

    compute_statistics(
        args.zarr_path, args.variable_name, args.index_path, args.output_path
    )