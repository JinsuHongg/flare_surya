import argparse

import dask
import dask.array as da
import xarray as xr
import zarr
from loguru import logger


def compute_statistics(zarr_path: str, variable_name: str):
    """
    Compute statistics for a given variable in a Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        variable_name (str): Name of the variable to compute statistics on.
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

    # Infer dimensions based on variable name and shape
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

    # Wrap the Zarr array in a dask array to make it a lazy-loading array
    dask_array = da.from_array(zarr_array, chunks=zarr_array.chunks)

    data_var = xr.DataArray(dask_array, dims=dims, name=variable_name)

    logger.info(f"Computing statistics for variable '{variable_name}'")

    # Ensure dask is used for computations
    with dask.config.set(scheduler="threads"):
        mean_val = data_var.mean().compute()
        std_val = data_var.std().compute()
        min_val = data_var.min().compute()
        max_val = data_var.max().compute()

    logger.info(f"Statistics for '{variable_name}':")
    logger.info(f"  Mean: {mean_val}")
    logger.info(f"  Std Dev: {std_val}")
    logger.info(f"  Min: {min_val}")
    logger.info(f"  Max: {max_val}")


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
    args = parser.parse_args()

    compute_statistics(args.zarr_path, args.variable_name)
