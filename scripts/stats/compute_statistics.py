import argparse

import dask
import xarray as xr
from loguru import logger


def compute_statistics(zarr_path: str, variable_name: str):
    """
    Compute statistics for a given variable in a Zarr dataset.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        variable_name (str): Name of the variable to compute statistics on.
    """
    logger.info(f"Opening Zarr dataset at {zarr_path}")
    # Open the dataset with dask chunks for lazy loading
    ds = xr.open_zarr(zarr_path, chunks="auto")

    if variable_name not in ds.variables:
        logger.error(f"Variable '{variable_name}' not found in the dataset.")
        logger.info(f"Available variables: {list(ds.variables)}")
        return

    data_var = ds[variable_name]
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
    parser = argparse.ArgumentParser(description="Compute statistics of a Zarr dataset.")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="/media/jhong90/storage/surya/xrs_24hour_slices_g15_g16_merged.zarr",
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