import argparse
import os

import zarr
from loguru import logger


def consolidate_zarr_metadata(zarr_path: str):
    """
    Consolidates metadata for a Zarr store.

    This creates a .zmetadata file in the Zarr store directory,
    which can significantly speed up opening the store.

    Args:
        zarr_path (str): Path to the Zarr store to consolidate.
    """
    if not os.path.isdir(zarr_path):
        logger.error(f"Zarr path provided is not a directory: {zarr_path}")
        return

    logger.info(f"Consolidating metadata for Zarr store at: {zarr_path}")
    try:
        zarr.consolidate_metadata(zarr_path)
        logger.success(f"Successfully consolidated metadata for {zarr_path}")
    except Exception as e:
        logger.error(f"Failed to consolidate metadata: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate metadata for a Zarr store to improve read performance."
    )
    parser.add_argument(
        "zarr_path",
        type=str,
        help="Path to the Zarr store to consolidate.",
    )
    args = parser.parse_args()

    consolidate_zarr_metadata(args.zarr_path)
