import argparse
from loguru import logger
import time
import multiprocessing
from pathlib import Path

import cv2
import xarray as xr
import numpy as np
from tqdm import tqdm
import hdf5plugin


def resize_worker(args):
    """
    Resizes NetCDF variables and saves to a new NetCDF file, preserving metadata.
    """
    input_path, output_dir, target_size, var_names, input_root = args

    try:
        # Define output path
        relative_path = input_path.relative_to(input_root)
        output_path = output_dir / relative_path
        if output_path.exists():
            return "skipped"

        # Open Dataset
        # We assume independent variables (e.g., aia94, hmi_m) are 2D (y, x)
        with xr.open_dataset(
            input_path, engine="h5netcdf", chunks=None, cache=False
        ) as ds:

            # Prepare new dataset container
            new_ds = xr.Dataset()
            # Copy global attributes (e.g., date, telescope info)
            new_ds.attrs = ds.attrs

            # Resize Loop
            for var in var_names:
                # Skip if variable missing in this specific file
                if var not in ds:
                    continue

                # Load raw data (assuming 2D: H, W)
                # If data has a time dimension (1, H, W), squeeze it or handle it.
                data = ds[var].values.astype(np.float32)

                # Handle extra dimensions if present (e.g. Time)
                if data.ndim == 3:
                    data = data.squeeze()  # Force to 2D for cv2

                # Resize
                # cv2.resize expects (W, H), so we flip target_size tuple
                resized_data = cv2.resize(
                    data, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
                )

                # Add to New Dataset
                # We strictly define dims as y, x for consistency in ML loaders
                new_ds[var] = xr.DataArray(
                    resized_data,
                    dims=("y", "x"),
                    coords={
                        "y": np.arange(target_size[0]),
                        "x": np.arange(target_size[1]),
                    },
                    attrs=ds[var].attrs,  # Copy variable metadata (units, wavelength)
                )

            # Save with Compression
            # Compressing is CRITICAL for keeping 600k files manageable
            encoding = {
                v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in new_ds.data_vars
            }

            new_ds.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)

        return "success"

    except Exception as e:
        logger.warning(f"Failed to process {input_path}: {e}")
        return f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Parallel NetCDF Resizer (NC Output)")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to source .nc files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save .nc files"
    )
    parser.add_argument(
        "--var_name",
        type=str,
        nargs="+",  # Allow passing multiple vars as list
        default=[
            "aia94",
            "aia131",
            "aia171",
            "aia193",
            "aia211",
            "aia304",
            "aia335",
            "aia1600",
            "hmi_m",
            "hmi_bx",
            "hmi_by",
            "hmi_bz",
            "hmi_v",
        ],
        help="List of variable names to process",
    )
    parser.add_argument("--workers", type=int, default=32, help="Number of CPU cores")
    parser.add_argument("--target_size", type=int, default=512, help="Target size")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_dir} for .nc files...")
    files = list(input_dir.rglob("*.nc"))

    if not files:
        print("No .nc files found!")
        return

    print(f"Found {len(files)} files. Processing with {args.workers} workers...")

    # Fix argument structure for vars if passed as string vs list
    if isinstance(args.var_name, str):
        args.var_name = [args.var_name]

    task_args = [
        (
            f,
            output_dir,
            (args.target_size, args.target_size),
            args.var_name,
            args.input_dir,
        )
        for f in files
    ]

    start_time = time.time()

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(
            tqdm(pool.imap_unordered(resize_worker, task_args), total=len(files))
        )

    success_count = results.count("success")
    skip_count = results.count("skipped")
    error_count = len(results) - success_count - skip_count

    duration = time.time() - start_time
    print(f"\nDone! Processed {len(files)} files in {duration:.2f}s")
    print(f"Success: {success_count} | Skipped: {skip_count} | Errors: {error_count}")


if __name__ == "__main__":
    main()
