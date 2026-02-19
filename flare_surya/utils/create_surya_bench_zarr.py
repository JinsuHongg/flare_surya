import os
from concurrent.futures import ThreadPoolExecutor

import hdf5plugin
import pandas as pd
import s3fs
import xarray as xr
import zarr
from tqdm import tqdm

# import zarr.codecs


def get_dataset(path, is_s3=False, s3=None):
    """Helper to open dataset efficiently without Dask overhead."""
    try:
        if is_s3:
            return xr.open_dataset(s3.open(path), engine="h5netcdf", chunks=None)
        return xr.open_dataset(path, engine="h5netcdf", chunks=None)
    except Exception as e:
        print(f"\nError loading {path}: {e}")
        return None


def create_zarr_optimized(data_local, data_aws, data_ref, zarr_path):
    channels = [
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
    ]
    num_channels = len(channels)
    s3 = s3fs.S3FileSystem(anon=True)

    # check total
    surya_bench_index = data_aws.index

    # define reference timestamps
    flare_index = data_ref.index
    shifted_minus = flare_index - pd.Timedelta(minutes=60)
    # shifted_plus = flare_index + pd.Timedelta(minutes=60)
    expanded_index = flare_index.append([shifted_minus])
    expanded_index = expanded_index.drop_duplicates().sort_values()

    # define valid index
    valid_index = surya_bench_index.intersection(expanded_index)
    total_len = len(valid_index)

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.SHUFFLE)

    if "img" not in root:
        img_arr = root.create_dataset(
            "img",
            shape=(total_len, num_channels, 4096, 4096),
            chunks=(1, num_channels, 1024, 1024),
            dtype="float32",
            compressor=compressor,
        )
        time_arr = root.create_dataset(
            "timestep", shape=(total_len,), chunks=(1024,), dtype="int64"
        )
        root.attrs["channel_names"] = channels
    else:
        img_arr = root["img"]
        time_arr = root["timestep"]

    # Resume logic: find where we left off
    existing_times = time_arr[:]
    processed_set = set(existing_times[existing_times > 0])

    print(f"Starting sequential processing for {total_len} timesteps...")

    pbar = tqdm(enumerate(valid_index), total=total_len, desc="Converting to Zarr")
    for i, idx in pbar:
        # Checkpoint check
        if int(idx.value) in processed_set:
            continue

        # Presence check
        if idx not in data_aws.index or data_aws.loc[idx, "present"] == 0:
            continue

        local_path = data_local.loc[idx, "path"]
        aws_path = data_aws.loc[idx, "path"]

        # Load data
        if os.path.exists(local_path):
            ds = get_dataset(local_path, is_s3=False, s3=s3)
        else:
            ds = get_dataset(aws_path, is_s3=True, s3=s3)

        if ds is not None:
            # Extract and write data
            img_data = ds[channels].to_array().values.astype("float32")
            img_arr[i] = img_data
            time_arr[i] = int(idx.value)

            ds.close()
            pbar.set_postfix({"timestamp": idx.strftime("%Y-%m-%d %H:%M")})


print("Consolidating metadata...")
zarr.consolidate_metadata(store)  # <--- THIS IS THE MISSING LINE
print(f"Success! Consolidated Zarr created at {zarr_path}")


if __name__ == "__main__":
    # Load indices
    df_anvil = pd.read_csv("../data/surya_input_data.csv")
    df_aws = pd.read_csv("../data/surya_aws_s3_full_index.csv")
    df_ref = pd.read_csv("../data/surya-bench-flare-forecasting/train_hour_24.csv")

    # Correct datetime parsing
    df_anvil["timestep"] = pd.to_datetime(
        df_anvil["timestep"], format="%Y-%m-%d %H:%M:%S"
    )
    df_aws["timestep"] = pd.to_datetime(df_aws["timestep"], format="%Y-%m-%d %H:%M:%S")
    df_ref["timestamp"] = pd.to_datetime(
        df_ref["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    df_anvil.set_index("timestep", inplace=True)
    df_aws.set_index("timestep", inplace=True)
    df_ref.set_index("timestamp", inplace=True)

    # Target path on Anvil scratch
    zarr_out = "/anvil/scratch/x-jhong6/data/surya_bench_train_hour_24.zarr"

    create_zarr_optimized(df_anvil, df_aws, df_ref, zarr_out)
