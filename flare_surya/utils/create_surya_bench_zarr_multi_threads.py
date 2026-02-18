import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import hdf5plugin
import pandas as pd
import s3fs
import xarray as xr
import zarr
from tqdm import tqdm


def get_dataset(path, is_s3=False, s3=None):
    """Helper to open dataset efficiently without Dask overhead."""
    try:
        if is_s3:
            return xr.open_dataset(s3.open(path), engine="h5netcdf", chunks=None)
        return xr.open_dataset(path, engine="h5netcdf", chunks=None)
    except Exception as e:
        print(f"\nError loading {path}: {e}")
        return None


def load_one(idx, data_local, data_aws, channels, s3):
    if data_aws.loc[idx, "present"] == 0:
        return None

    local_path = data_local.loc[idx, "path"]
    aws_path = data_aws.loc[idx, "path"]

    if os.path.exists(local_path):
        ds = get_dataset(local_path, is_s3=False, s3=s3)
    else:
        ds = get_dataset(aws_path, is_s3=True, s3=s3)

    if ds is None:
        return None

    try:
        img_data = ds[channels].to_array().values.astype("float32")
        ds.close()
        return (idx, img_data)
    except Exception as e:
        print(f"Error processing {idx}: {e}")
        if "ds" in locals():
            ds.close()
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

    s3 = s3fs.S3FileSystem(anon=True)
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    compressor = zarr.Blosc(cname="lz4", clevel=4, shuffle=zarr.Blosc.SHUFFLE)

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

    existing_times = time_arr[:]
    processed_set = set(existing_times[existing_times > 0])
    print(f"Starting parallel processing for {total_len} timesteps...")

    max_workers = 2
    to_process = [
        (i, idx)
        for i, idx in enumerate(valid_index)
        if int(idx.value) not in processed_set
    ]

    print(f"Processing {len(to_process)} remaining timesteps...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map handles the submission lazily
        results = executor.map(
            lambda p: load_one(p[1], data_local, data_aws, channels, s3), to_process
        )

        for i, result in enumerate(tqdm(results, total=len(to_process))):
            if result:
                idx, img_data = result
                abs_i = to_process[i][0]
                img_arr[abs_i] = img_data
                time_arr[abs_i] = int(idx.value)

    print("Consolidating metadata...")
    zarr.consolidate_metadata(store)
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
