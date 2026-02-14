import os
import s3fs
import xarray as xr
import zarr
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
REQUIRED_VARS = set(
    [
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
)


def get_dataset(path, is_s3=False, s3_fs=None):
    try:
        if is_s3:
            # Using h5netcdf with chunks=None to prevent xarray from
            # trying to be "smart" with Dask, which saves memory here.
            return xr.open_dataset(s3_fs.open(path), engine="h5netcdf", chunks=None)
        return xr.open_dataset(path, engine="h5netcdf", chunks=None)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def detect_file_order(df_local, df_aws, s3):
    print("Detecting native channel order...")
    # Check local first
    for _, row in df_local.head(20).iterrows():
        if os.path.exists(row["path"]):
            ds = get_dataset(row["path"])
            if ds:
                order = [v for v in list(ds.data_vars) if v in REQUIRED_VARS]
                ds.close()
                return order
    # Fallback to AWS
    for _, row in df_aws.head(20).iterrows():
        ds = get_dataset(row["path"], is_s3=True, s3_fs=s3)
        if ds:
            order = [v for v in list(ds.data_vars) if v in REQUIRED_VARS]
            ds.close()
            return order
    raise ValueError("Could not detect channel order!")


def load_and_extract(l_path, a_path, channel_order, s3):
    """
    Encapsulated load to ensure local variables go out of scope quickly.
    """
    ds = None
    if l_path and os.path.exists(l_path):
        ds = get_dataset(l_path, is_s3=False)
    if ds is None and a_path:
        ds = get_dataset(a_path, is_s3=True, s3_fs=s3)

    if ds is None:
        return None

    try:
        # Pre-allocate (13, 4096, 4096)
        img_data = np.zeros((len(channel_order), 4096, 4096), dtype="float32")
        for ch_idx, c in enumerate(channel_order):
            if c in ds:
                # .values triggers the actual read into memory
                img_data[ch_idx, :, :] = ds[c].values
        ds.close()
        return img_data
    except Exception as e:
        print(f"Error extracting data: {e}")
        if ds:
            ds.close()
        return None


def create_zarr_sequential(df_local, df_aws, df_ref, zarr_path):
    # Index alignment (same as your original logic)
    valid_index = df_aws.index.intersection(
        df_ref.index.append(
            [
                df_ref.index - pd.Timedelta(minutes=60),
                df_ref.index + pd.Timedelta(minutes=60),
            ]
        ).drop_duplicates()
    ).sort_values()

    total_len = len(valid_index)
    s3 = s3fs.S3FileSystem(anon=True)
    channel_order = detect_file_order(df_local, df_aws, s3)
    num_channels = len(channel_order)

    # Zarr Setup
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
        root.attrs["channel_names"] = channel_order
    else:
        img_arr = root["img"]
        time_arr = root["timestep"]

    # Resume logic: find where we left off
    existing_times = time_arr[:]
    processed_set = set(existing_times[existing_times > 0])

    print(f"Starting sequential processing for {total_len} timesteps...")

    for i, idx in enumerate(tqdm(valid_index, desc="Writing Zarr")):
        ts_int = int(idx.value)
        if ts_int in processed_set:
            continue

        l_path = df_local.loc[idx, "path"] if idx in df_local.index else None
        a_path = df_aws.loc[idx, "path"] if idx in df_aws.index else None

        data = load_and_extract(l_path, a_path, channel_order, s3)

        if data is not None:
            img_arr[i] = data
            time_arr[i] = ts_int

            # Explicit cleanup
            del data
            if i % 10 == 0:
                gc.collect()


if __name__ == "__main__":
    # Load and prep dataframes
    df_anvil = pd.read_csv("../data/surya_input_data.csv")
    df_aws = pd.read_csv("../data/surya_aws_s3_full_index.csv")
    df_ref = pd.read_csv("../data/surya-bench-flare-forecasting/train_hour_12.csv")

    for df, col in [
        (df_anvil, "timestep"),
        (df_aws, "timestep"),
        (df_ref, "timestamp"),
    ]:
        df[col] = pd.to_datetime(df[col], format="mixed")
        df.set_index(col, inplace=True)

    zarr_out = "/anvil/scratch/x-jhong6/data/surya_bench_train_cadence_12hour.zarr"

    create_zarr_sequential(df_anvil, df_aws, df_ref, zarr_out)

    print("Consolidating metadata...")
    zarr.consolidate_metadata(zarr_out)
    print("Done!")
