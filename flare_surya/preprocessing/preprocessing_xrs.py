from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import numcodecs
from numpy.lib.stride_tricks import sliding_window_view


def linear_interpolation(data):
    x = np.arange(len(data))
    mask = ~np.isnan(data)
    if mask.sum() == 0:
        return data
    return np.interp(x, x[mask], data[mask])


def process_files(
    file_paths: list,
    zarr_path: Path,
    window_hours: int,
    step_hours: int,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    is_first_write: bool = True,
):
    window_size_mins = window_hours * 60
    step_size_mins = step_hours * 60
    minute_offsets = np.arange(window_size_mins)

    compressor = numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE
    )

    buffer_soft = None
    buffer_hard = None
    buffer_time = None

    for file_idx, input_xrs_path in enumerate(file_paths):
        print(
            f"\nProcessing file {file_idx + 1}/{len(file_paths)}: {input_xrs_path.name}"
        )

        with xr.open_dataset(input_xrs_path) as ds:
            time = ds["time"].values
            hard = ds["xrsa_flux"].values
            soft = ds["xrsb_flux"].values

        soft_interp = linear_interpolation(soft)
        hard_interp = linear_interpolation(hard)

        if buffer_soft is not None:
            soft_interp = np.concatenate([buffer_soft, soft_interp])
            hard_interp = np.concatenate([buffer_hard, hard_interp])
            time = np.concatenate([buffer_time, time])
            print("Successfully bridged data from previous file.")

        soft_windows = sliding_window_view(soft_interp, window_shape=window_size_mins)
        hard_windows = sliding_window_view(hard_interp, window_shape=window_size_mins)
        time_windows = sliding_window_view(time, window_shape=window_size_mins)

        buffer_soft = soft_interp[-window_size_mins:]
        buffer_hard = hard_interp[-window_size_mins:]
        buffer_time = time[-window_size_mins:]

        target_t_times = time_windows[:, -1]

        target_pd = pd.to_datetime(target_t_times)
        valid_indices = np.where(target_pd.minute == 0)[0]

        if len(valid_indices) == 0:
            print("Warning: No valid on-the-hour targets found in this file.")
            continue

        first_valid = valid_indices[0]

        soft_windows = soft_windows[first_valid::step_size_mins]
        hard_windows = hard_windows[first_valid::step_size_mins]
        aligned_t_times = target_t_times[first_valid::step_size_mins]
        aligned_target_pd = pd.to_datetime(aligned_t_times)

        if start_date is not None:
            mask = aligned_target_pd >= start_date
            soft_windows = soft_windows[mask]
            hard_windows = hard_windows[mask]
            aligned_t_times = aligned_t_times[mask]
            aligned_target_pd = aligned_target_pd[mask]

        if end_date is not None:
            mask = aligned_target_pd <= end_date
            soft_windows = soft_windows[mask]
            hard_windows = hard_windows[mask]
            aligned_t_times = aligned_t_times[mask]

        if len(aligned_t_times) == 0:
            print("Warning: No valid data after date filtering.")
            continue

        xray_windows = np.stack([soft_windows, hard_windows], axis=-1)

        ds_out = xr.Dataset(
            {
                "xray": (["timestep", "minute_offset", "channel"], xray_windows),
            },
            coords={
                "timestep": aligned_t_times,
                "minute_offset": minute_offsets,
                "channel": ["soft", "hard"],
            },
        )

        if is_first_write:
            encoding = {
                "xray": {
                    "compressor": compressor,
                    "chunks": (500, window_size_mins, 2),
                },
                "timestep": {"dtype": "int64"},
            }
            ds_out.to_zarr(zarr_path, mode="w", encoding=encoding)
            is_first_write = False
            print(f"Created new Zarr store and saved data.")
        else:
            ds_out.to_zarr(zarr_path, append_dim="timestep")
            print(f"Appended data to Zarr store.")

        yield is_first_write


def main(
    file_paths_g15: list,
    file_paths_g16: list,
    zarr_path: Path,
    window_hours: int,
    step_hours: int,
):
    g15_end_date = pd.Timestamp("2017-02-06")
    g16_start_date = pd.Timestamp("2017-02-07")

    is_first_write = True

    print("\n" + "=" * 60)
    print("Processing G15 files (2010-04-04 to 2017-02-06)")
    print("=" * 60)
    for result in process_files(
        file_paths_g15,
        zarr_path,
        window_hours,
        step_hours,
        start_date=None,
        end_date=g15_end_date,
        is_first_write=is_first_write,
    ):
        is_first_write = result

    print("\n" + "=" * 60)
    print("Processing G16 files (2017-02-07 to 2025-04-06)")
    print("=" * 60)
    for result in process_files(
        file_paths_g16,
        zarr_path,
        window_hours,
        step_hours,
        start_date=g16_start_date,
        end_date=None,
        is_first_write=is_first_write,
    ):
        is_first_write = result

    print("\nFinished successfully! All files processed and merged.")


if __name__ == "__main__":
    window_size = 24
    step_size = 1

    data_dir = Path("/media/jhong90/storage/uq_mocp/")
    file_list_g15 = sorted(list(data_dir.glob("sci_xrsf-l2-avg1m_g15_*.nc")))
    file_list_g16 = sorted(list(data_dir.glob("sci_xrsf-l2-avg1m_g16_*.nc")))

    print(f"Found {len(file_list_g15)} G15 files")
    print(f"Found {len(file_list_g16)} G16 files")

    zarr_target = Path("./xrs_24hour_slices_g15_g16_merged.zarr")

    main(file_list_g15, file_list_g16, zarr_target, window_size, step_size)
