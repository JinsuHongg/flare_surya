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

def main(file_paths: list, zarr_path: Path, window_hours: int, step_hours: int):
    window_size_mins = window_hours * 60
    step_size_mins = step_hours * 60
    minute_offsets = np.arange(window_size_mins)
    
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    is_first_write = True
    
    buffer_soft = None
    buffer_hard = None
    buffer_time = None

    for file_idx, input_xrs_path in enumerate(file_paths):
        print(f"\nProcessing file {file_idx + 1}/{len(file_paths)}: {input_xrs_path.name}")
        
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
            print("Successfully bridged data from previous year.")

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
        
        # Step by 60 minutes
        soft_windows = soft_windows[first_valid::step_size_mins]
        hard_windows = hard_windows[first_valid::step_size_mins]
        aligned_t_times = target_t_times[first_valid::step_size_mins]
        
        # --- THE MERGE ---
        # Stack the two (N, 1440) arrays into a single (N, 1440, 2) array
        # axis=-1 puts the channels at the very end, perfect for PyTorch
        xray_windows = np.stack([soft_windows, hard_windows], axis=-1)
        
        # Package into the Dataset with the new "channel" dimension
        ds_out = xr.Dataset(
            {
                "xray": (["timestep", "minute_offset", "channel"], xray_windows),
            },
            coords={
                "timestep": aligned_t_times,
                "minute_offset": minute_offsets,
                "channel": ["soft", "hard"] # We explicitly name the channels here!
            }
        )
        
        # Write or Append to Zarr
        if is_first_write:
            encoding = {
                # Notice we added '2' to the chunk shape for the two channels
                "xray": {"compressor": compressor, "chunks": (500, window_size_mins, 2)},
                "timestep": {"dtype": "int64"} 
            }
            ds_out.to_zarr(zarr_path, mode="w", encoding=encoding)
            is_first_write = False
            print("Created new Zarr store and saved first year.")
        else:
            ds_out.to_zarr(zarr_path, append_dim="timestep") 
            print("Appended year to Zarr store.")
        
    print("\nFinished successfully! All 9 files processed and seamlessly merged.")

if __name__ == "__main__":
    window_size = 24
    step_size = 1
    
    data_dir = Path("/media/jhong90/storage/uq_mocp/")
    file_list = sorted(list(data_dir.glob("sci_xrsf-l2-avg1m_g16_*.nc")))
    
    zarr_target = Path("./xrs_24hour_slices.zarr")

    main(file_list, zarr_target, window_size, step_size)
