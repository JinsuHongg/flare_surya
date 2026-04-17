import xarray as xr
import dask.array as da
from omegaconf import OmegaConf

# ── Open lazily (chunks="auto" uses dask under the hood) ─────────────────────
ds = xr.open_dataset(
    "/media/jhong90/storage/surya/xrs_24hour_slices_g15_g16_merged.zarr",
    engine="zarr",
    chunks="auto",
)
xray = ds["xray"]  # shape: (timestep, minute_offset, channel)

eps = 1e-10

stats = {}
for channel in ["soft", "hard"]:
    arr = xray.sel(channel=channel).data  # dask array, shape: (timestep, minute_offset)

    # Clip and log — still lazy at this point
    arr_log = da.log10(da.clip(arr, eps, None))

    # .compute() triggers actual computation, chunk by chunk
    print(f"Computing stats for {channel}...")
    mean = float(arr_log.mean().compute())
    std = float(arr_log.std().compute())
    mn = float(arr_log.min().compute())
    mx = float(arr_log.max().compute())

    stats[channel] = {
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
    }
    print(f"  {channel}: mean={mean:.4f}, std={std:.4f}, min={mn:.4f}, max={mx:.4f}")

# ── Save to yaml ──────────────────────────────────────────────────────────────
OmegaConf.save(OmegaConf.create(stats), "xrs_stat.yaml")
print("Saved to xrs_stat.yaml")
