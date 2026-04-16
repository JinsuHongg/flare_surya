import xarray as xr
import zarr

if __name__ == "__main__":
    path = "/media/jhong90/storage/surya/xrs_24hour_slices.zarr"
    data = xr.open_dataset(path, chunks="auto", engine="zarr")
    print(data["timestep"].values)
