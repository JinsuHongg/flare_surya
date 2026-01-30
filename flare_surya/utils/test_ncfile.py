import xarray as xr
import hdf5plugin

if __name__ == "__main__":
    # Simple test to check if NetCDF file can be opened
    test_file_path = (
        "/nobackupnfs1/sroy14/processed_data/Helio/nc/2010/05/20100523_1548.nc"
    )
    try:
        with xr.open_dataset(test_file_path, engine="h5netcdf") as ds:
            print("NetCDF file opened successfully.")
            # print(ds)
            # print(ds["aia94"])
            print(ds.data_vars)
            # print(ds.attrs)
    except Exception as e:
        print(f"Failed to open NetCDF file: {e}")
