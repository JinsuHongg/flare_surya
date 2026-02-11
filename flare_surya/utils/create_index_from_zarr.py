import zarr
import numpy as np
import pandas as pd


def create_index(zarr_path: str):

    data = zarr.open(zarr_path, mode="r")
    datetime = pd.to_datetime(data["timestep"], unit="ns")

    df = pd.DataFrame({"timestep": datetime})

    df.to_csv("surya_zarr_index_8hourly.csv")


if __name__ == "__main__":

    zarr_path = "/anvil/scratch/x-jhong6/data/surya_bench_8hourly.zarr"
    create_index(zarr_path)
