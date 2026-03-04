import pandas as pd
from pathlib import Path

def main(new_data_path, tag):
    # read new data
    df_new = pd.read_csv(new_data_path / "data_thres_c_24hour_window.csv")

    # read existing datasplit
    df_train = pd.read_csv((new_data_path.parent / "train.csv").resolve())
    df_validation = pd.read_csv((new_data_path.parent / "validation.csv").resolve())
    df_leaky_val = pd.read_csv((new_data_path.parent / "leaky_validation.csv").resolve())
    df_test = pd.read_csv((new_data_path.parent / "test.csv").resolve())
    
    # convert str to datetime
    for df in [df_new, df_train, df_validation, df_leaky_val, df_test]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # split data based on existing file
    df_new_train = pd.merge(df_new, df_train[["timestamp"]], on="timestamp", how="inner")
    df_new_val = pd.merge(df_new, df_validation[["timestamp"]], on="timestamp", how="inner")
    df_new_leaky_val = pd.merge(df_new, df_leaky_val[["timestamp"]], on="timestamp", how="inner")
    df_new_test = pd.merge(df_new, df_test[["timestamp"]], on="timestamp", how="inner")

    # save new split
    df_new_train.to_csv(new_data_path / f"train_{tag}.csv", index=False)
    df_new_val.to_csv(new_data_path / f"val_{tag}.csv", index=False)
    df_new_leaky_val.to_csv(new_data_path / f"leaky_val_{tag}.csv", index=False)
    df_new_test.to_csv(new_data_path / f"test_{tag}.csv", index=False)
    print("complete data split!")

if __name__ == "__main__":

    tag = "C24w" 
    new_data_path = Path(f"./flare_surya/data/surya-bench-flare-forecasting/{tag}/")
    main(new_data_path, tag)
