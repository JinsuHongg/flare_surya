import pathlib as Path
import pandas as pd


if __name__ == "__main__":
    ref_csv = "../../data/surya-bench-flare-forecasting/data.csv"
    df = pd.read_csv(ref_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_month = [i for i in range(1, 10)]
    val_month = [10, 11]
    test_month = [12]

    year = df["timestamp"].dt.year == 2022
    train_condition = df["timestamp"].dt.month.isin(train_month)
    val_condition = df["timestamp"].dt.month.isin(val_month)
    test_condition = df["timestamp"].dt.month.isin(test_month)

    train = df[year & train_condition]
    val = df[year & val_condition]
    test = df[year & test_condition]

    save_dir = "../../data/pretrain/"

    train.to_csv(save_dir + "train.csv", index=False)
    val.to_csv(save_dir + "val.csv", index=False)
    test.to_csv(save_dir + "test.csv", index=False)
