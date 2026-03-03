import pandas as pd
from pathlib import Path


def create_data_with_hour(file, base_path, hour, target_file_name):
    
    df = pd.read_csv(base_path / file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    condition = df["timestamp"].dt.hour % hour == 0
    df.loc[condition].to_csv(
            base_path / target_file_name, 
            index=False,
            date_format="%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":

    base_dir = Path("./flare_surya/data/surya-bench-flare-forecasting/12w")
    target_files = ["train_12w.csv"]
    hour = 24

    for target_file in target_files:
        create_data_with_hour(
                target_file,
                base_dir,
                hour,
                target_file.split(".")[0] + f"_{hour}.csv"
        )


