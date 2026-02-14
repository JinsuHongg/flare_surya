import pandas as pd
import subprocess
import os
from pathlib import Path

if __name__ == "__main__":
    # Define paths
    dest_base_path = Path("/anvil/scratch/x-jhong6/data/surya-bench/")
    aws_index_path = "../data/surya_aws_s3_full_index.csv"
    flare_val_path = "../data/surya-bench-flare-forecasting/validation.csv"

    # Load DataFrames
    df_aws = pd.read_csv(aws_index_path)
    df_flare = pd.read_csv(flare_val_path)

    # Convert to datetime for merging
    df_aws["timestep"] = pd.to_datetime(df_aws["timestep"])
    df_flare["timestep"] = pd.to_datetime(df_flare["timestamp"])

    # Inner join to find matches
    df = df_aws.merge(df_flare, on="timestep", how="inner")

    print(f"Found {len(df)} matching files to download.")

    # Iterate through rows correctly
    for index, row in df.iterrows():
        s3_path = row["path"]
        time = row["timestep"]

        # Create a clean directory structure: year/month (e.g., 2024/05/)
        # Using pathlib makes directory creation much cleaner
        final_dest_dir = dest_base_path / f"{time.year:04d}" / f"{time.month:02d}"
        final_dest_dir.mkdir(parents=True, exist_ok=True)
        file_name = s3_path.split("/")[-1]
        local_file = final_dest_dir / file_name

        if local_file.exists():
            continue

        print(f"[{index+1}/{len(df)}] Downloading {s3_path}...")

        # Execute the AWS CLI command
        try:
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    s3_path,
                    str(final_dest_dir) + "/",
                    "--no-sign-request",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {s3_path}: {e}")
