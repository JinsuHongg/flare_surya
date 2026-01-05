import os
import argparse
import pandas as pd
from loguru import logger as lgr_logger


def main(start: str, end: str, base_dir: str, save_path: str):
    """
    Generate a list of file paths for NetCDF files between start and end timestamps,
    check if they exist, and store results in a dictionary.
    """
    data = {"path": [], "timestep": [], "present": []}

    for time in pd.date_range(start, end, freq="12min"):
        # Build file path
        file_path = os.path.join(
            base_dir,
            str(time.year),
            f"{time.month:02d}",
            f"{time.year:04d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}.nc",
        )

        # Format timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Check existence
        present = int(os.path.exists(file_path))

        # Append to dictionary
        data["path"].append(file_path)
        data["timestep"].append(timestamp)
        data["present"].append(present)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, "surya_input_data.csv"), index=False)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check presence of NetCDF files over a date range."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01 00:00:00",
        help="Start datetime (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31 23:59:59",
        help="End datetime (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/anvil/scratch/x-jhong6/data/surya-bench",
        help="Base directory for files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../data/",
        help="Base directory for files",
    )

    args = parser.parse_args()

    result = main(args.start, args.end, args.base_dir, args.save_path)

    # print a summary
    lgr_logger.info(
        f"Total files checked: {len(result['path'])}, Files present: {sum(result['present'])}"
    )
