import pandas as pd
from pathlib import Path


def create_data_with_hour(
    file: str, base_path: Path, source_freq: str, hour: int, target_file_name: str
) -> None:
    df = pd.read_csv(
        base_path / source_freq / file,
        parse_dates=["timestamp"],
    )
    df[df["timestamp"].dt.hour % hour == 0].to_csv(
        base_path / f"{source_freq}" / target_file_name,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    base_dir = Path("./data/surya-bench-flare-forecasting/")
    source_freq = "X24w"
    hour = 24
    stem_suffix = f"_freq{hour}.csv"

    target_files = ["train"]

    for stem in target_files:
        create_data_with_hour(
            f"{stem}_{source_freq}.csv",
            base_dir,
            source_freq,
            hour,
            f"{stem}_{source_freq}{stem_suffix}",
        )
