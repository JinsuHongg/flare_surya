"""Script to generate balanced datasets for binary classification.

This script finds all train*.csv files in the data/ directory, balances the data
using the label_max column via undersampling of the majority class, and saves
the balanced datasets as train*_balanced.csv files.

Usage:
    python -m flare_surya.utils.balance_data --data-dir data/
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger as lgr_logger


def find_train_csv_files(data_dir: Path) -> list[Path]:
    """Find all train*.csv files in the specified directory and subdirectories.

    Args:
        data_dir: The directory to search for CSV files.

    Returns:
        A list of Path objects pointing to train*.csv files.
    """
    return list(data_dir.rglob("train*.csv"))


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the CSV data.

    Raises:
        IOError: If the file cannot be read.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Failed to read CSV file {file_path}: {e}") from e


def balance_dataset(
    df: pd.DataFrame, label_column: str = "label_max"
) -> pd.DataFrame | None:
    """Balance a dataset by undersampling the majority class.

    Args:
        df: The input DataFrame.
        label_column: The column name containing the labels. Defaults to "label_max".

    Returns:
        A balanced DataFrame with equal class distribution, or None if already balanced.
    """
    class_counts = df[label_column].value_counts()

    if len(class_counts) < 2:
        lgr_logger.warning(f"Only one class found in dataset. Skipping balance.")
        return None

    if class_counts.iloc[0] == class_counts.iloc[1]:
        lgr_logger.info("Dataset is already balanced. Skipping.")
        return None

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_df = df[df[label_column] == majority_class]
    minority_df = df[df[label_column] == minority_class]

    minority_count = len(minority_df)
    balanced_majority = majority_df.sample(n=minority_count, random_state=42)

    balanced_df = pd.concat([minority_df, balanced_majority])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def save_balanced_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save a balanced DataFrame to a CSV file.

    Args:
        df: The balanced DataFrame to save.
        output_path: The path where the CSV file will be saved.

    Raises:
        IOError: If the file cannot be saved.
    """
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save CSV file {output_path}: {e}") from e


def process_file(file_path: Path) -> None:
    """Process a single CSV file to create a balanced dataset.

    Args:
        file_path: Path to the input CSV file.
    """
    lgr_logger.info(f"Processing file: {file_path}")

    try:
        df = load_csv(file_path)
    except IOError as e:
        lgr_logger.error(f"Skipping {file_path.name}: {e}")
        return

    if "label_max" not in df.columns:
        lgr_logger.warning(f"Skipping {file_path.name}: 'label_max' column not found.")
        return

    original_counts = df["label_max"].value_counts().to_dict()
    lgr_logger.info(f"Original class distribution: {original_counts}")

    balanced_df = balance_dataset(df)

    if balanced_df is None:
        lgr_logger.info(f"Skipping {file_path.name}: Already balanced or single-class.")
        return

    balanced_counts = balanced_df["label_max"].value_counts().to_dict()
    lgr_logger.info(f"Balanced class distribution: {balanced_counts}")

    output_path = file_path.parent / f"{file_path.stem}_balanced.csv"

    try:
        save_balanced_csv(balanced_df, output_path)
    except IOError as e:
        lgr_logger.error(f"Failed to save {file_path.name}: {e}")
        return

    lgr_logger.info(f"Saved balanced dataset to: {output_path}")


def main() -> None:
    """Main function to run the balancing script."""
    parser = argparse.ArgumentParser(
        description="Generate balanced datasets for binary classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train*.csv files (default: data)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    if not data_dir.exists():
        lgr_logger.error(f"Data directory not found: {data_dir}")
        return

    if not data_dir.is_dir():
        lgr_logger.error(f"Path is not a directory: {data_dir}")
        return

    train_files = find_train_csv_files(data_dir)

    if not train_files:
        lgr_logger.warning(f"No train*.csv files found in {data_dir}")
        return

    lgr_logger.info(f"Found {len(train_files)} train CSV file(s) to process.")

    for file_path in train_files:
        process_file(file_path)

    lgr_logger.info("Balancing complete.")


if __name__ == "__main__":
    main()
