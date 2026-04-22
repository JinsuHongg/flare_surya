import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger as lgr_logger
from omegaconf import DictConfig
from torch.utils.data import Dataset


class SolarPretrainDataset(Dataset):
    """
    A dataset for self-supervised pre-training of solar data using Zarr.

    This dataset loads raw solar data (either 1D time-series or 2D images)
    from Zarr and returns them as (input, target) pairs for reconstruction training.

    It expects:
    - A Zarr store containing the data.
    - An index file (CSV) containing timestamps for the split.
    """

    def __init__(
        self,
        zarr_path: str,
        index_path: str,
        channels: list[str],
        scalers: DictConfig | None = None,
        data_type: str = "1d",
        phase: str = "train",
        transform=None,
    ):
        """
        Args:
            zarr_path (str): Path to the Zarr store.
            index_path (str): Path to the index CSV file (containing timestamps).
            channels (list[str]): List of channels to load from Zarr.
            scalers (DictConfig, optional): Normalization statistics.
            data_type (str): Type of data, either '1d' or '2d'.
            phase (str): Phase of the dataset (train, val, test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.zarr_path = zarr_path
        self.index_path = index_path
        self.channels = channels
        self.scalers = scalers
        self.data_type = data_type
        self.phase = phase
        self.transform = transform

        # Load index
        self.logger = lgr_logger
        self.logger.info(f"Loading index from {index_path}")
        self.index = pd.read_csv(index_path)
        self.index["timestamp"] = pd.to_datetime(self.index["timestamp"]).values.astype(
            "datetime64[ns]"
        )
        self.index.set_index("timestamp", inplace=True)
        self.index.sort_index(inplace=True)

        # Open Zarr store
        self.logger.info(f"Opening Zarr store at {zarr_path}")
        # We don't load data here to avoid memory issues, we open it lazily or load per sample
        self.zarr_data = xr.open_zarr(zarr_path, consolidated=True)

        self.length = len(self.index)

    def __len__(self):
        return self.length

    def norm_log_zscore(self, data_arr, stats, eps=1e-10):
        """
        Normalize data using log10 and z-score.

        Args:
            data_arr: Numpy array or Xarray DataArray.
            stats: DictConfig with 'mean' and 'std'.

        Returns:
            Normalized data.
        """
        x = np.clip(data_arr, eps, None)  # avoid log(0)
        x_log = np.log10(x)
        return (x_log - stats.mean) / stats.std  # → ~N(0, 1)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (input, target, timestamp) where both are tensors.
        """
        timestamp = self.index.index[idx]

        # Load data for this timestamp
        # Assuming Zarr has a dimension 'timestep' matching the index
        try:
            data = self.zarr_data[self.channels].sel(timestep=timestamp)
        except KeyError:
            self.logger.error(f"Timestamp {timestamp} not found in Zarr.")
            # Return a dummy sample or raise error
            raise IndexError(f"Timestamp {timestamp} not found in Zarr.")

        # Convert to numpy
        data_np = data.to_numpy()

        # Normalize
        if self.scalers:
            # Assuming scalers has a key for each channel or a global scaler
            # Here we apply a generic normalization if scaler is provided
            # In a real scenario, you'd match channels to scalers
            for i, ch in enumerate(self.channels):
                if ch in self.scalers:
                    stats = self.scalers[ch]
                    data_np[i] = self.norm_log_zscore(data_np[i], stats)

        # Convert to tensor
        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        # Handle transform
        if self.transform:
            data_tensor = self.transform(data_tensor)

        # For self-supervised, input and target are the same
        # Return timestamp as well for predict step
        return data_tensor, data_tensor, timestamp
