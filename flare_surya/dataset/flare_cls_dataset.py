import os
import random
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger as lgr_logger
from omegaconf import OmegaConf, DictConfig
from typing import Optional
from numpy.random import default_rng
from terratorch_surya.datasets.helio import HelioNetCDFDataset

from flare_surya.dataset.helio_aws import HelioNetCDFDatasetAWS
from flare_surya.dataset.helio_zarr import HelioNetCDFDatasetZarr


class SolarFlareClsDataset(HelioNetCDFDataset):
    """
    The solar flare index data (flare_index_path) should be of the form

    timestamp,max_goes_class,cumulative_index,label_max,label_cum
    2011-01-01 00:00:00,B8.3,0.0,0,0
    2011-01-01 01:00:00,B8.3,0.0,0,0
    2011-01-01 02:00:00,B8.3,0.0,0,0
    2011-01-01 03:00:00,B8.3,0.0,0,0
    2011-01-01 04:00:00,B8.3,0.0,0,0
    2011-01-01 05:00:00,B8.3,0.0,0,0
    2011-01-01 06:00:00,B8.3,0.0,0,0
    """

    def __init__(
        self,
        sdo_data_root_path: str,
        index_path: str,
        flare_index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        label_type: str = "label_max",
        undersample_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.label_type = label_type
        self.undersample_factor = undersample_factor
        self.seed = seed
        lgr_logger.info(
            f"Dataset init: undersample_factor={undersample_factor}, seed={seed}"
        )
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(
            self.flare_index["timestamp"]
        ).values.astype("datetime64[ns]")
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        super().__init__(
            sdo_data_root_path=sdo_data_root_path,
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
        )
        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

        # Apply undersampling for training phase
        if self.phase == "train" and self.undersample_factor is not None:
            # Count labels before undersampling
            majority_count = sum(
                1 for t in self.valid_indices
                if self.flare_index.loc[t, self.label_type] == 0
            )
            minority_count = len(self.valid_indices) - majority_count
            lgr_logger.info(
                f"Before undersampling: {len(self.valid_indices)} samples "
                f"(majority={majority_count}, minority={minority_count})"
            )
            self.valid_indices = self._apply_undersampling(
                self.flare_index,
                self.valid_indices,
                self.label_type,
                self.undersample_factor,
                self.seed,
            )
            self.adjusted_length = len(self.valid_indices)
            lgr_logger.info(
                f"After undersampling: {self.adjusted_length} samples"
            )

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    @staticmethod
    def _apply_undersampling(
        flare_index: pd.DataFrame,
        valid_indices: list,
        label_type: str,
        undersample_factor: float,
        seed: int | None,
    ) -> list:
        """Apply undersampling to balance the dataset.

        Args:
            flare_index: The flare index DataFrame containing labels.
            valid_indices: List of valid indices (timestamps).
            label_type: The label column to use ('label_max' or 'label_cum').
            undersample_factor: Ratio of majority to minority samples.
                E.g., 2.0 means 2x non-flaring samples per flaring sample.
            seed: Random seed for reproducibility.

        Returns:
            New list of indices with undersampling applied (if applicable).
        """
        # Separate majority (label=0) and minority (label>0) indices
        majority_indices = []
        minority_indices = []

        for idx in valid_indices:
            label = flare_index.loc[idx, label_type]
            if label == 0:
                majority_indices.append(idx)
            else:
                minority_indices.append(idx)

        lgr_logger.info(
            f"_apply_undersampling: majority={len(majority_indices)}, "
            f"minority={len(minority_indices)}, factor={undersample_factor}"
        )

        # If there are no minority samples, return original indices
        if not minority_indices:
            return valid_indices

        # Calculate the target number of majority samples
        target_majority_count = int(len(minority_indices) * undersample_factor)
        lgr_logger.info(f"Target majority count: {target_majority_count}")

        # If there are fewer majority samples than needed, return all
        if len(majority_indices) <= target_majority_count:
            lgr_logger.info("Not undersampling: not enough majority samples")
            return valid_indices

        # Sample from majority indices using seeded generator
        rng = default_rng(seed)
        selected_majority_indices = list(
            rng.choice(majority_indices, size=target_majority_count, replace=False)
        )
        lgr_logger.info(f"Sampled {len(selected_majority_indices)} from majority")

        # Combine and sort
        return sorted(minority_indices + selected_majority_indices)

    def load_nc_data(self, filepath: str, channels: list[str]) -> np.ndarray:
        """
        Args:
            filepath: String or Pathlike. Points to NetCDF file to open.
        Returns:
            Numpy array of shape (C, H, W).
        """
        self.logger.info(f"Reading file {filepath}.")

        if self.sdo_data_root_path and not os.path.isabs(filepath):
            filepath = os.path.join(self.sdo_data_root_path, filepath)

        t0 = time.perf_counter()
        with xr.open_dataset(
            filepath,
            engine="h5netcdf",
            chunks=None,
            cache=False,
        ) as ds:
            t1 = time.perf_counter()
            data = ds[channels].to_array().load().to_numpy()

        t2 = time.perf_counter()

        return (
            data,
            {
                "open_time": t1 - t0,  # How long to find/open file
                "read_time": t2 - t1,  # How long to read bytes
            },
        )

    def _get_index_data(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        time_deltas = np.array(
            sorted(
                random.sample(
                    self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1
                )
            )
            + [self.time_delta_input_minutes[-1]]
            # + self.time_delta_target_minutes
        )
        reference_timestep = self.valid_indices[idx]
        required_timesteps = reference_timestep + time_deltas

        sequence_data = []
        for timestep in required_timesteps:
            data, debug = self.load_nc_data(
                self.index.loc[timestep, "path"], self.channels
            )
            sequence_data.append(self.transform_data(data))

        # Split sequence_data into inputs and target
        inputs = sequence_data
        # targets = sequence_data[-self.rollout_steps - 1 :]

        stacked_inputs = np.stack(inputs, axis=1)
        # stacked_targets = np.stack(targets, axis=1)

        timestamps_input = required_timesteps
        # timestamps_targets = required_timesteps[-self.rollout_steps - 1 :]

        if self.num_mask_aia_channels > 0 or self.drop_hmi_probability:
            stacked_inputs = self.masker(stacked_inputs)

        if not np.isfinite(stacked_inputs).all():
            self.logger.warning(
                f"NaN or Inf found in sample at index {idx}, reference_timestep: {reference_timestep}"
            )

        time_delta_input_float = (time_deltas[-1] - time_deltas[0]) / np.timedelta64(
            1, "h"
        )
        time_delta_input_float = time_delta_input_float.astype(np.float32)

        # lead_time_delta_float = (
        #     time_deltas[-self.rollout_steps - 2]
        #     - time_deltas[-self.rollout_steps - 1 :]
        # ) / np.timedelta64(1, "h")
        # lead_time_delta_float = lead_time_delta_float.astype(np.float32)

        metadata = {
            "timestamps_input": timestamps_input.astype(int),
            # "timestamps_targets": timestamps_targets.astype(int),
        }

        if self.random_vert_flip:
            if torch.bernoulli(torch.ones(()) / 2) == 1:
                stacked_inputs = torch.flip(stacked_inputs, dims=-2)
                # stacked_targets = torch.flip(stacked_inputs, dims=-2)

        if self.use_latitude_in_learned_flow:
            from sunpy.coordinates.ephemeris import get_earth

            sequence_latitude = [
                get_earth(timestep).lat.value for timestep in required_timesteps
            ]
            input_latitudes = sequence_latitude[: -self.rollout_steps - 1]
            # target_latitude = sequence_latitude[-self.rollout_steps - 1 :]

            return {
                "ts": stacked_inputs,
                "time_delta_input": time_delta_input_float,
                "input_latitudes": input_latitudes,
                # "forecast": stacked_targets,
                # "lead_time_delta": lead_time_delta_float,
                # "forecast_latitude": target_latitude,
                "label": self.flare_index.loc[reference_timestep, self.label_type],
                "debug": debug,
            }, metadata

        return {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            # "forecast": stacked_targets,
            # "lead_time_delta": lead_time_delta_float,
            "debug": debug,
            "label": self.flare_index.loc[reference_timestep, self.label_type],
        }, metadata

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """
        if self.logger is None:
            self.create_logger()
            self.logger.info(f"HelioNetCDFDataset of length {self.__len__()}.")

        self.logger.info(f"Starting to retrieve index {idx}.")
        sample = self._get_index_data(idx)
        self.logger.info(f"Returning index {idx}.")
        return sample

    def __len__(self):
        return self.adjusted_length


class SolarFlareClsDatasetAWS(HelioNetCDFDatasetAWS):
    """
    The solar flare index data (flare_index_path) should be of the form

    timestamp,max_goes_class,cumulative_index,label_max,label_cum
    2011-01-01 00:00:00,B8.3,0.0,0,0
    2011-01-01 01:00:00,B8.3,0.0,0,0
    2011-01-01 02:00:00,B8.3,0.0,0,0
    2011-01-01 03:00:00,B8.3,0.0,0,0
    2011-01-01 04:00:00,B8.3,0.0,0,0
    2011-01-01 05:00:00,B8.3,0.0,0,0
    2011-01-01 06:00:00,B8.3,0.0,0,0
    """

    def __init__(
        self,
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        s3_use_simplecache: bool = True,
        s3_cache_dir: str = "/tmp/helio_s3_cache",
        #### Put your donwnstream (DS) specific parameters below this line
        label_type: str = "label_max",
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        flare_index_path: str = "",
        undersample_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.label_type = label_type
        self.undersample_factor = undersample_factor
        self.seed = seed
        self.return_surya_stack = return_surya_stack
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(
            self.flare_index["timestamp"]
        ).values.astype("datetime64[ns]")
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        super().__init__(
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
            s3_use_simplecache=s3_use_simplecache,
            s3_cache_dir=s3_cache_dir,
        )

        if max_number_of_samples is not None:
            self.adjusted_length = min(self.adjusted_length, max_number_of_samples)

        self.valid_indices = self.filter_valid_indices()

        # Apply undersampling for training phase
        if self.phase == "train" and self.undersample_factor is not None:
            self.valid_indices = self._apply_undersampling(
                self.flare_index,
                self.valid_indices,
                self.label_type,
                self.undersample_factor,
                self.seed,
            )
            self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def _get_index_data(self, idx: int) -> tuple[dict, dict]:
        data, metadata = super()._get_index_data(idx)

        reference_timestamp = self.valid_indices[idx]
        data["label"] = self.flare_index.loc[reference_timestamp, self.label_type]

        return data, metadata


class SolarFlareClsDatasetZarr(HelioNetCDFDatasetZarr):
    """
    The solar flare index data (flare_index_path) should be of the form

    timestamp,max_goes_class,cumulative_index,label_max,label_cum
    2011-01-01 00:00:00,B8.3,0.0,0,0
    2011-01-01 01:00:00,B8.3,0.0,0,0
    2011-01-01 02:00:00,B8.3,0.0,0,0
    2011-01-01 03:00:00,B8.3,0.0,0,0
    2011-01-01 04:00:00,B8.3,0.0,0,0
    2011-01-01 05:00:00,B8.3,0.0,0,0
    2011-01-01 06:00:00,B8.3,0.0,0,0
    """

    def __init__(
        self,
        flare_index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        zarr_path: str = "",
        is_downstream: bool = False,
        label_type: str = "label_max",
        undersample_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.label_type = label_type
        self.undersample_factor = undersample_factor
        self.seed = seed
        self.flare_index = pd.read_csv(flare_index_path)
        self.flare_index["timestamp"] = pd.to_datetime(
            self.flare_index["timestamp"]
        ).values.astype("datetime64[ns]")
        self.flare_index.set_index("timestamp", inplace=True)
        self.flare_index.sort_index(inplace=True)

        super().__init__(
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
            zarr_path=zarr_path,
            is_downstream=is_downstream,
        )
        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

        # Apply undersampling for training phase
        if self.phase == "train" and self.undersample_factor is not None:
            self.valid_indices = self._apply_undersampling(
                self.flare_index,
                self.valid_indices,
                self.label_type,
                self.undersample_factor,
                self.seed,
            )
            self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def _get_index_data(self, idx: int) -> tuple[dict, dict]:
        data, metadata = super()._get_index_data(idx)

        reference_timestamp = self.valid_indices[idx]
        data["label"] = self.flare_index.loc[reference_timestamp, self.label_type]

        return data, metadata

    def __len__(self):
        return self.adjusted_length


class SolarFlareClsXRSDataset(SolarFlareClsDataset):
    """
    Add 24hour chunked Xray flux data into flare dataset.
    """

    def __init__(
        self,
        sdo_data_root_path: str,
        index_path: str,
        flare_index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probability=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        pooling: int | None = None,
        random_vert_flip: bool = False,
        xrs_data: Optional[xr.Dataset] = None,
        xrs_stat: Optional[DictConfig] = None,
        label_type: str = "label_max",
        undersample_factor: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        # Set undersample_factor BEFORE calling parent init so logging shows correct value
        self.undersample_factor = undersample_factor
        # Pass label_type and seed to parent, but NOT undersample_factor
        # since we want to apply it after XRS filtering
        super().__init__(
            sdo_data_root_path=sdo_data_root_path,
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probability=drop_hmi_probability,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
            pooling=pooling,
            random_vert_flip=random_vert_flip,
            flare_index_path=flare_index_path,
            label_type=label_type,
            seed=seed,
        )
        self.xrs_data = xrs_data
        self.xrs_stat = xrs_stat

        xrs_timesteps = pd.to_datetime(xrs_data["timestep"].values)
        new_valid_indices = [t for t in self.valid_indices if t in xrs_timesteps]
        self.valid_indices = new_valid_indices
        self.adjusted_length = len(self.valid_indices)

        # Apply undersampling for training phase after XRS filtering
        if self.phase == "train" and self.undersample_factor is not None:
            self.valid_indices = self._apply_undersampling(
                self.flare_index,
                self.valid_indices,
                self.label_type,
                self.undersample_factor,
                self.seed,
            )
            self.adjusted_length = len(self.valid_indices)

    def _get_index_data(self, idx: int) -> tuple[dict, dict]:
        data, metadata = super()._get_index_data(idx)
        reference_timestamp = self.valid_indices[idx]
        data["label"] = self.flare_index.loc[reference_timestamp, self.label_type]

        xrs = self.xrs_data["xray"].sel(timestep=reference_timestamp)
        # shape: (minute_offset, channel) after timestep selection

        xrs_soft = torch.tensor(
            self.norm_log_zscore(xrs.sel(channel="soft").values, self.xrs_stat.soft),
            dtype=torch.float32,
        )
        xrs_hard = torch.tensor(
            self.norm_log_zscore(xrs.sel(channel="hard").values, self.xrs_stat.hard),
            dtype=torch.float32,
        )
        # Stack soft and hard channels into single tensor [2, seq_len]
        data["xrs"] = torch.stack([xrs_soft, xrs_hard], dim=0)
        return data, metadata

    def norm_log_zscore(self, data_arr, stats, eps=1e-10):
        x = np.clip(data_arr, eps, None)  # avoid log(0)
        x_log = np.log10(x)
        return (x_log - stats.mean) / stats.std  # → ~N(0, 1)
