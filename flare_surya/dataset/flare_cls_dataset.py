import os
import random
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
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
    ):

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

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

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
                "label": self.flare_index.loc[reference_timestep, "label_max"],
                "debug": debug,
            }, metadata

        return {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            # "forecast": stacked_targets,
            # "lead_time_delta": lead_time_delta_float,
            "debug": debug,
            "label": self.flare_index.loc[reference_timestep, "label_max"],
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

        exception_counter = 0
        max_exception = 100

        self.logger.info(f"Starting to retrieve index {idx}.")

        while True:
            try:
                sample = self._get_index_data(idx)
            except Exception as e:
                exception_counter += 1
                if exception_counter >= max_exception:
                    raise e

                reference_timestep = self.valid_indices[idx]
                self.logger.warning(
                    f"Failed retrieving index {idx}. Timestamp {reference_timestep}. Attempt {exception_counter}."
                )

                idx = (idx + 1) % self.__len__()
            else:
                self.logger.info(f"Returning index {idx}.")
                return sample

    # def _get_index_data(self, idx: int) -> tuple[dict, dict]:
    #     data, metadata = super()._get_index_data(idx)

    #     reference_timestamp = self.valid_indices[idx]
    #     data["label"] = self.flare_index.loc[reference_timestamp, "label_max"]

    #     return data, metadata

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
        flare_index_path: str = None,
    ):
        self.label_type = label_type
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

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def _get_index_data(self, idx: int) -> tuple[dict, dict]:
        data, metadata = super()._get_index_data(idx)

        reference_timestamp = self.valid_indices[idx]
        data["label"] = self.flare_index.loc[reference_timestamp, "label_max"]

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
        zarr_path: str = None,
    ):

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
            zarr_path=zarr_path,
        )
        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self) -> list:
        valid_indices = super().filter_valid_indices()

        valid_indices = [t for t in valid_indices if t in self.flare_index.index]

        return valid_indices

    def _get_index_data(self, idx: int) -> tuple[dict, dict]:
        data, metadata = super()._get_index_data(idx)

        reference_timestamp = self.valid_indices[idx]
        data["label"] = self.flare_index.loc[reference_timestamp, "label_max"]

        return data, metadata

    def __len__(self):
        return self.adjusted_length
