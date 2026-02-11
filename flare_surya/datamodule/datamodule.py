# import os
from loguru import logger as lgr_logger
from omegaconf import OmegaConf
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from terratorch_surya.downstream_examples.solar_flare_forecasting.dataset import (
    SolarFlareDataset,
)
from terratorch_surya.utils.data import build_scalers
from flare_surya.utils.config import load_config
from flare_surya.dataset.flare_cls_dataset import (
    SolarFlareClsDatasetZarr,
    SolarFlareClsDatasetAWS,
)


class FlareDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # load scalers
        self.cfg["data"]["scalers"] = load_config(self.cfg["data"]["scalers_path"])
        self.scalers = build_scalers(info=self.cfg["data"]["scalers"])

    def _get_dataset(self, phase, index_path, flare_index_path):
        return SolarFlareDataset(
            sdo_data_root_path=self.cfg["data"]["sdo_data_root_path"],
            index_path=index_path,
            time_delta_input_minutes=self.cfg["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=self.cfg["data"]["time_delta_target_minutes"],
            n_input_timestamps=self.cfg["backbone"]["time_embedding"]["time_dim"],
            rollout_steps=self.cfg["rollout_steps"],
            channels=[ch.strip() for ch in self.cfg["data"]["channels"]],
            drop_hmi_probability=self.cfg["drop_hmi_probability"],
            num_mask_aia_channels=self.cfg["num_mask_aia_channels"],
            use_latitude_in_learned_flow=self.cfg["use_latitude_in_learned_flow"],
            scalers=self.scalers,
            phase=phase,
            flare_index_path=flare_index_path,
            pooling=self.cfg["data"]["pooling"],
            random_vert_flip=self.cfg["data"]["random_vert_flip"],
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_ds = self._get_dataset(
                "train",
                self.cfg["data"]["train_data_path"],
                self.cfg["data"]["train_flare_data_path"],
            )
            lgr_logger.info(f"Training # samples: {len(self.train_ds)}")

        # Assign validation dataset for use in dataloader(s)
        if stage in (None, "fit", "validate"):
            self.val_ds = self._get_dataset(
                "validation",
                self.cfg["data"]["valid_data_path"],
                self.cfg["data"]["valid_flare_data_path"],
            )

            if self.cfg.data.use_leaky_validation:
                self.leaky_val_ds = self._get_dataset(
                    "validation",
                    self.cfg.data.valid_data_path,
                    self.cfg.data.leaky_valid_flare_data_path,
                )

                self.val_ds = ConcatDataset([self.val_ds, self.leaky_val_ds])

            lgr_logger.info(f"Validation # samples: {len(self.val_ds)}")

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test_ds = self._get_dataset(
                "test",
                self.cfg["data"]["test_data_path"],
                self.cfg["data"]["test_flare_data_path"],
            )
            lgr_logger.info(f"Test # samples: {len(self.test_ds)}")

        if stage in (None, "predict"):
            self.pred_ds = self._get_dataset(
                "test",
                self.cfg["data"]["test_data_path"],
                self.cfg["data"]["test_flare_data_path"],
            )
            lgr_logger.info(f"Predict # samples: {len(self.pred_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=True,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
        )


class FlareDataModuleAWS(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # load scalers
        self.cfg.data.scalers = load_config(self.cfg.data.scalers_path)
        self.scalers = build_scalers(info=self.cfg.data.scalers)

    def _get_dataset(self, phase, index_path, flare_index_path):
        return SolarFlareClsDatasetAWS(
            index_path=index_path,
            time_delta_input_minutes=self.cfg.data.time_delta_input_minutes,
            time_delta_target_minutes=self.cfg.data.time_delta_target_minutes,
            n_input_timestamps=self.cfg.backbone.time_embedding.time_dim,
            rollout_steps=self.cfg.rollout_steps,
            channels=[ch.strip() for ch in self.cfg.data.channels],
            drop_hmi_probability=self.cfg.drop_hmi_probability,
            num_mask_aia_channels=self.cfg.num_mask_aia_channels,
            use_latitude_in_learned_flow=self.cfg.use_latitude_in_learned_flow,
            scalers=self.scalers,
            phase=phase,
            flare_index_path=flare_index_path,
            pooling=self.cfg.data.pooling,
            random_vert_flip=self.cfg.data.random_vert_flip,
            s3_use_simplecache=self.cfg.data.use_simplecache,
            s3_cache_dir=self.cfg.data.s3_cache_dir,
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_ds = self._get_dataset(
                "train",
                self.cfg.data.train_data_path,
                self.cfg.data.train_flare_data_path,
            )
            lgr_logger.info(f"Training # samples: {len(self.train_ds)}")

        # Assign validation dataset for use in dataloader(s)
        if stage in (None, "fit", "validate"):
            self.val_ds = self._get_dataset(
                "validation",
                self.cfg.data.valid_data_path,
                self.cfg.data.valid_flare_data_path,
            )

            if self.cfg.data.use_leaky_validation:
                self.leaky_val_ds = self._get_dataset(
                    "validation",
                    self.cfg.data.valid_data_path,
                    self.cfg.data.leaky_valid_flare_data_path,
                )

                self.val_ds = ConcatDataset([self.val_ds, self.leaky_val_ds])

            lgr_logger.info(f"Validation # samples: {len(self.val_ds)}")

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test_ds = self._get_dataset(
                "test",
                self.cfg.data.test_data_path,
                self.cfg.data.test_flare_data_path,
            )
            lgr_logger.info(f"Test # samples: {len(self.test_ds)}")

        if stage in (None, "predict"):
            self.pred_ds = self._get_dataset(
                "test",
                self.cfg.data.test_data_path,
                self.cfg.data.test_flare_data_path,
            )
            lgr_logger.info(f"Predict # samples: {len(self.pred_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.cfg.data.num_data_workers,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            pin_memory=self.cfg.data.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.cfg.data.num_data_workers,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.cfg.data.num_data_workers,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            num_workers=self.cfg.data.num_data_workers,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            pin_memory=self.cfg.data.pin_memory,
        )


class FlareDataModuleZarr(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # load scalers
        self.cfg["data"]["scalers"] = load_config(self.cfg["data"]["scalers_path"])
        self.scalers = build_scalers(info=self.cfg["data"]["scalers"])

    def _get_dataset(self, phase, index_path, flare_index_path):
        return SolarFlareClsDatasetZarr(
            index_path=index_path,
            time_delta_input_minutes=self.cfg["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=self.cfg["data"]["time_delta_target_minutes"],
            n_input_timestamps=self.cfg["backbone"]["time_embedding"]["time_dim"],
            rollout_steps=self.cfg["rollout_steps"],
            channels=[ch.strip() for ch in self.cfg["data"]["channels"]],
            drop_hmi_probability=self.cfg["drop_hmi_probability"],
            num_mask_aia_channels=self.cfg["num_mask_aia_channels"],
            use_latitude_in_learned_flow=self.cfg["use_latitude_in_learned_flow"],
            scalers=self.scalers,
            phase=phase,
            flare_index_path=flare_index_path,
            pooling=self.cfg["data"]["pooling"],
            random_vert_flip=self.cfg["data"]["random_vert_flip"],
            zarr_path=self.cfg.data.zarr_path,
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_ds = self._get_dataset(
                "train",
                self.cfg["data"]["train_data_path"],
                self.cfg["data"]["train_flare_data_path"],
            )
            lgr_logger.info(f"Training # samples: {len(self.train_ds)}")

        # Assign validation dataset for use in dataloader(s)
        if stage in (None, "fit", "validate"):
            self.val_ds = self._get_dataset(
                "validation",
                self.cfg["data"]["valid_data_path"],
                self.cfg["data"]["valid_flare_data_path"],
            )

            if self.cfg.data.use_leaky_validation:
                self.leaky_val_ds = self._get_dataset(
                    "validation",
                    self.cfg.data.valid_data_path,
                    self.cfg.data.leaky_valid_flare_data_path,
                )

                self.val_ds = ConcatDataset([self.val_ds, self.leaky_val_ds])

            lgr_logger.info(f"Validation # samples: {len(self.val_ds)}")

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test_ds = self._get_dataset(
                "test",
                self.cfg["data"]["test_data_path"],
                self.cfg["data"]["test_flare_data_path"],
            )
            lgr_logger.info(f"Test # samples: {len(self.test_ds)}")

        if stage in (None, "predict"):
            self.pred_ds = self._get_dataset(
                "test",
                self.cfg["data"]["test_data_path"],
                self.cfg["data"]["test_flare_data_path"],
            )
            lgr_logger.info(f"Predict # samples: {len(self.pred_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=True,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            num_workers=self.cfg["data"]["num_data_workers"],
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            prefetch_factor=self.cfg["data"]["prefetch_factor"],
            pin_memory=self.cfg["data"]["pin_memory"],
            persistent_workers=self.cfg["data"]["persistent_workers"],
        )


if __name__ == "__main__":
    cfg = OmegaConf.load("../configs/experiment_with_zarr.yaml")
    datamodule = FlareDataModuleZarr(cfg=cfg)
    datamodule.setup("fit")
    print("Done")
