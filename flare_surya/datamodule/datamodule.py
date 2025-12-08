
import lightning as L
from torch.utils.data import DataLoader
from terratorch_surya.downstream_examples.solar_flare_forecasting.dataset import SolarFlareDataset
from terratorch_surya.utils.data import build_scalers
from flare_surya.utils.config import load_config


class FlareDataModule(L.LightningDataModule):
    def __init__(
            self,
            config_path
            ):
        super().__init__()

        # load config
        self.config = load_config(config_path)
        
        # load scalers
        self.config["data"]["scalers"] = load_config(self.config["data"]["scalers_path"])
        self.scalers = build_scalers(info=self.config["data"]["scalers"])

    def _get_dataset(self, phase, index_path, flare_index_path):
        return SolarFlareDataset(
            sdo_data_root_path=self.config["data"]["sdo_data_root_path"],
            index_path=index_path,
            time_delta_input_minutes=self.config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=self.config["data"]["time_delta_target_minutes"],
            n_input_timestamps=self.config["backbone"]["time_embedding"]["time_dim"],
            rollout_steps=self.config["rollout_steps"],
            channels = [ch.strip() for ch in self.config["data"]["channels"]],
            drop_hmi_probability=self.config["drop_hmi_probability"],
            num_mask_aia_channels=self.config["num_mask_aia_channels"],
            use_latitude_in_learned_flow=self.config["use_latitude_in_learned_flow"],
            scalers=self.scalers,
            phase=phase,
            flare_index_path=flare_index_path,
            pooling=self.config["data"]["pooling"],
            random_vert_flip=self.config["data"]["random_vert_flip"],
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_ds = self._get_dataset(
                "train",
                self.config["data"]["train_data_path"],
                self.config["data"]["train_flare_data_path"]
                )

        # Assign validation dataset for use in dataloader(s)
        if stage in (None, "fit"):
            self.val_ds = self._get_dataset(
                "validation",
                self.config["data"]["valid_data_path"],
                self.config["data"]["valid_flare_data_path"]
                )
        
        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test_ds = self._get_dataset(
                "test",
                self.config["data"]["test_data_path"],
                self.config["data"]["test_flare_data_path"]
                )

        if stage in (None, "predict"):
            self.pred_ds = self._get_dataset(
                "test",
                self.config["data"]["test_data_path"],
                self.config["data"]["test_flare_data_path"]
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.config["data"]["num_data_workers"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            pin_memory=self.config["data"]["pin_memory"]
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.config["data"]["num_data_workers"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["data"]["pin_memory"]
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.config["data"]["num_data_workers"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["data"]["pin_memory"]
            )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds,
            num_workers=self.config["data"]["num_data_workers"],
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["data"]["pin_memory"]
            )