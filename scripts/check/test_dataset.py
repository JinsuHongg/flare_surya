import sys
import os
import torch.multiprocessing as mp
from omegaconf import OmegaConf

# Add the project root to the Python path
# This allows us to import from flare_surya
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from flare_surya.dataset.flare_cls_dataset import SolarFlareClsDataset


from terratorch_surya.utils.data import build_scalers


def test_dataset_loading(config_path):
    """
    Loads a dataset configuration, instantiates the SolarFlareClsDataset,
    and attempts to retrieve a single sample.
    """
    print(f"Loading configuration from: {config_path}")
    cfg = OmegaConf.load(config_path)

    print("Loading scalers...")
    # Construct absolute path for the scalers file
    scalers_path = os.path.join(project_root, cfg.data.scalers_path.lstrip("./"))
    scaler_cfg = OmegaConf.load(scalers_path)
    scalers = build_scalers(info=OmegaConf.to_container(scaler_cfg, resolve=True))
    print("Scalers loaded successfully.")

    print("Initializing SolarFlareClsDataset for the 'train' phase...")

    # Manually construct the arguments for the dataset from the config
    dataset_args = {
        "sdo_data_root_path": cfg.data.sdo_data_root_path,
        "index_path": cfg.data.train_data_path,
        "flare_index_path": cfg.data.train_flare_data_path,
        "time_delta_input_minutes": cfg.data.time_delta_input_minutes,
        "time_delta_target_minutes": cfg.data.time_delta_target_minutes,
        "n_input_timestamps": cfg.data.n_input_timestamps,
        "rollout_steps": cfg.rollout_steps,
        "scalers": scalers,
        "num_mask_aia_channels": cfg.num_mask_aia_channels,
        "drop_hmi_probability": cfg.drop_hmi_probability,
        "use_latitude_in_learned_flow": cfg.use_latitude_in_learned_flow,
        "channels": list(cfg.data.channels),
        "phase": "train",
        "pooling": cfg.data.pooling,
        "random_vert_flip": cfg.data.random_vert_flip,
    }

    try:
        dataset = SolarFlareClsDataset(**dataset_args)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print(
            "\nPlease ensure that the paths in the config file are correct and data is available."
        )
        return

    print(f"Dataset initialized successfully. Number of samples: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty. Cannot retrieve a sample.")
        return

    print("\nAttempting to retrieve the first sample (index 0)...")

    try:
        sample = dataset[0]
        print("Successfully retrieved sample!")
        print("\nSample keys and data shapes:")
        for key, value in sample[0].items():
            if hasattr(value, "shape"):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: {value}")

    except Exception as e:
        import traceback

        print(f"\nAn error occurred while retrieving a sample: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
        print(
            "\nThis might be due to a missing/corrupt data file for this specific index, or an issue within the data loading code."
        )


if __name__ == "__main__":
    # Set the start method to 'spawn' for cleaner, safer worker processes.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Can only be set once

    # The script assumes it is in ./scripts and the config is in ./configs
    config_file = os.path.join(project_root, "configs", "nas", "exp_surya.yaml")
    test_dataset_loading(config_file)
    print("\nTest script finished.")
    print(
        "NOTE: This script should be run from the root of the 'flare_surya' project directory."
    )
