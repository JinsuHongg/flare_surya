import time
import torch
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from flare_surya.task.finetuning import FlareDataModule  # Adjust import if needed
from terratorch_surya.utils.data import build_scalers
from terratorch_surya.datasets.helio import HelioNetCDFDataset
from flare_surya.dataset.helio_zarr import HelioNetCDFDatasetZarr


def benchmark_dataloader(cfg, num_batches=10):
    # Setup DataModule
    # dm = FlareDataModule(cfg)

    cfg["data"]["scalers"] = yaml.safe_load(open(cfg["data"]["scalers_path"], "r"))
    scalers = build_scalers(info=cfg["data"]["scalers"])

    # dataset_surya = HelioNetCDFDataset(
    #     sdo_data_root_path=cfg["data"]["sdo_data_root_path"],
    #     index_path=cfg["data"]["train_data_path"],
    #     time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
    #     time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
    #     n_input_timestamps=cfg["backbone"]["time_embedding"]["time_dim"],
    #     rollout_steps=cfg["rollout_steps"],
    #     channels=[ch.strip() for ch in cfg["data"]["channels"]],
    #     drop_hmi_probability=cfg["drop_hmi_probability"],
    #     num_mask_aia_channels=cfg["num_mask_aia_channels"],
    #     use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"],
    #     pooling=cfg["data"]["pooling"],
    #     random_vert_flip=cfg["data"]["random_vert_flip"],
    #     phase="train",
    #     scalers=scalers,
    # )

    dataset_surya = HelioNetCDFDatasetZarr(
        time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
        n_input_timestamps=cfg["backbone"]["time_embedding"]["time_dim"],
        rollout_steps=cfg["rollout_steps"],
        channels=[ch.strip() for ch in cfg["data"]["channels"]],
        drop_hmi_probability=cfg["drop_hmi_probability"],
        num_mask_aia_channels=cfg["num_mask_aia_channels"],
        use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"],
        pooling=cfg["data"]["pooling"],
        random_vert_flip=cfg["data"]["random_vert_flip"],
        phase="train",
        scalers=scalers,
        zarr_path=cfg.data.zarr_path,
    )

    loader = DataLoader(
        dataset_surya,
        num_workers=cfg["data"]["num_data_workers"],
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Get the loader (Train or Val)
    # loader = dm.train_dataloader()

    print(f"Benchmarking DataLoader...")
    print(f"Batch Size: {cfg.data.batch_size}")
    print(f"Num Workers: {cfg.data.num_data_workers}")
    print(f"Total Batches in Loader: {len(loader)}")

    # Warmup (Pre-fetch first batch)
    iter_loader = iter(loader)
    _ = next(iter_loader)

    # Timing Loop
    start_time = time.time()
    count = 0

    # Use tqdm to visualize speed
    for i, batch in tqdm(enumerate(loader), total=min(len(loader), num_batches)):
        # Optional: Move to GPU to include transfer time cost
        # batch = [b.cuda(non_blocking=True) for b in batch]

        count += 1
        if count >= num_batches:
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Results
    total_images = count * cfg.data.batch_size
    images_per_sec = total_images / total_time

    print(f"\nResults:")
    print(f"Processed {count} batches in {total_time:.2f} seconds.")
    print(f"Throughput: {images_per_sec:.2f} samples/second")

    return images_per_sec


if __name__ == "__main__":
    # Load your config
    # cfg = OmegaConf.load("../configs/first_experiment_model_comparison.yaml")
    cfg = OmegaConf.load("../configs/experiment_with_zarr.yaml")

    # OVERRIDE for testing (Use your updated settings)
    cfg.data.num_data_workers = 6
    cfg.data.batch_size = 1

    # Run Benchmark
    benchmark_dataloader(cfg, num_batches=30)
