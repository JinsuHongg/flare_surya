import time
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from flare_surya.task.finetuning import FlareDataModule  # Adjust import if needed


def benchmark_dataloader(cfg, num_batches=100):
    # 1. Setup DataModule
    dm = FlareDataModule(cfg)
    dm.setup("fit")

    # 2. Get the loader (Train or Val)
    loader = dm.train_dataloader()

    print(f"Benchmarking DataLoader...")
    print(f"Batch Size: {cfg.data.batch_size}")
    print(f"Num Workers: {cfg.data.num_data_workers}")
    print(f"Total Batches in Loader: {len(loader)}")

    # 3. Warmup (Pre-fetch first batch)
    iter_loader = iter(loader)
    _ = next(iter_loader)

    # 4. Timing Loop
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

    # 5. Results
    total_images = count * cfg.data.batch_size
    images_per_sec = total_images / total_time

    print(f"\nResults:")
    print(f"Processed {count} batches in {total_time:.2f} seconds.")
    print(f"Throughput: {images_per_sec:.2f} samples/second")

    return images_per_sec


if __name__ == "__main__":
    # Load your config
    cfg = OmegaConf.load("../configs/first_experiement_model_comparison.yaml")

    # OVERRIDE for testing (Use your updated settings)
    cfg.data.num_data_workers = 3
    cfg.data.batch_size = 8

    # Run Benchmark
    benchmark_dataloader(cfg, num_batches=50)
