"""Test pretraining with real XRS solar data using config from configs/pretrain/solar_pretrain.yaml."""

import os
import sys

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from flare_surya.datamodule import SolarPretrainDataModule
from flare_surya.models import PretrainSolarModel


def main():
    config_path = os.path.join(project_root, "configs/pretrain/solar_pretrain.yaml")
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    print("\n=== Config ===")
    print(f"Model: {OmegaConf.to_yaml(cfg.model)}")
    print(f"Data: {OmegaConf.to_yaml(cfg.data)}")

    model = PretrainSolarModel(
        in_channels=cfg.model.in_channels,
        seq_len=cfg.model.seq_len,
        embed_dim=cfg.model.embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        decoder_depth=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        data_type=cfg.model.data_type,
        lr=cfg.model.lr,
        mask_ratio=cfg.model.get("mask_ratio", 0.5),
        image_size=cfg.model.get("image_size", 224),
        patch_size=cfg.model.get("patch_size", 16),
    )
    print(f"\nModel created: {type(model).__name__}")
    print(f"  Data type: {model.data_type}")
    print(f"  Mask ratio: {model.mask_ratio}")
    print(f"  Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Decoder params: {sum(p.numel() for p in model.decoder.parameters()):,}")

    datamodule = SolarPretrainDataModule(
        zarr_path=cfg.data.zarr_path,
        train_index_path=cfg.data.train_index_path,
        val_index_path=cfg.data.val_index_path,
        test_index_path=cfg.data.test_index_path,
        channels=list(cfg.data.channels),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_type=cfg.data.data_type,
    )
    print(f"\nDatamodule created with config:")
    print(f"  Zarr: {cfg.data.zarr_path}")
    print(f"  Train index: {cfg.data.train_index_path}")
    print(f"  Val index: {cfg.data.val_index_path}")
    print(f"  Channels: {list(cfg.data.channels)}")
    print(f"  Batch size: {cfg.data.batch_size}")

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        enable_progress_bar=False,
    )
    print(f"\n=== Running fast_dev_run (1 train + 1 val batch) ===")

    trainer.fit(model=model, datamodule=datamodule)

    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    main()
