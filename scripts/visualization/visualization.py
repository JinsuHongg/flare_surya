import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from loguru import logger as lgr_logger

from flare_surya.datamodule import SolarPretrainDataModule
from flare_surya.models import PretrainSolarModel


def build_model(cfg):
    """Build the PretrainSolarModel."""
    model = PretrainSolarModel(
        in_channels=cfg.model.in_channels,
        seq_len=cfg.model.seq_len,
        embed_dim=cfg.model.embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        decoder_depth=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        data_type=cfg.model.data_type,
        save_embeddings_path=cfg.model.get("save_embeddings_path", None),
        mask_ratio=cfg.model.get("mask_ratio", 0.5),
        image_size=cfg.model.get("image_size", 224),
        patch_size=cfg.model.get("patch_size", 16),
        optimizer_dict=cfg.optimizer,
        loss_dict=cfg.loss,
    )
    return model


def load_checkpoint(cfg, model):
    """Load pretrained checkpoint."""
    checkpoint_dir = cfg.etc.get("ckpt_dir", None)
    checkpoint_file = cfg.etc.get("ckpt_file", None)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    if checkpoint_path and torch.cuda.is_available():
        lgr_logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
    elif checkpoint_path:
        lgr_logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
    return model


@hydra.main(
    version_base=None,
    config_path="../../configs/pretrain/",
    config_name="solar_pretrain",
)
def visualize(cfg: OmegaConf):
    datamodule = SolarPretrainDataModule(
        zarr_path=cfg.data.zarr_path,
        train_index_path=cfg.data.train_index_path,
        val_index_path=cfg.data.val_index_path,
        test_index_path=cfg.data.test_index_path,
        channels=list(cfg.data.channels),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_type=cfg.data.data_type,
        scalers=cfg.data.get("scalers", None),
    )

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    model = build_model(cfg=cfg)
    model = load_checkpoint(cfg, model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch = next(iter(test_loader))
    if len(batch) == 3:
        x, y, _ = batch
    else:
        x, y = batch

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        reconstruction = model.forward(x, use_mask=False)

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    recon_np = reconstruction.detach().cpu().numpy()

    data_type = cfg.model.data_type

    if data_type == "1d":
        visualize_1d(x_np, y_np, recon_np)
    elif data_type == "2d":
        visualize_2d(x_np, y_np, recon_np)


def visualize_1d(x, y, reconstruction, num_samples=4):
    """Visualize 1D time series data."""
    batch_size, channels, seq_len = x.shape
    num_samples = min(num_samples, batch_size)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i in range(num_samples):
        for ch in range(channels):
            ax_input = axes[i, 0] if num_samples > 1 else axes[0]
            ax_target = axes[i, 1] if num_samples > 1 else axes[1]
            ax_recon = axes[i, 2] if num_samples > 1 else axes[2]

            ax_input.plot(x[i, ch], label="Input", alpha=0.7)
            ax_target.plot(y[i, ch], label="Target", alpha=0.7)
            ax_recon.plot(reconstruction[i, ch], label="Reconstruction", alpha=0.7)

            ax_input.set_title(f"Sample {i} - Input (Channel {ch})")
            ax_target.set_title(f"Sample {i} - Target (Channel {ch})")
            ax_recon.set_title(f"Sample {i} - Reconstruction (Channel {ch})")

    plt.tight_layout()
    plt.savefig("visualization_1d.png", dpi=150)
    lgr_logger.info("Saved visualization_1d.png")
    plt.close()


def visualize_2d(x, y, reconstruction, num_samples=4):
    """Visualize 2D image data."""
    batch_size, channels, h, w = x.shape
    num_samples = min(num_samples, batch_size)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i in range(num_samples):
        for ch in range(channels):
            idx = i * 3 + ch
            ax_input = axes[idx // 3, 0] if num_samples > 1 else axes[0]
            ax_target = axes[idx // 3, 1] if num_samples > 1 else axes[1]
            ax_recon = axes[idx // 3, 2] if num_samples > 1 else axes[2]

            im_input = ax_input.imshow(x[i, ch], cmap="viridis", aspect="auto")
            im_target = ax_target.imshow(y[i, ch], cmap="viridis", aspect="auto")
            im_recon = ax_recon.imshow(
                reconstruction[i, ch], cmap="viridis", aspect="auto"
            )

            ax_input.set_title(f"Sample {i} - Input (Channel {ch})")
            ax_target.set_title(f"Sample {i} - Target (Channel {ch})")
            ax_recon.set_title(f"Sample {i} - Reconstruction (Channel {ch})")

            plt.colorbar(im_input, ax=ax_input)
            plt.colorbar(im_target, ax=ax_target)
            plt.colorbar(im_recon, ax=ax_recon)

    plt.tight_layout()
    plt.savefig("visualization_2d.png", dpi=150)
    lgr_logger.info("Saved visualization_2d.png")
    plt.close()


if __name__ == "__main__":
    visualize()
