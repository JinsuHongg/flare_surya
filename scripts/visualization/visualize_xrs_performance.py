import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger as lgr_logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from flare_surya.datamodule import SuryaFluxDataModule
from flare_surya.models import SuryaMultiModal


def build_model(cfg: DictConfig) -> SuryaMultiModal:
    """Build the SuryaMultiModal model."""
    model_hyperparameter = {
        "img_size": cfg.backbone.img_size,
        "patch_size": cfg.backbone.patch_size,
        "in_chans": len(cfg.data.channels),
        "embed_dim": cfg.backbone.embed_dim,
        "time_embedding": cfg.backbone.time_embedding,
        "depth": cfg.backbone.depth,
        "num_heads": cfg.backbone.num_heads,
        "mlp_ratio": cfg.backbone.mlp_ratio,
        "drop_rate": cfg.backbone.drop_rate,
        "dtype": torch.bfloat16,
        "window_size": cfg.backbone.window_size,
        "dp_rank": cfg.backbone.dp_rank,
        "learned_flow": cfg.backbone.learned_flow,
        "use_latitude_in_learned_flow": cfg.use_latitude_in_learned_flow,
        "init_weights": cfg.backbone.init_weights,
        "checkpoint_layers": cfg.backbone.checkpoint_layers,
        "n_spectral_blocks": cfg.backbone.n_spectral_blocks,
        "rpe": cfg.backbone.rpe,
        "ensemble": cfg.backbone.ensemble,
        "finetune": cfg.backbone.finetune,
        "nglo": cfg.backbone.nglo,
        "path_weights": cfg.backbone.path_weights,
        "pooling_type": cfg.backbone.pooling_type,
        "head_type": cfg.head.type,
        "head_layer_dict": cfg.head.hyper_parameters,
        "freeze_backbone": cfg.backbone.freeze_backbone,
        "lora_dict": cfg.lora,
        "optimizer_dict": cfg.optimizer,
        "loss_dict": cfg.loss,
        "threshold": cfg.head.threshold,
        "save_test_results_path": cfg.etc.save_test_results_path,
        "in_channels": cfg.secondary.in_channels,
        "seq_len": cfg.secondary.seq_len,
        "secondary_embed_dim": cfg.secondary.embed_dim,
        "secondary_depth": cfg.secondary.depth,
        "secondary_num_heads": cfg.secondary.num_heads,
        "fusion_type": cfg.fusion.type,
        "fuse_embed_dim": cfg.fusion.fuse_embed_dim,
        "secondary_pooling_type": cfg.secondary.pooling_type,
    }
    model = SuryaMultiModal(**model_hyperparameter)
    return model


def load_checkpoint(cfg: DictConfig, model: SuryaMultiModal) -> SuryaMultiModal:
    """Load model from checkpoint."""
    ckpt_dir = cfg.etc.ckpt_dir
    ckpt_file = cfg.etc.ckpt_file
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    if ckpt_file and os.path.exists(ckpt_path):
        lgr_logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        lgr_logger.warning(f"Checkpoint not found: {ckpt_path}")
    return model


def visualize_xrs_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
    num_samples: int = 4,
) -> None:
    """Visualize XRS time series: target, prediction, and delta."""
    batch_size, seq_len = targets.shape
    num_samples = min(num_samples, batch_size)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    time_axis = np.arange(seq_len)

    for i in range(num_samples):
        target = targets[i]
        pred = predictions[i]
        delta = target - pred

        ax_target = axes[i, 0]
        ax_pred = axes[i, 1]
        ax_delta = axes[i, 2]

        ax_target.plot(time_axis, target, label="Target", color="blue", alpha=0.8)
        ax_pred.plot(time_axis, pred, label="Prediction", color="green", alpha=0.8)
        ax_delta.plot(time_axis, delta, label="Delta (Target - Pred)", color="red", alpha=0.8)
        ax_delta.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

        ax_target.set_title(f"Sample {i} - Target")
        ax_pred.set_title(f"Sample {i} - Prediction")
        ax_delta.set_title(f"Sample {i} - Delta")

        ax_target.set_xlabel("Time step")
        ax_pred.set_xlabel("Time step")
        ax_delta.set_xlabel("Time step")

        ax_target.legend(loc="upper right", fontsize=8)
        ax_pred.legend(loc="upper right", fontsize=8)
        ax_delta.legend(loc="upper right", fontsize=8)

        ax_target.grid(True, alpha=0.3)
        ax_pred.grid(True, alpha=0.3)
        ax_delta.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "xrs_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    lgr_logger.info(f"Saved visualization to {output_path}")
    plt.close()


def visualize_xrs_heatmap(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
    num_samples: int = 8,
) -> None:
    """Visualize XRS data as heatmaps: target, prediction, delta side by side."""
    batch_size, seq_len = targets.shape
    num_samples = min(num_samples, batch_size)

    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 10))

    vmin_target = np.nanmin(targets[:num_samples])
    vmax_target = np.nanmax(targets[:num_samples])

    for i in range(num_samples):
        target = targets[i]
        pred = predictions[i]
        delta = target - pred

        im0 = axes[0, i].imshow(
            target.reshape(1, -1),
            aspect="auto",
            cmap="viridis",
            vmin=vmin_target,
            vmax=vmax_target,
        )
        axes[0, i].set_title(f"Sample {i} - Target")
        axes[0, i].set_xlabel("Time step")
        plt.colorbar(im0, ax=axes[0, i], orientation="horizontal", pad=0.2)

        im1 = axes[1, i].imshow(
            pred.reshape(1, -1),
            aspect="auto",
            cmap="viridis",
            vmin=vmin_target,
            vmax=vmax_target,
        )
        axes[1, i].set_title(f"Sample {i} - Prediction")
        axes[1, i].set_xlabel("Time step")
        plt.colorbar(im1, ax=axes[1, i], orientation="horizontal", pad=0.2)

        vmax_delta = np.nanmax(np.abs(delta))
        im2 = axes[2, i].imshow(
            delta.reshape(1, -1),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax_delta,
            vmax=vmax_delta,
        )
        axes[2, i].set_title(f"Sample {i} - Delta")
        axes[2, i].set_xlabel("Time step")
        plt.colorbar(im2, ax=axes[2, i], orientation="horizontal", pad=0.2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "xrs_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    lgr_logger.info(f"Saved heatmap visualization to {output_path}")
    plt.close()


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute simple metrics for model performance."""
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    rmse = np.sqrt(mse)

    target_std = np.std(targets)
    if target_std > 0:
        nrmse = rmse / target_std
    else:
        nrmse = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
    }


@hydra.main(
    version_base=None,
    config_path="../configs/nas/",
    config_name="exp_surya",
)
def visualize(cfg: OmegaConf) -> None:
    """Main visualization function."""
    output_dir = cfg.etc.get("output_dir", "visualization_output")
    os.makedirs(output_dir, exist_ok=True)

    lgr_logger.info("Setting up datamodule...")
    datamodule = SuryaFluxDataModule(cfg=cfg)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    lgr_logger.info("Building model...")
    model = build_model(cfg=cfg)
    model = load_checkpoint(cfg, model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lgr_logger.info(f"Using device: {device}")
    model = model.to(device)

    lgr_logger.info("Running inference on test set...")
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= cfg.etc.get("max_batches", 5):
                break

            if len(batch) >= 5:
                _, _, img, xrs_seq, target = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            xrs_seq = xrs_seq.to(device)
            target = target.to(device)

            if hasattr(model, "forward"):
                output = model.forward(img.to(device), xrs_seq)
            else:
                output = model(img.to(device), xrs_seq)

            if isinstance(output, torch.Tensor):
                predictions = output
            elif isinstance(output, dict):
                predictions = output.get("logits", output.get("pred", None))
                if predictions is None:
                    predictions = list(output.values())[0]
            else:
                predictions = output[0] if isinstance(output, (list, tuple)) else output

            if predictions.dim() == 2 and predictions.shape[1] > 1:
                predictions = predictions[:, 1]

            if target.dim() > 1:
                target = target.squeeze(-1) if target.shape[-1] == 1 else target.squeeze()

            all_targets.append(target.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    targets_np = np.concatenate(all_targets, axis=0)
    predictions_np = np.concatenate(all_predictions, axis=0)

    lgr_logger.info(f"Targets shape: {targets_np.shape}")
    lgr_logger.info(f"Predictions shape: {predictions_np.shape}")

    metrics = compute_metrics(targets_np, predictions_np)
    lgr_logger.info("Metrics:")
    for k, v in metrics.items():
        lgr_logger.info(f"  {k}: {v:.6f}")

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    lgr_logger.info(f"Saved metrics to {metrics_file}")

    visualize_xrs_predictions(
        targets=targets_np,
        predictions=predictions_np,
        output_dir=output_dir,
        num_samples=cfg.etc.get("num_viz_samples", 4),
    )

    visualize_xrs_heatmap(
        targets=targets_np,
        predictions=predictions_np,
        output_dir=output_dir,
        num_samples=cfg.etc.get("num_viz_samples", 8),
    )

    lgr_logger.info(f"Visualization complete! Output saved to {output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", "Importing from timm.models.layers is deprecated.*", FutureWarning
    )
    visualize()