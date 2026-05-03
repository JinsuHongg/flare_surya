from typing import Any
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger


def build_wandb(cfg: DictConfig) -> WandbLogger:
    """Builds a WandbLogger object based on the provided configuration.

    Args:
        cfg: Hydra/OmegaConf configuration object.

    Returns:
        WandbLogger: Configured WandbLogger instance.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

    # Automatically append lr and weight decay to name and id for unique tracking
    lr = cfg.optimizer.get("lr", "unknown")
    wd = cfg.optimizer.get("weight_decay", "unknown")
    wandb_name = f"{cfg.wandb.name}_lr{lr}_wd{wd}"
    wandb_id = f"{cfg.wandb.id}_lr{lr}_wd{wd}"

    wandb_logger = WandbLogger(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        save_dir=cfg["wandb"]["save_dir"],
        offline=cfg["wandb"]["offline"],
        log_model=cfg["wandb"]["log_model"],
        save_code=cfg["wandb"]["save_code"],
        notes=cfg["wandb"]["notes"],
        tags=cfg["wandb"]["tag"],
        name=wandb_name,
        group=cfg.wandb.group,
        config=cfg_dict,
        id=wandb_id,
        resume=cfg.wandb.resume,
    )

    hparams: dict[str, Any] = {
        "lr": lr,
        "weight_decay": wd,
        "batch_size": cfg.data.get("batch_size"),
    }

    if "backbone" in cfg:
        # Check if it's the Baseline structure (has model_name) or Surya structure (has embed_dim)
        if "model_name" in cfg.backbone:
            # Baseline-specific hyperparameters
            hparams.update(
                {
                    "model_name": cfg.backbone.get("model_name"),
                    "in_channels": cfg.backbone.get("in_channels"),
                    "time_steps": cfg.backbone.get("time_steps"),
                    "p_drop": cfg.backbone.get("p_drop"),
                }
            )
        else:
            # Surya-specific hyperparameters
            hparams.update(
                {
                    "embed_dim": cfg.backbone.get("embed_dim"),
                    "depth": cfg.backbone.get("depth"),
                    "num_heads": cfg.backbone.get("num_heads"),
                    "mlp_ratio": cfg.backbone.get("mlp_ratio"),
                    "drop_rate": cfg.backbone.get("drop_rate"),
                    "window_size": cfg.backbone.get("window_size"),
                    "dp_rank": cfg.backbone.get("dp_rank"),
                    "learned_flow": cfg.backbone.get("learned_flow"),
                    "finetune": cfg.backbone.get("finetune"),
                    "pooling_type": cfg.backbone.get("pooling_type"),
                }
            )
            if "head" in cfg:
                hparams.update(
                    {
                        "head_type": cfg.head.get("type"),
                        "hidden_channels": cfg.head.get("hyper_parameters", {}).get(
                            "hidden_channels"
                        ),
                        "dropout_head": cfg.head.get("hyper_parameters", {}).get(
                            "dropout"
                        ),
                    }
                )
    elif "model" in cfg:
        hparams.update(
            {
                "embed_dim": cfg.model.get("embed_dim"),
                "encoder_depth": cfg.model.get("encoder_depth"),
                "decoder_depth": cfg.model.get("decoder_depth"),
                "num_heads": cfg.model.get("num_heads"),
                "mask_ratio": cfg.model.get("mask_ratio"),
                "data_type": cfg.model.get("data_type"),
            }
        )

    wandb_logger.log_hyperparams(hparams)

    return wandb_logger
