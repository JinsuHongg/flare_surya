from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger


def build_wandb(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    wandb_logger = WandbLogger(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        save_dir=cfg["wandb"]["save_dir"],
        offline=cfg["wandb"]["offline"],
        log_model=cfg["wandb"]["log_model"],
        save_code=cfg["wandb"]["save_code"],
        notes=cfg["wandb"]["notes"],
        tags=cfg["wandb"]["tag"],
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        config=cfg_dict,
        id=cfg.wandb.id,
        resume=cfg.wandb.resume,
    )

    if "backbone" in cfg:
        wandb_logger.log_hyperparams(
            {
                "lr": cfg["optimizer"]["lr"],
                "batch_size": cfg["data"]["batch_size"],
                "embed_dim": cfg["backbone"]["embed_dim"],
                "depth": cfg["backbone"]["depth"],
                "num_heads": cfg["backbone"]["num_heads"],
                "mlp_ratio": cfg["backbone"]["mlp_ratio"],
                "drop_rate": cfg["backbone"]["drop_rate"],
                "window_size": cfg["backbone"]["window_size"],
                "dp_rank": cfg["backbone"]["dp_rank"],
                "learned_flow": cfg["backbone"]["learned_flow"],
                "finetune": cfg["backbone"]["finetune"],
                "head_type": cfg["head"]["type"],
                "pooling_type": cfg["backbone"]["pooling_type"],
                "hidden_channels": cfg["head"]["hyper_parameters"]["hidden_channels"],
                "dropout_head": cfg["head"]["hyper_parameters"]["dropout"],
            }
        )
    elif "model" in cfg:
        wandb_logger.log_hyperparams(
            {
                "lr": cfg["optimizer"]["lr"],
                "batch_size": cfg["data"]["batch_size"],
                "embed_dim": cfg["model"]["embed_dim"],
                "encoder_depth": cfg["model"]["encoder_depth"],
                "decoder_depth": cfg["model"]["decoder_depth"],
                "num_heads": cfg["model"]["num_heads"],
                "mask_ratio": cfg["model"]["mask_ratio"],
                "data_type": cfg["model"]["data_type"],
            }
        )

    return wandb_logger
