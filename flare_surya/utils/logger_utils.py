from omegaconf import DictConfig, OmegaConf
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
        config=cfg_dict,
        id=cfg.wandb.id,
        resume=cfg.wandb.resume,
    )

    # selected hparams for WandB
    wandb_logger.log_hyperparams(
        {
            # optimizer / training
            "lr": cfg["optimizer"]["lr"],
            "batch_size": cfg["data"]["batch_size"],
            # model backbone
            "embed_dim": cfg["backbone"]["embed_dim"],
            "depth": cfg["backbone"]["depth"],
            "num_heads": cfg["backbone"]["num_heads"],
            "mlp_ratio": cfg["backbone"]["mlp_ratio"],
            "drop_rate": cfg["backbone"]["drop_rate"],
            "window_size": cfg["backbone"]["window_size"],
            "dp_rank": cfg["backbone"]["dp_rank"],
            "learned_flow": cfg["backbone"]["learned_flow"],
            "finetune": cfg["backbone"]["finetune"],
            # head
            "head_type": cfg["head"]["type"],
            "pooling_type": cfg["backbone"]["pooling_type"],
            "hidden_channels": cfg["head"]["hyper_parameters"]["hidden_channels"],
            "dropout_head": cfg["head"]["hyper_parameters"]["dropout"],
        }
    )

    return wandb_logger
