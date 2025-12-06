import datetime
from lightning.pytorch.loggers import WandbLogger


def build_wandb(cfg, model):
    name = f"{cfg['head']['type']}_lr{cfg['optimizer']['hyper_parameters']['lr']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb_logger = WandbLogger(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        save_dir=cfg["wandb"]["save_dir"],
        offline=cfg["wandb"]["offline"],
        log_model=cfg["wandb"]["log_model"],
        save_code=cfg["wandb"]["save_code"],
        notes=cfg["wandb"]["notes"],
        tags=cfg["wandb"]["tag"],
        name=name,
        config=cfg
    )

    # selected hparams for WandB
    wandb_logger.log_hyperparams({
        # optimizer / training
        "lr": cfg["optimizer"]["hyper_parameters"]["lr"],
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
        "token_type": cfg["head"]["token_type"],
        "in_channels": cfg["head"]["hyper_parameters"]["in_feature"]["cls_token"],  # example
        "hidden_channels": cfg["head"]["hyper_parameters"]["hidden_channels"],
        "dropout_head": cfg["head"]["hyper_parameters"]["dropout"]
    })

    # wandb_logger.watch(model, log="parameters", log_freq=2000)
    return wandb_logger