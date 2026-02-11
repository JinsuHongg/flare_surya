import os
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger as lgr_logger

import torch
from lightning.pytorch import Trainer

from flare_surya.datamodule import (
    FlareDataModule,
    FlareDataModuleAWS,
    FlareDataModuleZarr,
)
from flare_surya.models.modules import FlareSurya

# from flare_surya.utils.config import load_config
from flare_surya.utils.logger_utils import build_wandb
from flare_surya.utils.callbacks import build_callbacks

torch.set_float32_matmul_precision("medium")


def build_model(cfg):

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
        # Put finetuning additions below this line
        "token_type": cfg.head.token_type,
        "in_feature": cfg.head.hyper_parameters.in_feature[cfg.head.token_type],
        "head_type": cfg.head.type,
        "head_layer_dict": cfg.head.hyper_parameters,
        "freeze_backbone": cfg.backbone.freeze_backbone,
        "lora_dict": cfg.lora,
        "optimizer_dict": cfg.optimizer,
        "threshold": cfg.head.threshold,
        "log_step_size": cfg.head.log_step_size,
        "save_test_results_path": cfg.etc.save_test_results_path,
    }
    if cfg.etc.resume and cfg.etc.ckpt_weights_only:

        ckpt_path = os.path.join(
            cfg.etc.ckpt_dir,
            cfg.etc.ckpt_file,
        )

        ckpt = torch.load(
            ckpt_path,
            weights_only=True,
            map_location="cpu",
        )

        lgr_logger.info("Resuming from checkpoint weights only...")
        lgr_logger.info(f"ckpt: {cfg.etc.ckpt_file}")

        # load weights and hyperparameters
        model = FlareSurya(**model_hyperparameter)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model = FlareSurya(**model_hyperparameter)

    return model


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="first_experiment_model_comparison.yaml",
)
def train(cfg: OmegaConf):

    # Datamodule
    datamodule = FlareDataModuleZarr(cfg=cfg)

    # Load model
    model = build_model(cfg=cfg)

    # Create wandb obejct
    wandb_logger = build_wandb(cfg=cfg, model=model)

    # Trainer
    callbacks = build_callbacks(cfg=cfg, wandb_logger=wandb_logger)
    trainer = Trainer(
        accelerator=cfg.etc.accelerator,
        devices=cfg.etc.devices,
        num_nodes=cfg.etc.num_nodes,
        max_epochs=cfg.etc.max_epochs,
        precision=cfg.etc.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.head.log_step_size,
        limit_train_batches=cfg.etc.limit_train_batches,
        limit_val_batches=cfg.etc.limit_val_batches,
        strategy=cfg.etc.strategy,
    )

    lgr_logger.info("Start training...")
    if cfg.etc.phase == "train":
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=(
                os.path.join(cfg.etc.ckpt_dir, cfg.etc.ckpt_file)
                if cfg.etc.resume and not cfg.etc.ckpt_weights_only
                else None
            ),
        )
    elif cfg.etc.phase == "test":
        trainer.test(
            model=model,
            dataloaders=datamodule,
            ckpt_path=os.path.join(cfg.etc.ckpt_dir, cfg.etc.ckpt_file),
            verbose=True,
        )


if __name__ == "__main__":

    train()
