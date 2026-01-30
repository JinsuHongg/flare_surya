import os
import hydra
import datetime
from omegaconf import DictConfig, OmegaConf
from loguru import logger as lgr_logger

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    # EarlyStopping,
    # ModelSummary
)

from flare_surya.datamodule import FlareDataModule
from flare_surya.models.modules import BaseLineModel

# from flare_surya.utils.logger_utils import build_wandb
# from flare_surya.utils.callbacks import build_callbacks

torch.set_float32_matmul_precision("medium")
# This changes the sharing strategy from RAM (shm) to Disk (file_system)
# torch.multiprocessing.set_sharing_strategy("file_system")


def build_model(config):

    return BaseLineModel(
        model_name=config.backbone.model_name,
        in_channels=config.backbone.in_channels,
        time_steps=config.backbone.time_steps,
        num_classes=config.backbone.num_classes,
        p_drop=config.backbone.p_drop,
        threshold=config.backbone.threshold,
        optimizer_dict=config.optimizer,
        log_step_size=config.backbone.log_step_size,
        save_test_results_path=config.etc.save_test_results_path,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="baseline_experiement.yaml",
)
def train(cfg: OmegaConf):

    # Datamodule
    datamodule = FlareDataModule(cfg=cfg)

    # Model
    model = build_model(config=cfg)

    # Load weights only
    if cfg.etc.resume and cfg.etc.ckpt_weights_only:
        ckpt_path = os.path.join(
            cfg.etc.ckpt_dir,
            cfg.etc.ckpt_file,
        )
        lgr_logger.info(f"Load model weights only...")
        lgr_logger.info(f"ckpt from: {ckpt_path}")
        # only pretrained weights are used
        model = BaseLineModel.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
        )

    # Create wandb obejct
    name = f"{cfg.backbone.model_name}_lr{cfg.optimizer.lr}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    wandb_logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        save_dir=cfg.wandb.save_dir,
        offline=cfg.wandb.offline,
        log_model=cfg.wandb.log_model,
        save_code=cfg.wandb.save_code,
        notes=cfg.wandb.notes,
        tags=cfg.wandb.tag,
        name=name,
        config=cfg_dict,
    )

    # Trainer
    best_val_ckpt = ModelCheckpoint(
        monitor=cfg.optimizer.scheduler.monitor,
        dirpath=cfg.etc.ckpt_dir,
        filename=(
            f"{wandb_logger.experiment.id}_"
            f"{cfg.etc.ckpt_name_tag}_"
            f"{cfg.backbone.model_name}_"
            "{epoch}-{val_loss:.4f}"
        ),
        save_top_k=3,
        verbose=True,
        mode="min",
    )

    step_ckpt = ModelCheckpoint(
        dirpath=cfg.etc.ckpt_dir,
        filename=(
            f"{wandb_logger.experiment.id}_"
            f"{cfg.etc.ckpt_name_tag}_"
            f"{cfg.backbone.model_name}_"
            "{epoch}-{step}"
        ),
        every_n_train_steps=2000,
        save_top_k=-1,
        verbose=True,
    )

    last_ckpt = ModelCheckpoint(
        dirpath=cfg.etc.ckpt_dir,
        save_last=True,
        filename="last",
        verbose=True,
    )
    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        best_val_ckpt,
        step_ckpt,
        last_ckpt,
    ]

    trainer = Trainer(
        accelerator=cfg.etc.accelerator,
        devices=cfg.etc.devices,
        num_nodes=cfg.etc.num_nodes,
        max_epochs=cfg.etc.max_epochs,
        precision=cfg.etc.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.backbone.log_step_size,
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
