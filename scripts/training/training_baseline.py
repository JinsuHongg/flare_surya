import warnings

# Suppress the specific FutureWarning from timm
warnings.filterwarnings(
    "ignore", "Importing from timm.models.layers is deprecated.*", FutureWarning
)

import os
import hydra
import datetime
from omegaconf import DictConfig, OmegaConf
from loguru import logger as lgr_logger

import torch
import torch.multiprocessing as mp
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from flare_surya.datamodule import FlareDataModule
from flare_surya.models import BaseLineModel
from flare_surya.utils.callbacks import build_baseline_callbacks

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
        loss_dict=config.loss,
        save_test_results_path=config.etc.save_test_results_path,
    )


@hydra.main(
    version_base=None,
    config_path="../configs/nas/",
    config_name="baselines_exp",
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
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        config=cfg_dict,
        id=cfg.wandb.id,
        resume=cfg.wandb.resume,
    )

    callbacks = build_baseline_callbacks(cfg, wandb_id=wandb_logger.experiment.id)

    trainer = Trainer(
        enable_progress_bar=False,
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
        accumulate_grad_batches=cfg.etc.accumulate_grad_batches,
        gradient_clip_val=cfg.etc.gradient_clip_val,
        gradient_clip_algorithm=cfg.etc.gradient_clip_algorithm,
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
            weights_only=False,
        )
    elif cfg.etc.phase == "test":
        trainer.test(
            model=model,
            dataloaders=datamodule,
            ckpt_path=os.path.join(cfg.etc.ckpt_dir, cfg.etc.ckpt_file),
            verbose=True,
            weights_only=False,
        )


if __name__ == "__main__":
    # Set the start method to 'spawn' for cleaner, safer worker processes.
    # This must be done inside the __main__ block and before any other
    # multiprocessing or CUDA code is called.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Can only be set once

    train()
