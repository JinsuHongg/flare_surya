from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    # EarlyStopping,
    # ModelSummary
)


def build_callbacks(cfg, wandb_logger):

    best_val_ckpt = ModelCheckpoint(
        monitor=cfg.optimizer.scheduler.monitor,
        dirpath=cfg.etc.ckpt_dir,
        filename=(
            f"{wandb_logger.experiment.id}_"
            f"{cfg.etc.ckpt_name_tag}_"
            f"{cfg.head.type}_"
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
            f"{cfg.head.type}_"
            "{epoch}-{step}-{val_loss:.4f}"
        ),
        every_n_train_steps=cfg.etc.every_n_train_steps,
        save_top_k=-1,
        verbose=True,
    )

    last_ckpt = ModelCheckpoint(
        dirpath=cfg.etc.ckpt_dir,
        save_last=True,
        filename="last",
        verbose=True,
    )

    return [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        best_val_ckpt,
        step_ckpt,
        last_ckpt,
    ]
