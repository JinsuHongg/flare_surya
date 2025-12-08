from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    EarlyStopping,
    # ModelSummary
)


def build_callbacks(cfg, wandb_logger):
    ckpt_name = (
        f"{wandb_logger.experiment.id}_"
        f"{cfg['etc']['ckpt_file_name']}_"
        f"{cfg['head']['type']}_"
        "{epoch}-{val_loss:.4f}"
    )

    return [
        RichProgressBar(),
        # ModelSummary(max_depth=-1),
        LearningRateMonitor(logging_interval="step"),
        # EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            monitor=cfg["optimizer"]["scheduler"]["monitor"],
            dirpath=cfg["etc"]["ckpt_dir"],
            filename=ckpt_name,
            save_top_k=3,
            verbose=True,
            mode="min",
        ),
    ]
