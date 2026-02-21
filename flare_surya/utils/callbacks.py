import time
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
    # EarlyStopping,
    # ModelSummary
)


class PerformanceMonitor(pl.Callback):
    def __init__(self):
        self.last_batch_end_time = None
        self.gpu_start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Runs AFTER data is loaded and moved to GPU.
        The time since the LAST batch ended is purely Data Loading (I/O).
        """
        self.gpu_start_time = time.perf_counter()

        if self.last_batch_end_time is not None:
            # Calculate the "dead time" the GPU spent waiting for data
            loading_time = self.gpu_start_time - self.last_batch_end_time

            pl_module.log(
                "perf/data_loading_seconds",
                loading_time,
                on_step=True,
                prog_bar=False,
                sync_dist=False,  # Do not sync timers!
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Runs after the Forward/Backward pass is done.
        """
        now = time.perf_counter()

        # Calculate how long the GPU actually worked
        compute_time = now - self.gpu_start_time

        pl_module.log(
            "perf/gpu_compute_seconds",
            compute_time,
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        # Mark the end time for the NEXT batch's calculation
        self.last_batch_end_time = now


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

    epoch_ckpt = ModelCheckpoint(
        dirpath=cfg.etc.ckpt_dir,
        filename=(
            f"{cfg.etc.ckpt_name_tag}_"
            f"{cfg.head.type}_lastepoch"
        ),
        save_on_train_epoch_end=True,
        save_top_k=-1,
        verbose=True,
    )
 
    performance_monitor = PerformanceMonitor()
    return [
        # RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        best_val_ckpt,
        epoch_ckpt,
        performance_monitor,
    ]
