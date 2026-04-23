import time
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
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


class TimeLogger(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()
        self._total_batches = trainer.num_training_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:  # print every 50 steps
            elapsed = time.time() - self._epoch_start
            progress = (batch_idx + 1) / self._total_batches
            eta_epoch = (elapsed / progress) - elapsed  # remaining time in this epoch

            print(
                f"  [Epoch {trainer.current_epoch+1} | "
                f"Step {batch_idx+1}/{self._total_batches} ({100*progress:.0f}%)] "
                f"elapsed={elapsed:.1f}s | ETA_epoch={eta_epoch:.0f}s",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", metrics.get("train/loss", float("nan")))
        val_loss = metrics.get("val_loss", metrics.get("val/loss", float("nan")))
        print(
            f"[Epoch {trainer.current_epoch+1}/{trainer.max_epochs}] "
            f"time={elapsed:.1f}s | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}",
            flush=True,
        )

    def on_test_epoch_start(self, trainer, pl_module):
        self._test_epoch_start = time.time()
        self._total_test_batches = trainer.num_test_batches[0]  # list per dataloader

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx % 50 == 0:
            elapsed = time.time() - self._test_epoch_start
            batches_from_prev_loaders = sum(trainer.num_test_batches[:dataloader_idx])
            global_batch_idx = batches_from_prev_loaders + batch_idx

            if self._total_test_batches == 0:
                return

            progress = (global_batch_idx + 1) / self._total_test_batches
            eta = (elapsed / progress) - elapsed
            print(
                f"  [Test | "
                f"Step {batch_idx+1}/{self._total_test_batches} ({100*progress:.0f}%)] "
                f"elapsed={elapsed:.1f}s | ETA={eta:.0f}s",
                flush=True,
            )

    def on_test_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._test_epoch_start
        metrics = trainer.callback_metrics
        test_loss = metrics.get("test/loss", float("nan"))
        print(
            f"[Test Complete] " f"time={elapsed:.1f}s | " f"test_loss={test_loss:.4f}",
            flush=True,
        )


def build_callbacks(cfg):

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.optimizer.scheduler.monitor,  # e.g., "val_loss"
        dirpath=cfg.etc.ckpt_dir,
        filename=(
            f"{cfg.etc.ckpt_name_tag}_" f"{cfg.head.type}_" "{epoch}-{val_hss:.4f}"
        ),
        save_top_k=3,
        mode="max",
        verbose=True,
        save_last=True,
        enable_version_counter=cfg.etc.enable_version_counter,
    )

    performance_monitor = PerformanceMonitor()
    time_monitor = TimeLogger()

    return [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
        performance_monitor,
        time_monitor,
    ]


def build_pretrain_callbacks(cfg):
    """Build callbacks for pretraining without downstream-specific checkpoints."""
    performance_monitor = PerformanceMonitor()
    time_monitor = TimeLogger()

    return [
        LearningRateMonitor(logging_interval="step"),
        performance_monitor,
        time_monitor,
    ]
