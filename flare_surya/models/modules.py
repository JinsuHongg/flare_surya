import os
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError, R2Score
import wandb
from loguru import logger as lgr_logger
from peft import LoraConfig, get_peft_model
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import (
    AlexNetClassifier,
    ResNet18Classifier,
    ResNet34Classifier,
    ResNet50Classifier,
)
from terratorch_surya.models.helio_spectformer import HelioSpectFormer

from flare_surya.metrics.classification_metrics import DistributedClassificationMetrics
from .base import BaseModule
from .heads import SuryaHead
from .criterions import FlareSSMLoss, get_criterion

# SecondaryEncoder is now SolarEncoder from solar_models.py
from .solar_models import SolarEncoder as SecondaryEncoder
from .fusion import FusionModule
from flare_surya.models.solar_models import (
    SolarDecoder,
    SolarEncoder,
)


class SolarPretrainingMetrics(MetricCollection):
    def __init__(self, prefix: str = None, postfix: str = None):
        super().__init__(
            {
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            },
            prefix=prefix,
            postfix=postfix,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_flat = preds.reshape(preds.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        super().update(preds_flat, target_flat)

    def compute(self):
        computed = super().compute()
        if "mse" in computed:
            computed["rmse"] = torch.sqrt(computed["mse"])
        return computed


class FlareSurya(BaseModule):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        time_embedding,
        depth,
        num_heads,
        mlp_ratio,
        drop_rate,
        dtype,
        window_size,
        dp_rank,
        learned_flow,
        use_latitude_in_learned_flow,
        init_weights,
        checkpoint_layers,
        n_spectral_blocks,
        rpe,
        ensemble,
        finetune,
        nglo,
        path_weights,
        # head parameters
        pooling_type,
        head_type,
        head_layer_dict,
        freeze_backbone,
        lora_dict,
        optimizer_dict,
        loss_dict,
        threshold=0.5,
        # misc
        save_embeddings_path=None,
        save_test_results_path=None,
    ):
        super().__init__(optimizer_dict=optimizer_dict)
        self.save_hyperparameters()
        self.pooling_type = pooling_type
        self.save_test_results_path = save_test_results_path

        self.backbone = HelioSpectFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            dtype=dtype,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            nglo=nglo,
            rpe=rpe,
            ensemble=ensemble,
            finetune=finetune,
        )

        # load pretrained weights for backbone
        if path_weights:
            lgr_logger.info(f"Pretrained weights loaded from: {path_weights}")
            weights = torch.load(
                path_weights, map_location=torch.device("cpu"), weights_only=True
            )
            self.backbone.load_state_dict(weights, strict=False)

        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        if self.lora_dict["use"]:
            config = LoraConfig(**self.lora_dict["config"])
            # get peft model
            self.backbone = get_peft_model(self.backbone, config)

        if self.pooling_type == "attention_pooling":
            self.attn_pooling = nn.Linear(embed_dim, 1)

        # define head
        self.head = SuryaHead(
            in_feature=embed_dim,
            layer_type=head_type,
            layer_dict=head_layer_dict,
        )

        self.criterion = get_criterion(loss_dict)

        # Initialize the metrics instances
        self.train_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.val_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.test_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.threshold = threshold
        self.save_embeddings_path = save_embeddings_path
        if save_embeddings_path is not None:
            self.embeddings_buffer = {"timestamps": [], "embeddings": []}
            self.embeddings_zarr = None

    def forward_features(self, data):
        """
        Helper method to handle backbone forward pass and pooling.
        Reduces repetition across train/val/test steps.
        """
        tokens = self.backbone(data)

        match self.pooling_type:
            case "cls_token":
                return tokens[:, 0, :]
            case "avg_pooling":
                return tokens.mean(dim=1)
            case "max_pooling":
                return tokens.max(dim=1).values
            case "attention_pooling":
                w = torch.softmax(self.attn_pooling(tokens), dim=1)  # [B, T, 1]
                return (w * tokens).sum(dim=1)  # [B, D]
            case _:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def train(self, mode=True):
        """
        Override train mode to ensure frozen backbone stays in eval mode
        (disabling Dropout/Batchnorm updates) during finetuning.
        """
        super().train(mode)
        if self.freeze_backbone and mode:
            self.backbone.eval()

            # Re-enable training ONLY for the trainable adapters
            for name, module in self.backbone.named_modules():
                if hasattr(module, "training") and any(
                    p.requires_grad for p in module.parameters()
                ):
                    module.train()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, metadata = batch
        stats = data["debug"]
        target = data["label"].float().unsqueeze(1)
        tokens = self.forward_features(data)

        if isinstance(self.criterion, FlareSSMLoss):
            x_hat, h = self.head.forward_with_hidden(tokens)
            loss = self.criterion(x_hat, target, h, current_epoch=self.current_epoch)
        else:
            x_hat = self.head(tokens)
            loss = self.criterion(x_hat, target)

        probs = torch.sigmoid(x_hat)

        # Update Metrics
        self.train_metrics.update(probs, target)

        self.log_dict(
            {
                "perf/file_open_latency_sec": stats["open_time"].float().mean(),
                "perf/file_read_bandwidth_sec": stats["read_time"].float().mean(),
                "perf/cpu_to_gpu_sec": stats["cpu_to_gpu_sec"],
            },
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        # Log training loss every step
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
            batch_size=target.shape[0],
        )

        return loss

    def on_train_epoch_end(self):
        # Compute, log, and reset the metrics accumulated over the last 100 steps
        metrics = self.train_metrics.compute()

        # Log the computed metrics
        self.log_dict(
            {f"train/{k}": v.float() for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        tokens = self.forward_features(data)

        if isinstance(self.criterion, FlareSSMLoss):
            x_hat, h = self.head.forward_with_hidden(tokens)
            loss = self.criterion(x_hat, target, h, current_epoch=self.current_epoch)
        else:
            x_hat = self.head(tokens)
            loss = self.criterion(x_hat, target)

        probs = torch.sigmoid(x_hat)

        # Update states
        self.val_metrics.update(probs, target)

        # Log step loss normally
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        # Compute global metrics (Auto-synced across GPUs)
        metrics = self.val_metrics.compute()

        self.log_dict(
            {f"val/{k}": v.float() for k, v in metrics.items()}, sync_dist=True
        )

        # Reset for next epoch
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        tokens = self.forward_features(data)

        if isinstance(self.criterion, FlareSSMLoss):
            x_hat, h = self.head.forward_with_hidden(tokens)
            loss = self.criterion(x_hat, target, h, current_epoch=self.current_epoch)
        else:
            x_hat = self.head(tokens)
            loss = self.criterion(x_hat, target)

        probs = torch.sigmoid(x_hat)

        # Store predictions and targets for later analysis
        self.test_results["timestamps"].append(
            metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        )
        self.test_results["targets"].append(target.item())
        self.test_results["predictions"].append(probs.item())

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(probs, target)

        # Log Test Loss
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Every rank saves its own shard
        if batch_idx % 50 == 0:
            self._flush_test_results()

        return loss

    def predict_step(self, batch, batch_idx):
        data, metadata = batch
        tokens = self.backbone(data)

        timestamps = metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        embeddings = tokens.detach().cpu().numpy()

        if self.save_embeddings_path:
            zarr_path = self.save_embeddings_path

            # Check if zarr file exists
            if not os.path.exists(zarr_path):
                # Create new zarr file with xarray
                ds = xr.Dataset(
                    {
                        "embeddings": (["time", "token_dim"], embeddings),
                        "timestamps": (["time"], np.array(timestamps, dtype=np.int64)),
                    }
                )
                ds.to_zarr(zarr_path, mode="w")
                lgr_logger.info(f"Created new zarr file: {zarr_path}")
            else:
                # Load existing zarr file and check which timestamps are new
                existing_ds = xr.open_zarr(zarr_path)
                existing_timestamps = existing_ds["timestamps"].values
                current_timestamps_arr = np.array(timestamps, dtype=np.int64)

                # Find timestamps that don't exist yet
                existing_set = set(existing_timestamps)
                new_mask = np.array(
                    [ts not in existing_set for ts in current_timestamps_arr]
                )

                # Only save new timestamps
                new_timestamps = current_timestamps_arr[new_mask]
                new_embeddings = embeddings[new_mask]

                if len(new_timestamps) > 0:
                    # Append new data using xarray
                    # Load existing data, concat with new data, and save
                    existing_embeddings = existing_ds["embeddings"].values
                    combined_embeddings = np.concatenate(
                        [existing_embeddings, new_embeddings], axis=0
                    )
                    combined_timestamps = np.concatenate(
                        [existing_timestamps, new_timestamps]
                    )

                    ds = xr.Dataset(
                        {
                            "embeddings": (["time", "token_dim"], combined_embeddings),
                            "timestamps": (["time"], combined_timestamps),
                        }
                    )
                    ds.to_zarr(zarr_path, mode="w")
                    lgr_logger.info(
                        f"Appended {len(new_timestamps)} new embeddings to {zarr_path}"
                    )
                else:
                    lgr_logger.info(
                        f"All {len(timestamps)} timestamps already exist in {zarr_path}"
                    )

        return {"embeddings": tokens}

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v.float():.4f}")
        print("===================\n")
        self.log_dict(
            {f"test/{k}": v.float() for k, v in metrics.items()}, sync_dist=False
        )

        # Log ROC and PR curves to wandb
        if self.test_results["predictions"]:
            y_true = np.array(self.test_results["targets"])
            y_probas = np.array(self.test_results["predictions"])

            # Log ROC curve
            self.log({"test/roc_curve": wandb.plot.roc_curve(y_true, y_probas)})

            # Log PR curve
            self.log({"test/pr_curve": wandb.plot.pr_curve(y_true, y_probas)})

            # Compute skill scores at multiple thresholds
            thresholds = np.linspace(0.01, 0.99, 99)
            tss_scores = []
            hss_scores = []
            f1_macro_scores = []

            for thresh in thresholds:
                preds = (y_probas > thresh).astype(int)

                tp = np.sum((preds == 1) & (y_true == 1))
                tn = np.sum((preds == 0) & (y_true == 0))
                fp = np.sum((preds == 1) & (y_true == 0))
                fn = np.sum((preds == 0) & (y_true == 1))

                eps = 1e-7

                # TSS = Sensitivity + Specificity - 1
                sensitivity = tp / (tp + fn + eps)
                specificity = tn / (tn + fp + eps)
                tss = sensitivity + specificity - 1
                tss_scores.append(tss)

                # HSS
                numerator = 2 * (tp * tn - fn * fp)
                denominator = (tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)
                hss = numerator / (denominator + eps)
                hss_scores.append(hss)

                # F1-macro (average of F1 for each class)
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1_pos = 2 * (precision * recall) / (precision + recall + eps)

                # F1 for negative class
                precision_neg = tn / (tn + fn + eps)
                recall_neg = tn / (tn + fp + eps)
                f1_neg = (
                    2
                    * (precision_neg * recall_neg)
                    / (precision_neg + recall_neg + eps)
                )

                f1_macro = (f1_pos + f1_neg) / 2
                f1_macro_scores.append(f1_macro)

            # Log skill scores table and line plot to wandb
            df = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "TSS": tss_scores,
                    "HSS": hss_scores,
                    "F1_macro": f1_macro_scores,
                }
            )
            self.log({"test/threshold_df": wandb.Table(dataframe=df)})
            self.log(
                {
                    "test/threshold_vs_scores": wandb.plot.line_series(
                        xs=thresholds,
                        ys=[tss_scores, hss_scores, f1_macro_scores],
                        keys=["TSS", "HSS", "F1_macro"],
                        title="Threshold vs Skill Scores",
                        xname="Threshold",
                    )
                }
            )

        self.test_metrics.reset()
        self._flush_test_results()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        t0 = time.perf_counter()
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        t1 = time.perf_counter()

        data, metadata = batch

        if isinstance(data, dict):
            if "debug" not in data:
                data["debug"] = {}

            data["debug"]["cpu_to_gpu_sec"] = t1 - t0

        return (data, metadata)

    def _flush_test_results(self, mode="a"):
        if not self.test_results["predictions"]:  # nothing buffered, skip
            return
        df = pd.DataFrame(self.test_results)
        write_header = not os.path.exists(self.save_test_results_path)
        df.to_csv(
            self.save_test_results_path, mode=mode, header=write_header, index=False
        )
        self.test_results = {"timestamps": [], "predictions": [], "targets": []}


class BaseLineModel(BaseModule):
    def __init__(
        self,
        model_name,
        in_channels,
        time_steps,
        num_classes,
        p_drop,
        threshold,
        # head parameters
        optimizer_dict,
        loss_dict,
        # misc
        save_test_results_path=None,
    ):
        super().__init__(optimizer_dict=optimizer_dict)
        self.save_hyperparameters()
        self.save_test_results_path = save_test_results_path
        self.test_results = {
            "timestamps": [],
            "predictions": [],
            "targets": [],
        }
        self.model_name = model_name

        match model_name:
            case "alexnet":
                self.backbone = AlexNetClassifier(
                    in_channels=in_channels,
                    time_steps=time_steps,
                    num_classes=num_classes,
                    dropout=p_drop,
                )
            case "resnet18":
                self.backbone = ResNet18Classifier(
                    in_channels=in_channels,
                    time_steps=time_steps,
                    num_classes=num_classes,
                    dropout=p_drop,
                )
            case "resnet34":
                self.backbone = ResNet34Classifier(
                    in_channels=in_channels,
                    time_steps=time_steps,
                    num_classes=num_classes,
                    dropout=p_drop,
                )
            case "resnet50":
                self.backbone = ResNet50Classifier(
                    in_channels=in_channels,
                    time_steps=time_steps,
                    num_classes=num_classes,
                    dropout=p_drop,
                )
            case _:
                raise ValueError(f"Unknown model_name: {model_name}")

        # define loss
        self.criterion = get_criterion(loss_dict, module_name="BaseLineModel")

        # Initialize the metrics instances
        self.train_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.val_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.test_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.threshold = threshold

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, metadata = batch
        stats = data["debug"]
        target = data["label"].float()
        x_hat = self.backbone(data)
        probs = torch.sigmoid(x_hat)

        loss = self.criterion(x_hat, target)

        # Update Metrics
        self.train_metrics.update(probs, target)

        self.log_dict(
            {
                "perf/file_open_latency_sec": stats["open_time"].float().mean(),
                "perf/file_read_bandwidth_sec": stats["read_time"].float().mean(),
                "perf/cpu_to_gpu_sec": stats["cpu_to_gpu_sec"],
            },
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        # Log training loss every step
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return loss

    def on_train_epoch_end(self):
        # Compute, log, and reset the metrics accumulated over the last 100 steps
        metrics = self.train_metrics.compute()

        # Log the computed metrics
        self.log_dict(
            {f"train/{k}": v.float() for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float()
        x_hat = self.backbone(data)
        probs = torch.sigmoid(x_hat)

        loss = self.criterion(x_hat, target)

        # Log Training Loss
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=target.shape[0],
        )

        # Update Metrics (x_hat contains the logits)
        self.val_metrics.update(probs, target)

        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = self.val_metrics.compute()

        # Log all computed metrics
        self.log_dict(
            {f"val/{k}": v.float() for k, v in metrics.items()}, sync_dist=True
        )

        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float()
        x_hat = self.backbone(data)
        probs = torch.sigmoid(x_hat)

        # Store predictions and targets for later analysis
        self.test_results["timestamps"].append(
            metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        )
        self.test_results["targets"].append(target.item())
        self.test_results["predictions"].append(probs.item())

        # Calculate Loss
        loss = self.criterion(x_hat, target)

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(probs, target)

        # Log Test Loss
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Every rank saves its own shard
        if batch_idx % 50 == 0:
            self._flush_test_results()

        return loss

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v.float():.4f}")
        print("===================\n")
        self.log_dict(
            {f"test/{k}": v.float() for k, v in metrics.items()}, sync_dist=False
        )

        # Log ROC and PR curves to wandb
        if self.test_results["predictions"]:
            y_true = np.array(self.test_results["targets"])
            y_probas = np.array(self.test_results["predictions"])

            # Log ROC curve
            self.log({"test/roc_curve": wandb.plot.roc_curve(y_true, y_probas)})

            # Log PR curve
            self.log({"test/pr_curve": wandb.plot.pr_curve(y_true, y_probas)})

            # Compute skill scores at multiple thresholds
            thresholds = np.linspace(0.01, 0.99, 99)
            tss_scores = []
            hss_scores = []
            f1_macro_scores = []

            for thresh in thresholds:
                preds = (y_probas > thresh).astype(int)

                tp = np.sum((preds == 1) & (y_true == 1))
                tn = np.sum((preds == 0) & (y_true == 0))
                fp = np.sum((preds == 1) & (y_true == 0))
                fn = np.sum((preds == 0) & (y_true == 1))

                eps = 1e-7

                # TSS = Sensitivity + Specificity - 1
                sensitivity = tp / (tp + fn + eps)
                specificity = tn / (tn + fp + eps)
                tss = sensitivity + specificity - 1
                tss_scores.append(tss)

                # HSS
                numerator = 2 * (tp * tn - fn * fp)
                denominator = (tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)
                hss = numerator / (denominator + eps)
                hss_scores.append(hss)

                # F1-macro (average of F1 for each class)
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1_pos = 2 * (precision * recall) / (precision + recall + eps)

                # F1 for negative class
                precision_neg = tn / (tn + fn + eps)
                recall_neg = tn / (tn + fp + eps)
                f1_neg = (
                    2
                    * (precision_neg * recall_neg)
                    / (precision_neg + recall_neg + eps)
                )

                f1_macro = (f1_pos + f1_neg) / 2
                f1_macro_scores.append(f1_macro)

            # Log skill scores table and line plot to wandb
            df = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "TSS": tss_scores,
                    "HSS": hss_scores,
                    "F1_macro": f1_macro_scores,
                }
            )
            self.log({"test/threshold_df": wandb.Table(dataframe=df)})
            self.log(
                {
                    "test/threshold_vs_scores": wandb.plot.line_series(
                        xs=thresholds,
                        ys=[tss_scores, hss_scores, f1_macro_scores],
                        keys=["TSS", "HSS", "F1_macro"],
                        title="Threshold vs Skill Scores",
                        xname="Threshold",
                    )
                }
            )

        self.test_metrics.reset()
        self._flush_test_results()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        t0 = time.perf_counter()
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        t1 = time.perf_counter()

        data, metadata = batch

        if isinstance(data, dict):
            if "debug" not in data:
                data["debug"] = {}

            data["debug"]["cpu_to_gpu_sec"] = t1 - t0

        return (data, metadata)

    def _flush_test_results(self, mode="a"):
        if not self.test_results["predictions"]:  # nothing buffered, skip
            return
        df = pd.DataFrame(self.test_results)
        write_header = not os.path.exists(self.save_test_results_path)
        df.to_csv(
            self.save_test_results_path, mode=mode, header=write_header, index=False
        )
        self.test_results = {"timestamps": [], "predictions": [], "targets": []}


class SuryaMultiModal(BaseModule):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        time_embedding,
        depth,
        num_heads,
        mlp_ratio,
        drop_rate,
        dtype,
        window_size,
        dp_rank,
        learned_flow,
        use_latitude_in_learned_flow,
        init_weights,
        checkpoint_layers,
        n_spectral_blocks,
        rpe,
        ensemble,
        finetune,
        nglo,
        path_weights,
        # second modality parameters
        in_channels,
        seq_len,
        secondary_embed_dim,
        secondary_depth,
        secondary_num_heads,
        # head parameters
        pooling_type,
        head_type,
        head_layer_dict,
        freeze_backbone,
        lora_dict,
        optimizer_dict,
        loss_dict,
        threshold=0.5,
        # fusion parameters
        fusion_type="concat",
        fuse_embed_dim=1280,
        # secondary encoder parameters
        secondary_pooling_type="avg_pooling",
        # misc
        save_test_results_path=None,
    ):
        super().__init__(optimizer_dict=optimizer_dict)
        self.save_hyperparameters(ignore=["optimizer_dict", "loss_dict", "lora_dict"])
        self.pooling_type = pooling_type
        self.secondary_pooling_type = secondary_pooling_type
        self.freeze_backbone = freeze_backbone
        self.lora_dict = lora_dict
        self.test_results = {
            "timestamps": [],
            "predictions": [],
            "targets": [],
        }
        self.save_test_results_path = save_test_results_path

        self.backbone = HelioSpectFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            time_embedding=time_embedding,
            depth=depth,
            n_spectral_blocks=n_spectral_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            dtype=dtype,
            window_size=window_size,
            dp_rank=dp_rank,
            learned_flow=learned_flow,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            init_weights=init_weights,
            checkpoint_layers=checkpoint_layers,
            nglo=nglo,
            rpe=rpe,
            ensemble=ensemble,
            finetune=finetune,
        )

        self.secondary_encoder = SecondaryEncoder(
            in_channels=in_channels,
            seq_len=seq_len,
            embed_dim=secondary_embed_dim,
            depth=secondary_depth,
            num_heads=secondary_num_heads,
        )

        self.fusion = FusionModule(
            fusion_type=fusion_type,
            img_dim=embed_dim,
            secondary_dim=secondary_embed_dim,
            fuse_dim=fuse_embed_dim,
            num_heads=secondary_num_heads,
        )

        # load pretrained weights for backbone
        if path_weights:
            lgr_logger.info(f"Pretrained weights loaded from: {path_weights}")
            weights = torch.load(
                path_weights, map_location=torch.device("cpu"), weights_only=True
            )
            self.backbone.load_state_dict(weights, strict=False)

        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        if self.lora_dict["use"]:
            config = LoraConfig(**self.lora_dict["config"])
            # get peft model
            self.backbone = get_peft_model(self.backbone, config)

        if self.pooling_type == "attention_pooling":
            self.attn_pooling = nn.Linear(embed_dim, 1)

        # define head
        in_feature = self.fusion.output_dim

        # Create post-fusion pooling layer for cross_attention fusion
        if self.pooling_type == "attention_pooling":
            self.post_fusion_pooling = nn.Linear(in_feature, 1)

        # Create secondary encoder pooling layer
        if self.secondary_pooling_type == "attention_pooling":
            self.secondary_attn_pooling = nn.Linear(secondary_embed_dim, 1)

        self.head = SuryaHead(
            in_feature=in_feature,
            layer_type=head_type,
            layer_dict=head_layer_dict,
        )

        self.criterion = get_criterion(loss_dict, module_name="SuryaMultiModal")

        # Initialize the metrics instances
        self.train_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.val_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.test_metrics = DistributedClassificationMetrics(threshold=threshold)
        self.threshold = threshold

    def forward_features(self, data):
        """
        Helper method to handle backbone forward pass and pooling.
        Reduces repetition across train/val/test steps.
        """
        tokens = self.backbone(data)
        return self.pool_tokens(tokens)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to tokens."""
        match self.pooling_type:
            case "cls_token":
                return tokens[:, 0, :]
            case "avg_pooling":
                return tokens.mean(dim=1)
            case "max_pooling":
                return tokens.max(dim=1).values
            case "attention_pooling":
                return self._attention_pool(tokens)
            case _:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

    def _attention_pool(
        self, tokens: torch.Tensor, attn_layer: nn.Module | None = None
    ) -> torch.Tensor:
        """Apply attention-based pooling."""
        attn_layer = attn_layer or self.attn_pooling
        w = torch.softmax(attn_layer(tokens), dim=1)  # [B, T, 1]
        return (w * tokens).sum(dim=1)  # [B, D]

    def pool_secondary(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to secondary tokens."""
        match self.secondary_pooling_type:
            case "avg_pooling":
                return tokens.mean(dim=1)
            case "max_pooling":
                return tokens.max(dim=1).values
            case "attention_pooling":
                return self._attention_pool(tokens, self.secondary_attn_pooling)
            case "none":
                return tokens  # keep as sequence
            case _:
                raise ValueError(
                    f"Unknown secondary_pooling_type: {self.secondary_pooling_type}"
                )

    def forward(self, data):
        if self.fusion.requires_pooled:
            img_tokens = self.forward_features(data)
        else:
            img_tokens = self.backbone(data)

        secondary_tokens = self.secondary_encoder(data["xrs"].to(self.device))

        if self.fusion.requires_pooled:
            secondary_tokens = self.pool_secondary(secondary_tokens)

        tokens = self.fusion(img_tokens, secondary_tokens)

        if not self.fusion.requires_pooled:
            if self.pooling_type == "attention_pooling":
                tokens = self._attention_pool(tokens, self.post_fusion_pooling)
            else:
                tokens = self.pool_tokens(tokens)

        if isinstance(self.criterion, FlareSSMLoss):
            x_hat, h = self.head.forward_with_hidden(tokens)
            return {"logits": x_hat, "hidden": h}
        else:
            x_hat = self.head(tokens)
            return {"logits": x_hat}

    def _compute_loss(self, output, target):
        x_hat = output["logits"]
        target = target.to(x_hat.device)
        if isinstance(self.criterion, FlareSSMLoss):
            loss = self.criterion(
                x_hat, target, output["hidden"], current_epoch=self.current_epoch
            )
        elif (
            isinstance(self.criterion, tuple) and self.criterion[0] == "bce_with_logits"
        ):
            class_weights = self.criterion[1]
            if class_weights is not None:
                pos_weight = torch.tensor(
                    [class_weights[1] / class_weights[0]], device=x_hat.device
                )
                loss = F.binary_cross_entropy_with_logits(
                    x_hat, target, pos_weight=pos_weight
                )
            else:
                loss = F.binary_cross_entropy_with_logits(x_hat, target)
        else:
            loss = self.criterion(x_hat, target)
        return x_hat, loss

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone and mode:
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, metadata = batch
        stats = data["debug"]
        target = data["label"].float().unsqueeze(1)

        output = self(data)
        x_hat, loss = self._compute_loss(output, target)
        probs = torch.sigmoid(x_hat)

        # Update Metrics
        self.train_metrics.update(probs, target)
        self.log_dict(
            {
                "perf/file_open_latency_sec": stats["open_time"].float().mean(),
                "perf/file_read_bandwidth_sec": stats["read_time"].float().mean(),
                "perf/cpu_to_gpu_sec": stats["cpu_to_gpu_sec"],
            },
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        # Log training loss every step
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=target.shape[0],
        )

        return loss

    def on_train_epoch_end(self):
        # Compute, log, and reset the metrics accumulated over the last 100 steps
        metrics = self.train_metrics.compute()

        # Log the computed metrics
        self.log_dict(
            {f"train/{k}": v.float() for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        output = self(data)
        x_hat, loss = self._compute_loss(output, target)

        probs = torch.sigmoid(x_hat)

        # Update states
        self.val_metrics.update(probs, target)

        # Log step loss normally
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=target.shape[0],
        )

    def on_validation_epoch_end(self):
        # Compute global metrics (Auto-synced across GPUs)
        metrics = self.val_metrics.compute()

        self.log_dict(
            {f"val/{k}": v.float() for k, v in metrics.items()}, sync_dist=True
        )

        # Reset for next epoch
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        output = self(data)
        x_hat, loss = self._compute_loss(output, target)
        probs = torch.sigmoid(x_hat)

        # Store predictions and targets for later analysis
        self.test_results["timestamps"].append(
            metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        )
        self.test_results["targets"].extend(target.detach().cpu().squeeze(1).tolist())
        self.test_results["predictions"].extend(
            probs.detach().cpu().squeeze(1).tolist()
        )

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(probs, target)

        # Log Test Loss
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Every rank saves its own shard
        if batch_idx % 50 == 0:
            self._flush_test_results()

        return loss

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v.float():.4f}")
        print("===================\n")
        self.log_dict(
            {f"test/{k}": v.float() for k, v in metrics.items()}, sync_dist=False
        )

        # Log ROC and PR curves to wandb
        if self.test_results["predictions"]:
            y_true = np.array(self.test_results["targets"])
            y_probas = np.array(self.test_results["predictions"])

            # Log ROC curve
            self.log({"test/roc_curve": wandb.plot.roc_curve(y_true, y_probas)})

            # Log PR curve
            self.log({"test/pr_curve": wandb.plot.pr_curve(y_true, y_probas)})

            # Compute skill scores at multiple thresholds
            thresholds = np.linspace(0.01, 0.99, 99)
            tss_scores = []
            hss_scores = []
            f1_macro_scores = []

            for thresh in thresholds:
                preds = (y_probas > thresh).astype(int)

                tp = np.sum((preds == 1) & (y_true == 1))
                tn = np.sum((preds == 0) & (y_true == 0))
                fp = np.sum((preds == 1) & (y_true == 0))
                fn = np.sum((preds == 0) & (y_true == 1))

                eps = 1e-7

                # TSS = Sensitivity + Specificity - 1
                sensitivity = tp / (tp + fn + eps)
                specificity = tn / (tn + fp + eps)
                tss = sensitivity + specificity - 1
                tss_scores.append(tss)

                # HSS
                numerator = 2 * (tp * tn - fn * fp)
                denominator = (tp + fn) * (fn + tn) + (tp + fp) * (tn + fp)
                hss = numerator / (denominator + eps)
                hss_scores.append(hss)

                # F1-macro (average of F1 for each class)
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1_pos = 2 * (precision * recall) / (precision + recall + eps)

                # F1 for negative class
                precision_neg = tn / (tn + fn + eps)
                recall_neg = tn / (tn + fp + eps)
                f1_neg = (
                    2
                    * (precision_neg * recall_neg)
                    / (precision_neg + recall_neg + eps)
                )

                f1_macro = (f1_pos + f1_neg) / 2
                f1_macro_scores.append(f1_macro)

            # Log skill scores table and line plot to wandb
            df = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "TSS": tss_scores,
                    "HSS": hss_scores,
                    "F1_macro": f1_macro_scores,
                }
            )
            self.log({"test/threshold_df": wandb.Table(dataframe=df)})
            self.log(
                {
                    "test/threshold_vs_scores": wandb.plot.line_series(
                        xs=thresholds,
                        ys=[tss_scores, hss_scores, f1_macro_scores],
                        keys=["TSS", "HSS", "F1_macro"],
                        title="Threshold vs Skill Scores",
                        xname="Threshold",
                    )
                }
            )

        self.test_metrics.reset()
        self._flush_test_results()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        t0 = time.perf_counter()
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        t1 = time.perf_counter()

        data, metadata = batch

        if isinstance(data, dict):
            if "debug" not in data:
                data["debug"] = {}

            data["debug"]["cpu_to_gpu_sec"] = t1 - t0

        return (data, metadata)

    def _flush_test_results(self, mode="a"):
        if not self.save_test_results_path:
            return
        if not self.test_results["predictions"]:
            return
        df = pd.DataFrame(self.test_results)
        write_header = not os.path.exists(self.save_test_results_path)
        df.to_csv(
            self.save_test_results_path, mode=mode, header=write_header, index=False
        )


class PretrainSolarModel(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        seq_len=1440,
        embed_dim=768,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=12,
        data_type="1d",
        save_embeddings_path: str | None = None,
        mask_ratio: float = 0.5,
        image_size: int = 224,
        patch_size: int = 16,
        optimizer_dict=None,
        loss_dict=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_type = data_type
        self.image_size = image_size
        self.patch_size = patch_size

        self.optimizer_dict = optimizer_dict or {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "eps": 1e-8,
            "betas": [0.9, 0.999],
            "scheduler": {
                "use": "cosine_warmup",
                "monitor": "val/loss",
                "cosine_warmup": {
                    "total_steps": 10000,
                    "warmup_ratio": 0.1,
                    "min_lr": 1e-6,
                },
            },
        }
        self.loss_dict = loss_dict or {"type": "mse"}

        self.encoder = SolarEncoder(
            in_channels=in_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            data_type=data_type,
            image_size=image_size,
            patch_size=patch_size,
        )

        self.decoder = SolarDecoder(
            in_channels=in_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            data_type=data_type,
            image_size=image_size,
        )

        self.data_type = data_type
        self.save_embeddings_path = save_embeddings_path
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pred_timestamps = []
        self.pred_embeddings = []
        self._last_mask_indices = None
        self._last_seq_mask_indices = None

        self.train_metrics = SolarPretrainingMetrics(prefix="train/")
        self.val_metrics = SolarPretrainingMetrics(prefix="val/")
        self.test_metrics = SolarPretrainingMetrics(prefix="test/")

    def random_mask(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        num_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_mask = ids_shuffle[:, :num_mask]

        tokens_masked = tokens.clone()
        mask_token = self.mask_token.to(tokens.dtype).to(tokens.device)

        mask_2d = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        mask_2d.scatter_(1, ids_mask, torch.ones_like(ids_mask, dtype=torch.bool))
        tokens_masked = torch.where(mask_2d.unsqueeze(-1), mask_token, tokens_masked)

        return tokens_masked, ids_restore, ids_mask

    def forward(self, x, use_mask: bool = True):
        tokens = self.encoder.tokenizer(x)

        if use_mask and self.training:
            tokens, ids_restore, ids_mask = self.random_mask(tokens)
            self._last_mask_indices = ids_mask
            self._last_seq_mask_indices = ids_mask
        else:
            self._last_mask_indices = None
            self._last_seq_mask_indices = None

        embedding = self.encoder.encoder(tokens)
        decoded = self.decoder.sequence_decoder(embedding)

        reconstruction = self.decoder.detokenizer(decoded)
        return reconstruction

    def encode(self, x):
        """Encode input to embeddings without decoding."""
        return self.encoder(x)

    def _compute_loss(self, pred, target):
        loss_type = self.loss_dict.get("type", "mse").lower()
        if loss_type == "mae":
            return nn.functional.l1_loss(pred, target)
        elif loss_type == "rmse":
            return torch.sqrt(nn.functional.mse_loss(pred, target))
        else:
            return nn.functional.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        pred = self.forward(x, use_mask=True)
        mask_indices = self._last_seq_mask_indices

        if mask_indices is not None:
            if self.data_type == "2d":
                patch_size = self.encoder.tokenizer.patch_size
                image_size = self.encoder.tokenizer.image_size
                num_patches_per_side = image_size // patch_size

                def to_patches(tensor):
                    B, C, H, W = tensor.shape
                    patches = tensor.unfold(2, patch_size, patch_size).unfold(
                        3, patch_size, patch_size
                    )
                    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
                        B,
                        num_patches_per_side * num_patches_per_side,
                        C,
                        patch_size * patch_size,
                    )
                    patches = patches.mean(-1)
                    return patches

                y_patches = to_patches(y)
                pred_patches = to_patches(pred)

                B_mask, num_mask = mask_indices.shape
                y_masked = y_patches[
                    torch.arange(B_mask, device=mask_indices.device).unsqueeze(1),
                    mask_indices,
                ]
                pred_masked = pred_patches[
                    torch.arange(B_mask, device=mask_indices.device).unsqueeze(1),
                    mask_indices,
                ]
            else:
                y_masked = torch.gather(
                    y,
                    dim=2,
                    index=mask_indices.unsqueeze(1).expand(-1, y.shape[1], -1),
                )
                pred_masked = torch.gather(
                    pred,
                    dim=2,
                    index=mask_indices.unsqueeze(1).expand(-1, pred.shape[1], -1),
                )
            loss = self._compute_loss(pred_masked, y_masked)
            self.train_metrics.update(pred_masked, y_masked)
        else:
            loss = self._compute_loss(pred, y)
            self.train_metrics.update(pred, y)

        self.log("train/loss", loss, prog_bar=True, batch_size=x.shape[0], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        pred = self.forward(x, use_mask=False)

        self.val_metrics.update(pred, y)

        loss = nn.functional.mse_loss(pred, y)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=x.shape[0])

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict({k: v for k, v in metrics.items()}, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict({k: v for k, v in metrics.items()}, sync_dist=True)
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx):
        # Expects batch to contain (x, timestamp) or just x
        # If batch is a tuple, assume it's (x, timestamp)
        if isinstance(batch, (list, tuple)):
            x, timestamps = batch[0], batch[1]
        else:
            x = batch
            timestamps = None

        embeddings = self.encode(x)

        # Store embeddings and timestamps
        # Convert to numpy for storage
        emb_np = embeddings.cpu().detach().numpy()

        if timestamps is not None:
            # Assuming timestamps are datetime or int
            # If they are tensors, convert to list or numpy
            if isinstance(timestamps, torch.Tensor):
                ts_np = timestamps.cpu().detach().numpy()
            else:
                ts_np = np.array(timestamps)
        else:
            ts_np = np.arange(len(emb_np))

        # Append to lists (be careful with memory usage for large datasets)
        self.pred_timestamps.append(ts_np)
        self.pred_embeddings.append(emb_np)

        return embeddings

    def on_predict_epoch_end(self, results):
        # Save embeddings to Zarr
        if self.save_embeddings_path:
            lgr_logger.info(f"Saving embeddings to {self.save_embeddings_path}")

            # Concatenate all batches
            all_timestamps = np.concatenate(self.pred_timestamps, axis=0)
            all_embeddings = np.concatenate(self.pred_embeddings, axis=0)

            # Create xarray Dataset
            # Embeddings shape: [total_samples, seq_len, embed_dim]
            # Or flatten to [total_samples, seq_len * embed_dim] if preferred

            # We need to be careful with dimensions.
            # Let's assume we keep it as [time, seq, dim]

            ds = xr.Dataset(
                {
                    "embeddings": (["timestep", "seq", "feature"], all_embeddings),
                },
                coords={
                    "timestep": all_timestamps,
                },
            )

            # Save to zarr
            # Using consolidated=True for faster loading later
            ds.to_zarr(self.save_embeddings_path, mode="w", consolidated=True)

            lgr_logger.info(
                f"Embeddings saved successfully. Shape: {all_embeddings.shape}"
            )

            # Clear buffers
            self.pred_timestamps = []
            self.pred_embeddings = []

    def configure_optimizers(self):
        optimizer_type = self.optimizer_dict.get("type", "adamw").lower()
        lr = self.optimizer_dict.get("lr", 1e-4)
        weight_decay = self.optimizer_dict.get("weight_decay", 0.0)
        eps = self.optimizer_dict.get("eps", 1e-8)
        betas = tuple(self.optimizer_dict.get("betas", [0.9, 0.999]))

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                betas=betas,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                betas=betas,
            )

        scheduler_cfg = self.optimizer_dict.get("scheduler")
        if scheduler_cfg and scheduler_cfg.get("use") == "cosine_warmup":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            total_steps = self.trainer.estimated_stepping_batches
            # Check for edge cases where Lightning returns infinity or valid steps are unknown
            if isinstance(total_steps, (float, int)) and (
                total_steps == float("inf") or total_steps == 0
            ):
                lgr_logger.warning(
                    "Warning: Could not calculate total steps automatically."
                )
                total_steps = scheduler_cfg["cosine_warmup"].get("total_steps", 10000)

            warmup_ratio = scheduler_cfg["cosine_warmup"].get("warmup_ratio", 0.1)
            min_lr = scheduler_cfg["cosine_warmup"].get("min_lr", 1e-6)

            warmup_steps = int(total_steps * warmup_ratio)
            train_steps = total_steps - warmup_steps

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-6 / lr,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_steps,
                eta_min=min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer
