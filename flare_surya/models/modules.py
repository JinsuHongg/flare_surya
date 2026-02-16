import os
import time
from pprint import pprint

import pandas as pd
import torch
# from torch import nn
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
# import lightning as L
# from terratorch_surya.downstream_examples.solar_flare_forecasting.models import HelioSpectformer1D
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import (
    AlexNetClassifier, MobileNetClassifier, ResNet18Classifier,
    ResNet34Classifier, ResNet50Classifier)
from terratorch_surya.models.helio_spectformer import HelioSpectFormer

from flare_surya.metrics.classification_metrics import \
    DistributedClassificationMetrics
from flare_surya.models.base import BaseModule
from flare_surya.models.heads import SuryaHead


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
        token_type,
        in_feature,
        head_type,
        head_layer_dict,
        freeze_backbone,
        lora_dict,
        optimizer_dict,
        threshold=0.5,
        log_step_size=100,
        # misc
        save_test_results_path=None,
    ):
        super().__init__(optimizer_dict=optimizer_dict)
        self.save_hyperparameters()
        self.token_type = token_type
        self.log_step_size = log_step_size
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

        # load pretrained weights for backbone
        if path_weights:
            logger.info(f"Pretrained weights loaded from: {path_weights}")
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

        # define head
        self.head = SuryaHead(
            in_feature=in_feature,
            layer_type=head_type,
            layer_dict=head_layer_dict,
        )

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

        match self.token_type:
            case "cls_token":
                return tokens[:, 0, :]
            case "avg_pooling":
                return tokens.mean(dim=1)
            case "max_pooling":
                return tokens.max(dim=1).values
            case _:
                raise ValueError(f"Unknown token_type: {self.token_type}")

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
        x_hat = self.head(tokens)
        probs = torch.sigmoid(x_hat)

        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Update Metrics
        self.train_metrics.update(probs, target)

        # Step-Wise Logging Logic
        # Check if the current global step is a multiple of 100
        if (self.trainer.global_step + 1) % self.log_step_size == 0:
            # Compute, log, and reset the metrics accumulated over the last 100 steps
            metrics = self.train_metrics.compute()

            # Log the computed metrics
            self.log_dict(
                {f"train/step_{k}": v.float() for k, v in metrics.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

            self.train_metrics.reset()

        self.log_dict(
            {
                "perf/file_open_latency_sec": stats["open_time"].float().mean(),
                "perf/file_read_bandwidth_sec": stats["read_time"].float().mean(),
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
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        tokens = self.forward_features(data)
        x_hat = self.head(tokens)
        probs = torch.sigmoid(x_hat)

        # Update states
        self.val_metrics.update(probs, target)

        # Log step loss normally
        loss = F.binary_cross_entropy_with_logits(x_hat, target)
        self.log("val_loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        # Compute global metrics (Auto-synced across GPUs)
        metrics = self.val_metrics.compute()

        self.log_dict(
            {f"val_{k}": v.float() for k, v in metrics.items()}, sync_dist=True
        )

        # Reset for next epoch
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        tokens = self.forward_features(data)
        x_hat = self.head(tokens)
        probs = torch.sigmoid(x_hat)

        # Store predictions and targets for later analysis
        self.test_results["timestamps"].append(
            metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        )
        self.test_results["targets"].append(target.item())
        self.test_results["predictions"].append(probs.item())

        # Calculate Loss
        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(probs, target)

        # Log Test Loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        # Compute and log metrics
        # The compute_and_reset method handles distributed aggregation (all_reduce)
        metrics = self.test_metrics.compute()

        # Log all computed metrics with a 'test/' prefix
        self.log_dict(
            {f"test/{k}": v.float() for k, v in metrics.items()}, sync_dist=True
        )

        self.test_metrics.reset()

        # Gather across all ranks (list of lists)
        timestamps = torch.as_tensor(
            self.test_results["timestamps"],
            dtype=torch.float64,
            device=self.device,
        )
        all_timestamps = self.all_gather(timestamps)
        all_predictions = self.all_gather(self.test_results["predictions"])
        all_targets = self.all_gather(self.test_results["targets"])

        if self.trainer.is_global_zero:

            # Flatten the gathered lists (from multiple ranks) correctly
            all_timestamps = [
                ts.detach().cpu().tolist()
                for rank_ts in all_timestamps
                for ts in rank_ts
            ]

            all_predictions = [
                float(p.detach().cpu().item())
                for rank_p in all_predictions
                for p in rank_p
            ]

            all_targets = [
                int(y.detach().cpu().item()) for rank_y in all_targets for y in rank_y
            ]

            results = pd.DataFrame(
                {
                    "timestamps": all_timestamps,  # list of lists
                    "predictions": all_predictions,  # scalar per row
                    "targets": all_targets,  # scalar per row
                }
            )

            results = pd.DataFrame(results)
            results.to_csv(
                os.path.join(self.save_test_results_path, "surya_test_results.csv"),
                index=False,
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        t0 = time.perf_counter()

        # the standard move (CPU -> GPU)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        t1 = time.perf_counter()

        if self.trainer.is_global_zero and self.logger:
            try:
                self.logger.experiment.log(
                    {"perf/cpu_to_gpu_transfer_sec": t1 - t0},
                    commit=False,
                )
            except AttributeError:
                pass

        return batch


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
        log_step_size=100,
        # misc
        save_test_results_path=None,
    ):
        super().__init__(optimizer_dict=optimizer_dict)
        self.save_hyperparameters()
        self.log_step_size = log_step_size
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

        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Update Metrics
        self.train_metrics.update(probs, target)

        # Step-Wise Logging Logic
        # Check if the current global step is a multiple of 100
        if (self.trainer.global_step + 1) % self.log_step_size == 0:
            # Compute, log, and reset the metrics accumulated over the last 100 steps
            metrics = self.train_metrics.compute()

            # Log the computed metrics
            self.log_dict(
                {f"train/step_{k}": v.float() for k, v in metrics.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

            self.train_metrics.reset()

        self.log_dict(
            {
                "perf/file_open_latency_sec": stats["open_time"].float().mean(),
                "perf/file_read_bandwidth_sec": stats["read_time"].float().mean(),
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
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float()
        x_hat = self.backbone(data)
        probs = torch.sigmoid(x_hat)

        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Log Training Loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Update Metrics (x_hat contains the logits)
        self.val_metrics.update(probs, target)

        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = self.val_metrics.compute()

        # Log all computed metrics
        self.log_dict(
            {f"val_{k}": v.float() for k, v in metrics.items()}, sync_dist=True
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
        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(probs, target)

        # Log Test Loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        # Compute and log metrics
        # The compute_and_reset method handles distributed aggregation (all_reduce)
        metrics = self.test_metrics.compute()

        # Log all computed metrics with a 'test/' prefix
        self.log_dict(
            {f"test/{k}": v.float() for k, v in metrics.items()}, sync_dist=True
        )

        self.test_metrics.reset()

        # Gather across all ranks (list of lists)
        timestamps = torch.as_tensor(
            self.test_results["timestamps"],
            dtype=torch.float64,
            device=self.device,
        )
        all_timestamps = self.all_gather(timestamps)
        all_predictions = self.all_gather(self.test_results["predictions"])
        all_targets = self.all_gather(self.test_results["targets"])

        if self.trainer.is_global_zero:

            # Flatten the gathered lists (from multiple ranks) correctly
            all_timestamps = [
                ts.detach().cpu().tolist()
                for rank_ts in all_timestamps
                for ts in rank_ts
            ]

            all_predictions = [
                float(p.detach().cpu().item())
                for rank_p in all_predictions
                for p in rank_p
            ]

            all_targets = [
                int(y.detach().cpu().item()) for rank_y in all_targets for y in rank_y
            ]

            results = pd.DataFrame(
                {
                    "timestamps": all_timestamps,  # list of lists
                    "predictions": all_predictions,  # scalar per row
                    "targets": all_targets,  # scalar per row
                }
            )

            results = pd.DataFrame(results)
            results.to_csv(
                os.path.join(
                    self.save_test_results_path, f"{self.model_name}_test_results.csv"
                ),
                index=False,
            )

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            t0 = time.perf_counter()

            # the standard move (CPU -> GPU)
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

            t1 = time.perf_counter()

            if self.trainer.is_global_zero and self.logger:
                try:
                    self.logger.experiment.log(
                        {"perf/cpu_to_gpu_transfer_sec": t1 - t0},
                        commit=False,
                    )
                except AttributeError:
                    pass

            return batch
