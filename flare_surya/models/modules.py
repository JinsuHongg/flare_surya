import os
import pandas as pd
from loguru import logger

import torch

# from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# import lightning as L
# from terratorch_surya.downstream_examples.solar_flare_forecasting.models import HelioSpectformer1D
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import (
    AlexNetClassifier,
    MobileNetClassifier,
    ResNet18Classifier,
    ResNet34Classifier,
    ResNet50Classifier,
)

from terratorch_surya.models.helio_spectformer import HelioSpectFormer
from flare_surya.models.heads import SuryaHead
from flare_surya.models.base import BaseModule
from terratorch_surya.downstream_examples.solar_flare_forecasting.metrics import (
    DistributedClassificationMetrics,
)


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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        tokens = self.forward_features(data)
        x_hat = self.head(tokens)

        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Update Metrics
        self.train_metrics.update(torch.sigmoid(x_hat), data["label"])

        # Step-Wise Logging Logic
        # Check if the current global step is a multiple of 100
        if (self.trainer.global_step + 1) % self.log_step_size == 0:
            # Compute, log, and reset the metrics accumulated over the last 100 steps
            metrics = self.train_metrics.compute_and_reset()

            # Log the computed metrics
            self.log_dict(
                {f"train/step_{k}": v for k, v in metrics.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
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
        self.val_metrics.update(torch.sigmoid(x_hat), data["label"])

        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = self.val_metrics.compute_and_reset()

        # Log all computed metrics
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, sync_dist=True)

        # You can also log a single key metric for checkpointing
        self.log("prog_bar/val_f1", metrics["f1"], prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"]
        tokens = self.forward_features(data)
        x_hat = self.head(tokens)

        # Store predictions and targets for later analysis
        self.test_results["timestamps"].append(
            metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        )
        self.test_results["targets"].append(target.item())
        self.test_results["predictions"].append(torch.sigmoid(x_hat).item())

        # Calculate Loss
        loss = F.binary_cross_entropy_with_logits(x_hat, target.float().unsqueeze(1))

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(torch.sigmoid(x_hat), data["label"])

        # Log Test Loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        # Compute and log metrics
        # The compute_and_reset method handles distributed aggregation (all_reduce)
        metrics = self.test_metrics.compute_and_reset()

        # Log all computed metrics with a 'test/' prefix
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)

        # Optional: Log the primary test metric (e.g., F1 or TSS) prominently
        self.log("test/f1_final", metrics["f1"], prog_bar=True, sync_dist=True)
        self.log("test/tss_final", metrics["tss"], prog_bar=True, sync_dist=True)

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
                os.path.join(self.save_test_results_path, "test_results.csv"),
                index=False,
            )


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
        target = data["label"].float()
        x_hat = self.backbone(data)

        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # Update Metrics
        self.train_metrics.update(torch.sigmoid(x_hat), data["label"])

        # Step-Wise Logging Logic
        # Check if the current global step is a multiple of 100
        if (self.trainer.global_step + 1) % self.log_step_size == 0:
            # Compute, log, and reset the metrics accumulated over the last 100 steps
            metrics = self.train_metrics.compute_and_reset()

            # Log the computed metrics
            self.log_dict(
                {f"train/step_{k}": v for k, v in metrics.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
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
        self.val_metrics.update(torch.sigmoid(x_hat), data["label"])

        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = self.val_metrics.compute_and_reset()

        # Log all computed metrics
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, sync_dist=True)

        # You can also log a single key metric for checkpointing
        self.log("prog_bar/val_f1", metrics["f1"], prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"]
        x_hat = self.backbone(data)

        # Store predictions and targets for later analysis
        self.test_results["timestamps"].append(
            metadata["timestamps_input"][0].detach().cpu().numpy().tolist()
        )
        self.test_results["targets"].append(target.item())
        self.test_results["predictions"].append(torch.sigmoid(x_hat).item())

        # Calculate Loss
        loss = F.binary_cross_entropy_with_logits(x_hat, target.float().unsqueeze(1))

        # Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        self.test_metrics.update(torch.sigmoid(x_hat), data["label"])

        # Log Test Loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        # Compute and log metrics
        # The compute_and_reset method handles distributed aggregation (all_reduce)
        metrics = self.test_metrics.compute_and_reset()

        # Log all computed metrics with a 'test/' prefix
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)

        # Optional: Log the primary test metric (e.g., F1 or TSS) prominently
        self.log("test/f1_final", metrics["f1"], prog_bar=True, sync_dist=True)
        self.log("test/tss_final", metrics["tss"], prog_bar=True, sync_dist=True)

        # Save test results to csf
        results = pd.DataFrame(self.test_results)
        results.to_csv(
            os.path.join(self.save_test_results_path, "test_results.csv"), index=False
        )
