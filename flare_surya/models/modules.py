import os
import time
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import (
    AlexNetClassifier,
    MobileNetClassifier,
    ResNet18Classifier,
    ResNet34Classifier,
    ResNet50Classifier,
)
from terratorch_surya.models.helio_spectformer import HelioSpectFormer

from flare_surya.metrics.classification_metrics import DistributedClassificationMetrics
from .base import BaseModule
from .heads import SuryaHead
from .baselines_models import ResNet18
from .criterions import BinaryFocalLoss, FlareSSMLoss, get_criterion
from .secondary_modality_models import SecondaryEncoder
from .fusion import FusionModule


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
            {f"train/step_{k}": v.float() for k, v in metrics.items()},
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
            "val_loss",
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
            {f"val_{k}": v.float() for k, v in metrics.items()}, sync_dist=True
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

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v.float():.4f}")
        print("===================\n")
        self.log_dict(
            {f"test/{k}": v.float() for k, v in metrics.items()}, sync_dist=False
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
            {f"train/step_{k}": v.float() for k, v in metrics.items()},
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
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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

        secondary_tokens = self.secondary_encoder(data["xrs"])

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
        if isinstance(self.criterion, FlareSSMLoss):
            loss = self.criterion(
                x_hat, target, output["hidden"], current_epoch=self.current_epoch
            )
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
        )

        return loss

    def on_train_epoch_end(self):
        # Compute, log, and reset the metrics accumulated over the last 100 steps
        metrics = self.train_metrics.compute()

        # Log the computed metrics
        self.log_dict(
            {f"train/step_{k}": v.float() for k, v in metrics.items()},
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
            "val_loss",
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
            {f"val_{k}": v.float() for k, v in metrics.items()}, sync_dist=True
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
        self.test_results = {"timestamps": [], "predictions": [], "targets": []}
