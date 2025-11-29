import os
from loguru import logger

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import HelioSpectformer1D
from terratorch_surya.models.helio_spectformer import HelioSpectFormer
from flare_surya.models.heads import SuryaHead
from flare_surya.models.base import BaseModule
from terratorch_surya.downstream_examples.solar_flare_forecasting.metrics import DistributedClassificationMetrics


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
            weight_path,
            # head parameters
            token_type,
            in_feature,
            head_type,
            head_layer_dict,
            freeze_backbone,
            optimizer_dict,
            threshold=0.5,
            log_step_size=100,
            ):
        super().__init__(
            optimizer_dict=optimizer_dict
            )
        self.token_type = token_type
        self.log_step_size = log_step_size

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
        if weight_path:
            logger.info(f"Pretrained weights loaded: {weight_path}")
            self.backbone.load_state_dict(weight_path, strict=True)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        tokens = self.backbone(data)
        match self.token_type:
            case "cls_token":
                tokens = tokens[:, 0, :]
            case "avg_pooling":
                tokens = tokens.mean(dim=1)
            case "max_pooling":
                tokens = tokens.max(dim=1).values

        x_hat = self.head(tokens)
        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # 2. Update Metrics
        self.train_metrics.update(torch.sigmoid(x_hat), data["label"])

        # 3. Step-Wise Logging Logic
        # Check if the current global step is a multiple of 100
        if (self.trainer.global_step + 1) % self.log_step_size == 0:
            # Compute, log, and reset the metrics accumulated over the last 100 steps
            metrics = self.train_metrics.compute_and_reset()
            
            # Log the computed metrics
            self.log_dict(
                {f"train/step_{k}": v for k, v in metrics.items()}, 
                on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Log training loss every step
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, metadata = batch
        target = data["label"].float().unsqueeze(1)
        
        # 1. Forward Pass
        tokens = self.backbone(data)
        # Apply pooling logic
        match self.token_type:
            case "cls_token":
                tokens = tokens[:, 0, :]
            case "avg_pooling":
                tokens = tokens.mean(dim=1)
            case "max_pooling":
                tokens = tokens.max(dim=1).values
        x_hat = self.head(tokens)
        
        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # 2. Log Training Loss
        self.log("val_loss", loss, prog_bar=True)
        
        # 3. Update Metrics (x_hat contains the logits)
        self.val_metrics.update(torch.sigmoid(x_hat), data["label"])
        
        return loss
    
    def on_validation_epoch_end(self):
        # 1. Compute and log metrics
        metrics = self.val_metrics.compute_and_reset()
        
        # 2. Log all computed metrics
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, sync_dist=True)
        
        # You can also log a single key metric for checkpointing
        self.log("prog_bar/val_f1", metrics["f1"], prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # 1. Prepare Data
        data, metadata = batch
        # Ensure target is float for loss calculation
        target = data["label"].float().unsqueeze(1) 
        
        # 2. Forward Pass
        tokens = self.backbone(data)
        
        # Apply Pooling (Matching logic from training/validation)
        match self.token_type:
            case "cls_token":
                tokens = tokens[:, 0, :]
            case "avg_pooling":
                tokens = tokens.mean(dim=1)
            case "max_pooling":
                tokens = tokens.max(dim=1).values

        x_hat = self.head(tokens)
        
        # 3. Calculate Loss
        loss = F.binary_cross_entropy_with_logits(x_hat, target)

        # 4. Update Metrics
        # Pass predicted probabilities (sigmoid(x_hat)) and the target (squeezed to [B])
        # Note: data["label"] is the original integer/long tensor before conversion
        self.test_metrics.update(torch.sigmoid(x_hat), data["label"])
        
        # 5. Log Test Loss
        # Logging on_step=False, on_epoch=True ensures averaging across the test set
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True) 

        return loss

    def on_test_epoch_end(self):
        # 1. Compute and log metrics
        # The compute_and_reset method handles distributed aggregation (all_reduce)
        metrics = self.test_metrics.compute_and_reset()
        
        # 2. Log all computed metrics with a 'test/' prefix
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)
        
        # Optional: Log the primary test metric (e.g., F1 or TSS) prominently
        self.log("test/f1_final", metrics["f1"], prog_bar=True, sync_dist=True)
        self.log("test/tss_final", metrics["tss"], prog_bar=True, sync_dist=True)