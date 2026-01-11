import os
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger as lgr_logger

import torch
from lightning.pytorch import Trainer

from flare_surya.datamodule import FlareDataModule
from flare_surya.models.modules import FlareSurya

# from flare_surya.utils.config import load_config
from flare_surya.utils.logger_utils import build_wandb
from flare_surya.utils.callbacks import build_callbacks

torch.set_float32_matmul_precision("medium")


def build_model(config):

    return FlareSurya(
        img_size=config["backbone"]["img_size"],
        patch_size=config["backbone"]["patch_size"],
        in_chans=len(config["data"]["channels"]),
        embed_dim=config["backbone"]["embed_dim"],
        time_embedding=config["backbone"]["time_embedding"],
        depth=config["backbone"]["depth"],
        num_heads=config["backbone"]["num_heads"],
        mlp_ratio=config["backbone"]["mlp_ratio"],
        drop_rate=config["backbone"]["drop_rate"],
        dtype=torch.bfloat16,
        window_size=config["backbone"]["window_size"],
        dp_rank=config["backbone"]["dp_rank"],
        learned_flow=config["backbone"]["learned_flow"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        init_weights=config["backbone"]["init_weights"],
        checkpoint_layers=config["backbone"]["checkpoint_layers"],
        n_spectral_blocks=config["backbone"]["n_spectral_blocks"],
        rpe=config["backbone"]["rpe"],
        ensemble=config["backbone"]["ensemble"],
        finetune=config["backbone"]["finetune"],
        nglo=config["backbone"]["nglo"],
        path_weights=config["backbone"]["path_weights"],
        # Put finetuning additions below this line
        token_type=config["head"]["token_type"],
        in_feature=config["head"]["hyper_parameters"]["in_feature"][
            config["head"]["token_type"]
        ],
        head_type=config["head"]["type"],
        head_layer_dict=config["head"]["hyper_parameters"],
        freeze_backbone=config["backbone"]["freeze_backbone"],
        lora_dict=config["lora"],
        optimizer_dict=config["optimizer"],
        threshold=config["head"]["threshold"],
        log_step_size=config["head"]["log_step_size"],
        save_test_results_path=config["etc"]["save_test_results_path"],
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="first_experiement_model_comparison.yaml",
)
def train(cfg: OmegaConf):

    # Datamodule
    datamodule = FlareDataModule(cfg=cfg)

    # Load model
    if cfg["pretrained_downstream_model_path"]:
        model = FlareSurya.load_from_checkpoint(cfg["pretrained_downstream_model_path"])
    else:
        model = build_model(config=cfg)

    # Create wandb obejct
    wandb_logger = build_wandb(cfg=cfg, model=model)

    # Trainer
    callbacks = build_callbacks(cfg=cfg, wandb_logger=wandb_logger)
    trainer = Trainer(
        accelerator=cfg["etc"]["accelerator"],
        devices=cfg["etc"]["devices"],
        num_nodes=cfg["etc"]["num_nodes"],
        max_epochs=cfg["etc"]["max_epochs"],
        precision=cfg["etc"]["precision"],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg["etc"]["log_every_n_steps"],
        limit_train_batches=cfg["etc"]["limit_train_batches"],
        limit_val_batches=cfg["etc"]["limit_val_batches"],
        strategy=cfg["etc"]["strategy"],
    )

    lgr_logger.info(f"Start training...")
    # trainer.fit(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=os.path.join(
    #         cfg.etc.ckpt_dir,
    #         cfg.etc.ckpt_file
    #     ) if cfg.etc.resume else None,
    # )
    trainer.test(
        model=model,
        dataloaders=datamodule,
        ckpt_path=os.path.join(cfg.etc.ckpt_dir, cfg.etc.ckpt_file),
        verbose=True,
    )


if __name__ == "__main__":

    train()
