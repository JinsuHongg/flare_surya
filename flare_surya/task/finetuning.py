import argparse

from loguru import logger as lgr_logger

import torch
from lightning.pytorch import Trainer

from flare_surya.datamodule import FlareDataModule
from flare_surya.models.modules import FlareSurya
from flare_surya.utils.config import load_config
from flare_surya.utils.logging import build_wandb
from flare_surya.utils.callbacks import build_callbacks


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
        # Put finetuning additions below this line
        token_type=config["head"]["token_type"],
        in_feature=config["head"]["hyper_parameters"]["in_feature"][config["head"]["token_type"]],
        head_type=config["head"]["type"],
        head_layer_dict=config["head"]["hyper_parameters"],
        freeze_backbone=config["backbone"]["freeze_backbone"],
        optimizer_dict=config["optimizer"]["hyper_parameters"],
        threshold=config["head"]["threshold"],
        log_step_size=config["head"]["log_step_size"]
    )


def train(config_path):

    # load config
    config = load_config(config_path)

    # Datamodule
    datamodule = FlareDataModule(
        config_path=config_path
    )

    # Load model
    model = build_model(config=config)

    # Create wandb obejct
    wandb_logger = build_wandb(cfg=config, model=model)

    # Trainer
    callbacks = build_callbacks(cfg=config, wandb_logger=wandb_logger)
    trainer = Trainer(
        accelerator=config["etc"]["accelerator"],
        devices=config["etc"]["devices"],
        max_epochs=config["etc"]["max_epochs"],
        precision=config["etc"]["precision"],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=config["etc"]["log_every_n_steps"],
        limit_train_batches=config["etc"]["limit_train_batches"],
        limit_val_batches=config["etc"]["limit_val_batches"],
        strategy=config["etc"]["strategy"]
    )

    lgr_logger.info(f"Start training...")
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(dataloaders=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Surya-Flare-finetuning',
                    description='Finetuning Surya with flare dataset')
    parser.add_argument('--config-path', default="../configs/first_experiement_model_comparison.yaml")  
    args = parser.parse_args()
    train(config_path=args.config_path)
