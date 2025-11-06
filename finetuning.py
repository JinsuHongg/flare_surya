import yaml
from terratorch.tasks import ClassificationTask
from terratorch_surya.downstream_examples.solar_flare_forecasting.models import HelioSpectformer1D
from lightning.pytorch import Trainer
from .datamodule import FlareDataModule

if __name__ == "__main__":
    # Configs
    config_path = "./configs/first_experiement_model_comparison.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Datamodule
    datamodule = FlareDataModule(
        config_path=config_path
    )

    # Model
    model_args = dict(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=config["model"]["in_channels"],
        embed_dim=config["model"]["embed_dim"],
        time_embedding=config["model"]["time_embedding"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        dtype=config["dtype"],
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=config["model"]["learned_flow"],
        use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
        init_weights=config["model"]["init_weights"],
        checkpoint_layers=config["model"]["checkpoint_layers"],
        n_spectral_blocks=config["model"]["spectral_blocks"],
        rpe=config["model"]["rpe"],
        ensemble=config["model"]["ensemble"],
        finetune=config["model"]["finetune"],
        nglo=config["model"]["nglo"],
        # Put finetuning additions below this line
        dropout=config["model"]["dropout"],
        num_penultimate_transformer_layers=0,
        num_penultimate_heads=0,
        num_outputs=1,
        config=config,
    )

    # Task
    task = ClassificationTask(
        model_args=model_args,
        model=HelioSpectformer1D,
        lr=config["optimizer"]["learning_rate"],
        optimizer=config["optimizer"]["type"],
        scheduler=config["scheduler"]["type"],
        freeze_backbone=config["etc"]["freeze_backbone"],
        class_names=config["data"]["class_names"],
        path_to_record_metrics=config["etc"]["save_score_path"],
    )

    trainer = Trainer(
        accelerator=config["etc"]["accelerator"],
        devices=config["etc"]["devices"],
        max_epochs=config["optimizer"]["max_epoch"],
    )

    trainer.fit(model=task, datamodule=datamodule)
    # trainer.test(dataloaders=datamodule)