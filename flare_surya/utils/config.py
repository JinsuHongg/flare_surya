from loguru import logger
from omegaconf import OmegaConf


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg