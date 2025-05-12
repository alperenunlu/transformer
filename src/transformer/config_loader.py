import os
from .hparams import load_hyperparameters_from_toml
from .presets import PRESET_CONFIGS


def get_hparams(config_name: str = "base"):
    if config_name.endswith(".toml"):
        if not os.path.exists(config_name):
            raise FileNotFoundError(f"TOML file not found: {config_name}")
        return load_hyperparameters_from_toml(config_name)

    if config_name in PRESET_CONFIGS:
        return PRESET_CONFIGS[config_name]()

    raise ValueError(f"Unknown configuration: '{config_name}'")
