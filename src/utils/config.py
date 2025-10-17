import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r") as f:
        return yaml.safe_load(f)
