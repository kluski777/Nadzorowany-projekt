import os
from pathlib import Path
import numpy as np


def load_latent_spaces(input_dir: str, splits: list[str]) -> np.ndarray:
    """Load and flatten latent spaces from .npz files in the specified directory."""
    latent_spaces = []

    input_path = Path(input_dir)
    for file_name in os.listdir(input_path):
        name, _ = os.path.splitext(file_name)

        if file_name.endswith(".npz") and name in splits:
            data = np.load(input_path / file_name)
            latent_full = data["full"]
            latent_spaces.append(latent_full.reshape(latent_full.shape[0], -1))

    latent_spaces = np.vstack(latent_spaces)
    return latent_spaces
