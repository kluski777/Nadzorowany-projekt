from pathlib import Path
import numpy as np


def load_latent_spaces(input_dir: str, split: str) -> np.ndarray:
    """Load and flatten latent spaces from .npz files in the specified directory."""
    full_path = Path(input_dir) / f"{split}.npz"
    data = np.load(full_path)
    latent_full = data["full"]

    return latent_full.reshape(latent_full.shape[0], -1)
