from pathlib import Path
import numpy as np


def load_latent_spaces(input_dir: str, split: str) -> dict[str, np.ndarray]:
    full_path = Path(input_dir) / f"{split}.npz"
    return np.load(full_path)
