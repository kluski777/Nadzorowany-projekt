from pathlib import Path
import numpy as np


def load_clusters(split: str, input_dir: str = "data/clusters") -> np.ndarray:
    """Load clusters from .npy files in the specified directory."""
    full_path = Path(input_dir) / f"{split}.npy"
    clusters = np.load(full_path)

    return clusters
