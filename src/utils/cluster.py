from pathlib import Path
import numpy as np


def load_clusters(split: str, input_dir: str = "data/clusters") -> dict[str, np.ndarray]:
    full_path = Path(input_dir) / f"{split}.npz"
    return np.load(full_path)
