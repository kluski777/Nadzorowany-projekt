import numpy as np
from pathlib import Path

from models import Clusterizer
from utils import load_latent_components


def _load_clusterizer(checkpoint_path: str) -> Clusterizer:
    return Clusterizer.load(checkpoint_path)


def _save_clusters(clusters: np.array, output_path: Path, split_name: str):
    print(f"Saving {len(clusters)} clusters to {split_name}.npy...")

    file_path = output_path / f"{split_name}.npy"
    np.save(file_path, clusters)

    print(f"Clusters of {split_name} split have been saved to {file_path}")


def generate_clusters(input_dir: str, checkpoint_path: str, splits: list[str], output_dir: str):
    clt = _load_clusterizer(checkpoint_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        latent_spaces = load_latent_components(input_dir, split)
        latent_components = clt.predict(latent_spaces)
        _save_clusters(latent_components, output_path, split)
