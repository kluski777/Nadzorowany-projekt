import numpy as np
from pathlib import Path
import joblib

from models import Clusterizer
from utils import load_latent_spaces


def generate_clusters(input_dir: str, checkpoint_path: str, splits: list[str], output_dir: str):
    clusterizer = Clusterizer.load(checkpoint_path)
    scaler = joblib.load(f"{Path(checkpoint_path).parent}/scaler.pkl")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        latent_components = load_latent_spaces(input_dir, split)

        full_latent_components = latent_components["full"]
        full_latent_components = scaler.transform(full_latent_components)
        full_clusters = clusterizer.predict(full_latent_components)

        cut_latent_components = latent_components["cut"]
        cut_latent_components = scaler.transform(cut_latent_components)
        cut_clusters = clusterizer.predict(cut_latent_components)

        indices = latent_components["indices"]

        file_path = output_path / f"{split}.npz"
        np.savez_compressed(
            file_path,
            indices=indices,
            full=full_clusters,
            cut=cut_clusters,
        )
