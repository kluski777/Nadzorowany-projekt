import numpy as np
from pathlib import Path

from models import FeatureExtractor
from utils import load_latent_spaces


def _load_feature_extractor(checkpoint_path: str) -> FeatureExtractor:
    return FeatureExtractor.load(checkpoint_path)


def _save_latent_components(latent_components: np.ndarray, output_path: Path, split_name: str):
    print(f"Saving {len(latent_components)} latent components to {split_name}.npy...")

    file_path = output_path / f"{split_name}.npy"
    np.save(file_path, latent_components)

    print(f"Latent components of {split_name} split have been saved to {file_path}")


def generate_latent_components(input_dir: str, checkpoint_path: str, splits: list[str], output_dir: str):
    fe = _load_feature_extractor(checkpoint_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        latent_spaces = load_latent_spaces(input_dir, split)
        latent_components = fe.transform(latent_spaces)
        _save_latent_components(latent_components, output_path, split)
