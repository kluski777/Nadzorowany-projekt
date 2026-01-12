import numpy as np
from pathlib import Path

from models import FeatureExtractor
from utils import load_latent_spaces


def generate_latent_components(input_dir: str, checkpoint_path: str, splits: list[str], output_dir: str):
    fe = FeatureExtractor.load(checkpoint_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in splits:
        latent_spaces = load_latent_spaces(input_dir, split)

        if "target_latent" in latent_spaces.files:
            full_latent_spaces = latent_spaces["target_latent"]
            cut_latent_spaces = latent_spaces["masked_latent"]
        elif "full" in latent_spaces.files:
            full_latent_spaces = latent_spaces["full"]
            cut_latent_spaces = latent_spaces["cut"]
        else:
            raise ValueError(
                f"Latent space file must contain either ('target_latent', 'masked_latent') "
                f"or ('full', 'cut') fields. Available fields: {list(latent_spaces.files)}"
            )

        full_latent_spaces = full_latent_spaces.reshape(full_latent_spaces.shape[0], -1)
        full_latent_components = fe.transform(full_latent_spaces)

        cut_latent_spaces = cut_latent_spaces.reshape(cut_latent_spaces.shape[0], -1)
        cut_latent_components = fe.transform(cut_latent_spaces)

        indices = latent_spaces["indices"]

        file_path = output_path / f"{split}.npz"
        np.savez_compressed(
            file_path,
            indices=indices,
            full=full_latent_components,
            cut=cut_latent_components,
        )
