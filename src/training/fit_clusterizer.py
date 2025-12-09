from models import Clusterizer
from utils import load_latent_spaces


def fit_clusterizer(input_dir: str, output_dir: str, n_clusters: int) -> None:
    latent_components = load_latent_spaces(input_dir, "train")
    full_latent_components = latent_components["full"]  # Shape: (n_samples, n_components)

    cl = Clusterizer(n_clusters=n_clusters)
    cl.fit(full_latent_components)
    cl.save(output_dir=output_dir, filename="clusterizer")
