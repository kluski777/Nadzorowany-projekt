from models import Clusterizer
from utils import load_latent_components


def fit_clusterizer(input_dir: str, output_dir: str, n_clusters: int) -> None:
    latent_components = load_latent_components(input_dir, "train")

    cl = Clusterizer(n_clusters=n_clusters)
    cl.fit(latent_components)
    cl.save(output_dir=output_dir, filename="clusterizer")
