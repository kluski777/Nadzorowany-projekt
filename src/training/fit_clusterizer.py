import joblib
from sklearn.preprocessing import MinMaxScaler

from models import Clusterizer
from utils import load_latent_spaces


def fit_clusterizer(input_dir: str, output_dir: str, n_clusters: int) -> None:
    latent_components = load_latent_spaces(input_dir, "train")
    full_latent_components = latent_components["full"]  # Shape: (n_samples, n_components)

    print("Fitting scaler and transforming latent components...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    full_latent_components = scaler.fit_transform(full_latent_components)
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    print(f"Scaler has been saved to {output_dir}/scaler.pkl")

    print("Fitting clusterizer...")

    cl = Clusterizer(n_clusters=n_clusters)
    cl.fit(full_latent_components)
    cl.save(output_dir=output_dir, filename="clusterizer")

    print(f"Clusterizer with {n_clusters} clusters has been saved to {output_dir}/clusterizer.pkl")
