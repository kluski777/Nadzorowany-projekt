from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

from models import Clusterizer
from utils import load_latent_spaces


def fit_clusterizer(input_dir: str, output_dir: str, n_clusters: int):
    """
    Fit a Clusterizer model on latent components.
    
    Args:
        input_dir: Directory containing latent component files
        output_dir: Directory to save the clusterizer and scaler
        n_clusters: Number of clusters for K-means
    """
    # Load latent components from train split
    latent_components = load_latent_spaces(input_dir, "train")
    full_latent_components = latent_components["full"]  # Shape: (n_samples, n_components)
    
    # Fit scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaled_components = scaler.fit_transform(full_latent_components)
    
    # Fit clusterizer
    print(f"Fitting Clusterizer with {n_clusters} clusters...")
    clusterizer = Clusterizer(n_clusters=n_clusters)
    clusterizer.fit(scaled_components)
    
    # Save clusterizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clusterizer.save(output_dir=str(output_path), filename="clusterizer")
    
    # Save scaler
    scaler_path = output_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    print(f"Clusterizer saved to: {output_path / 'clusterizer.pkl'}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Inertia: {clusterizer.inertia_:.2f}")
