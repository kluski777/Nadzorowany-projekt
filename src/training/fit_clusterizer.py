from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

from models import Clusterizer
from utils import load_latent_spaces


def fit_clusterizer(input_dir: str, output_dir: str, n_clusters: int):
    latent_components = load_latent_spaces(input_dir, "train")
    full_latent_components = latent_components["full"]
    
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaled_components = scaler.fit_transform(full_latent_components)
    
    print(f"Fitting Clusterizer with {n_clusters} clusters...")
    clusterizer = Clusterizer(n_clusters=n_clusters)
    clusterizer.fit(scaled_components)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clusterizer.save(output_dir=str(output_path), filename="clusterizer")
    
    scaler_path = output_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    print(f"Clusterizer saved to: {output_path / 'clusterizer.pkl'}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Inertia: {clusterizer.inertia_:.2f}")
