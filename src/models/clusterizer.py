import os
import joblib
from sklearn.cluster import KMeans


class Clusterizer:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, verbose=0):
        self.model = KMeans(
            n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )

    def fit(self, latent_components):
        self.model.fit(latent_components)

    def predict(self, latent_components):
        return self.model.predict(latent_components)

    def fit_predict(self, latent_components):
        return self.model.fit_predict(latent_components)

    def save(self, output_dir="data/models", filename="clusterizer"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        full_path = os.path.join(output_dir, f"{filename}.pkl")
        joblib.dump(self, full_path, 3)

    @property
    def inertia_(self):
        return self.model.inertia_

    @staticmethod
    def load(checkpoint_path: str = "data/models/clusterizer.pkl"):
        return joblib.load(checkpoint_path)
