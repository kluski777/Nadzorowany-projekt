import os
import joblib
from sklearn.decomposition import PCA
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class FeatureExtractor:
    """
    A wrapper class for sklearn's PCA to perform feature extraction on latent spaces.

    This class provides methods to fit, transform, and save/load PCA models for dimensionality reduction.
    """

    def __init__(self, n_components: int = None):
        """
        Initialize the FeatureExtractor with a PCA model.

        Args:
            n_components (int or None): Number of principal components to keep. If None, all components are kept.
        """
        self.model = PCA(n_components=n_components)

    def fit(self, latent_spaces: ndarray) -> None:
        """
        Fit the PCA model to the latent spaces.

        Args:
            latent_spaces (array-like): The input data to fit the PCA model on.
        """
        self.model.fit(latent_spaces)

    def transform(self, latent_spaces: ndarray) -> ndarray:
        """
        Transform the latent spaces using the fitted PCA model.

        Args:
            latent_spaces (array-like): The input data to transform.

        Returns:
            array-like: The transformed data in the reduced dimensional space.
        """
        return self.model.transform(latent_spaces)

    def fit_transform(self, latent_spaces: ndarray) -> ndarray:
        """
        Fit the PCA model and transform the latent spaces in one step.

        Args:
            latent_spaces (array-like): The input data to fit and transform.

        Returns:
            array-like: The transformed data in the reduced dimensional space.
        """
        return self.model.fit_transform(latent_spaces)

    def save(self, output_dir: str = "data/models", filename: str = "feature_extractor") -> None:
        """
        Save the FeatureExtractor instance to a file using joblib.

        Args:
            output_dir (str): Directory to save the file. Defaults to 'data/models'.
            filename (str): Name of the file without extension. Defaults to 'feature_extractor'.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        full_path = os.path.join(output_dir, f"{filename}.pkl")
        joblib.dump(self, full_path, 3)

    def save_explained_variance_plot(self, output_dir: str = "data/plots", filename: str = "explained-variance"):
        """
        Display a bar chart of the explained variance ratio of each principal component.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(1, len(self.model.explained_variance_ratio_) + 1), np.cumsum(self.model.explained_variance_ratio_)
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Explained Variance by Principal Components (Cumulative)")

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f"{filename}.png")
        plt.close()

    @property
    def variance_percentage(self) -> float:
        """
        Get the percentage of variance by the selected components.

        Returns:
            float: Percentage of variance explained by the selected components.
        """
        return np.sum(self.model.explained_variance_ratio_) * 100

    @staticmethod
    def load(checkpoint_path: str = "data/models/feature_extractor.pkl") -> "FeatureExtractor":
        """
        Load a FeatureExtractor instance from a file.

        Args:
            checkpoint_path (str): Full path to the checkpoint file. Defaults to 'data/models/feature_extractor.pkl'.

        Returns:
            FeatureExtractor: The loaded FeatureExtractor instance.
        """
        return joblib.load(checkpoint_path)
