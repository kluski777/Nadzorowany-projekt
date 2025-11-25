import os
import joblib
from sklearn.decomposition import PCA


class FeatureExtractor:
    """
    A wrapper class for sklearn's PCA to perform feature extraction on latent spaces.

    This class provides methods to fit, transform, and save/load PCA models for dimensionality reduction.
    """

    def __init__(self, n_components=None):
        """
        Initialize the FeatureExtractor with a PCA model.

        Args:
            n_components (int or None): Number of principal components to keep. If None, all components are kept.
        """
        self.model = PCA(n_components=n_components)

    def fit(self, latent_spaces):
        """
        Fit the PCA model to the latent spaces.

        Args:
            latent_spaces (array-like): The input data to fit the PCA model on.
        """
        self.model.fit(latent_spaces)

    def transform(self, latent_spaces):
        """
        Transform the latent spaces using the fitted PCA model.

        Args:
            latent_spaces (array-like): The input data to transform.

        Returns:
            array-like: The transformed data in the reduced dimensional space.
        """
        return self.model.transform(latent_spaces)
  
    def fit_transform(self, latent_spaces):
        """
        Fit the PCA model and transform the latent spaces in one step.

        Args:
            latent_spaces (array-like): The input data to fit and transform.

        Returns:
            array-like: The transformed data in the reduced dimensional space.
        """
        return self.model.fit_transform(latent_spaces)

    def save(self, output_dir='data/models', filename='feature-extractor'):
        """
        Save the FeatureExtractor instance to a file using joblib.

        Args:
            output_dir (str): Directory to save the file. Defaults to 'data/models'.
            filename (str): Name of the file without extension. Defaults to 'feature-extractor'.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        full_path = os.path.join(output_dir, f'{filename}.pkl')
        joblib.dump(self, full_path, 3)

    @staticmethod
    def load(input_dir='data/models', filename='feature-extractor'):
        """
        Load a FeatureExtractor instance from a file.

        Args:
            input_dir (str): Directory where the file is located. Defaults to 'data/models'.
            filename (str): Name of the file without extension. Defaults to 'feature-extractor'.

        Returns:
            FeatureExtractor: The loaded FeatureExtractor instance.
        """
        full_path = os.path.join(input_dir, f'{filename}.pkl')
        return joblib.load(full_path)
