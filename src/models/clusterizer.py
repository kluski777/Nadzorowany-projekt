import os
import joblib
from sklearn.cluster import KMeans


class Clusterizer:
    """
    A wrapper class for sklearn's KMeans to perform clustering on latent spaces.

    This class provides methods to fit, predict, and save/load KMeans models for unsupervised clustering.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=0):
        """
        Initialize the Clusterizer with a KMeans model.

        Args:
            n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Defaults to 8.
            max_iter (int): Maximum number of iterations of the k-means algorithm for a single run. Defaults to 300.
            tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. Defaults to 1e-4.
            verbose (int): Verbosity mode. Defaults to 0.
        """
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, verbose=verbose)

    def fit(self, latent_components):
        """
        Fit the KMeans model to the latent components.

        Args:
            latent_components (array-like): The input data to fit the KMeans model on.
        """
        self.model.fit(latent_components)

    def predict(self, latent_components):
        """
        Predict the closest cluster each sample in latent_components belongs to.

        Args:
            latent_components (array-like): The input data to predict clusters for.

        Returns:
            array-like: Index of the cluster each sample belongs to.
        """
        return self.model.predict(latent_components)

    def fit_predict(self, latent_components):
        """
        Fit the KMeans model and predict the closest cluster each sample in latent_components belongs to in one step.

        Args:
            latent_components (array-like): The input data to fit and predict clusters for.

        Returns:
            array-like: Index of the cluster each sample belongs to.
        """
        return self.model.fit_predict(latent_components)

    def save(self, output_dir="data/models", filename="clusterizer"):
        """
        Save the Clusterizer instance to a file using joblib.

        Args:
            output_dir (str): Directory to save the file. Defaults to 'data/models'.
            filename (str): Name of the file without extension. Defaults to 'clusterizer'.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        full_path = os.path.join(output_dir, f"{filename}.pkl")
        joblib.dump(self, full_path, 3)

    @staticmethod
    def load(checkpoint_path: str = "data/models/clusterizer.pkl"):
        """
        Load a Clusterizer instance from a file.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pkl). Defaults to 'data/models/clusterizer.pkl'.

        Returns:
            Clusterizer: The loaded Clusterizer instance.
        """
        return joblib.load(checkpoint_path)
