import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
import joblib


class Clusterizer:
    def __init__(self, n_clusters=8, max_iter=100, tol=0.0, patience=10):
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            max_no_improvement=patience
        )

    def fit(self, dataset, batch_size=32):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for batch in dataloader:
            self.model.partial_fit(batch)

    def predict(self, latent_spaces):
        return self.model.predict(latent_spaces)

    def save(self, filename):
        filename = os.path.join('out', f'{filename}.pkl')
        joblib.dump(self, filename, 3)

    @staticmethod
    def load(filename):
        return joblib.load(filename)