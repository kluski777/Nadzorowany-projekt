import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import IncrementalPCA
import joblib


class FeatureExtractor:
  def __init__(self, n_components=None):
    self.model = IncrementalPCA(n_components=n_components)

  def fit(self, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for latent_spaces in dataloader:
      self.model.partial_fit(latent_spaces)

  def transform(self, latent_spaces):
    reduced_latent_spaces = self.model.transform(latent_spaces)
    return torch.tensor(reduced_latent_spaces)

  def save(self):
    filename = os.path.join('out', 'feature-extractor.pkl')
    joblib.dump(self, filename, 3)

  @staticmethod
  def load(filename):
    return joblib.load(filename)
