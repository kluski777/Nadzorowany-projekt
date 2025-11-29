from models.pca import FeatureExtractor
from utils.latent_space import load_latent_spaces

def fit_feature_extractor(input_dir: str, output_dir: str, n_components: int):
    latent_spaces = load_latent_spaces(input_dir, splits=['train'])

    fe = FeatureExtractor(n_components=n_components)
    fe.fit(latent_spaces)
    fe.save(output_dir=output_dir, filename='feature-extractor')
