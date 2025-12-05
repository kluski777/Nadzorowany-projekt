from models import FeatureExtractor
from utils import load_latent_spaces


def fit_feature_extractor(input_dir: str, output_dir: str, n_components: int):
    latent_spaces = load_latent_spaces(input_dir, "train")

    fe = FeatureExtractor(n_components=n_components)
    fe.fit(latent_spaces)
    fe.save(output_dir=output_dir, filename="feature_extractor")

    print(f"Explained variance by {n_components} components: {fe.variance_percentage:.0f}%")
    fe.save_explained_variance_plot()
