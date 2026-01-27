from models import FeatureExtractor
from utils import load_latent_spaces


def fit_feature_extractor(input_dir: str, output_dir: str, n_components: int):
    latent_spaces = load_latent_spaces(input_dir, "train")
    
    if "target_latent" in latent_spaces.files:
        full_latent_spaces = latent_spaces["target_latent"]
    elif "full" in latent_spaces.files:
        full_latent_spaces = latent_spaces["full"]
    else:
        raise ValueError(
            "Latent space file must contain either 'target_latent' or 'full' field. "
            f"Available fields: {list(latent_spaces.files)}"
        )
    
    full_latent_spaces = full_latent_spaces.reshape(full_latent_spaces.shape[0], -1)

    print(f"Fitting FeatureExtractor with {n_components} components...")
    fe = FeatureExtractor(n_components=n_components)
    fe.fit(full_latent_spaces)
    fe.save(output_dir=output_dir, filename="feature_extractor")

    print(f"Explained variance by {n_components} components: {fe.variance_percentage:.2f}%")
    fe.save_explained_variance_plot()