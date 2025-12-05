import torch
import matplotlib.pyplot as plt
import numpy as np
import umap

from .latent import load_latent_components
from .cluster import load_clusters


def visualize_results(
    model,
    data_module,
    num_samples: int = 8,
    output_path: str = "reconstruction_results.png",
):
    """Visualize original vs reconstructed images and save to a file."""
    model.eval()

    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    images = batch["image"][:num_samples]

    device = next(model.parameters()).device
    images = images.to(device)

    with torch.inference_mode():
        reconstructed = model(images)
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

    images = images.cpu()
    reconstructed = reconstructed.cpu()

    _, axes = plt.subplots(2, num_samples, figsize=(20, 5))

    for i in range(num_samples):
        # Original image
        img_orig = images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12)

        # Reconstructed image
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(np.clip(img_recon, 0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_umap(input_dir: str):
    """Visualize UMAP embeddings."""
    train_latent_spaces = load_latent_components(input_dir, "train")
    val_latent_spaces = load_latent_components(input_dir, "val")
    test_latent_spaces = load_latent_components(input_dir, "test")
    latent_spaces = np.vstack((train_latent_spaces, val_latent_spaces, test_latent_spaces))

    train_clusters = load_clusters("train")
    val_clusters = load_clusters("val")
    test_clusters = load_clusters("test")
    clusters = np.concat((train_clusters, val_clusters, test_clusters))

    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(latent_spaces)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, edgecolors="k", cmap="tab10")
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()
