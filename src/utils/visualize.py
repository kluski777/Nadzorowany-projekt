import torch
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from models import Clusterizer
from .latent import load_latent_spaces
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
    train_latent_components = load_latent_spaces(input_dir, "train")["full"]
    val_latent_components = load_latent_spaces(input_dir, "val")["full"]
    test_latent_components = load_latent_spaces(input_dir, "test")["full"]
    latent_components = np.vstack((train_latent_components, val_latent_components, test_latent_components))

    train_clusters = load_clusters("train")["full"]
    val_clusters = load_clusters("val")["full"]
    test_clusters = load_clusters("test")["full"]
    clusters = np.concatenate((train_clusters, val_clusters, test_clusters))

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_components)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, edgecolors="k", cmap="tab10")
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()


def visualize_elbow_plot(
    input_dir: str,
    min_clusters: int = 2,
    max_clusters: int = 20,
    output_dir: str = "data/plots",
):
    """
    Visualize elbow plot to determine optimal number of clusters.

    Args:
        input_dir: Directory containing latent spaces
        min_clusters: Minimum number of clusters to test
        max_clusters: Maximum number of clusters to test
        output_path: Path to save the elbow plot
    """
    print("Loading training latent components...")
    latent_components = load_latent_spaces(input_dir, "train")
    full_latent_components = latent_components["full"]

    print("Fitting scaler and transforming latent components...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    full_latent_components = scaler.fit_transform(full_latent_components)

    print(f"Computing inertia for {min_clusters} to {max_clusters} clusters...")
    inertias = []
    k_range = range(min_clusters, max_clusters + 1)

    for k in k_range:
        print(f"  Testing k={k}...")
        clusterizer = Clusterizer(n_clusters=k, random_state=42)
        clusterizer.fit(full_latent_components)
        inertias.append(clusterizer.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), inertias, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    plt.title("Elbow Method for Optimal k", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(list(k_range))
    plt.tight_layout()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"elbow-plot-{max_clusters}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Elbow plot saved to {output_path}")
    plt.show()
