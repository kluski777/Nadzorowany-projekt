import torch
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from models import Clusterizer
from .latent import load_latent_spaces
from .cluster import load_clusters
from .device import get_device


def visualize_inpainter(
    inpainter,
    data_module,
    num_samples: int = 8,
    output_path: str = "reconstruction_results.png",
):
    inpainter.eval()
    val_loader = data_module.val_dataloader()
    
    # Collect samples with masked regions
    valid_targets, valid_masked, valid_images, valid_masks = [], [], [], []
    
    for batch in val_loader:
        mask = batch["mask"]
        valid_idx = [(mask[i] == 0).sum() > 0 for i in range(len(mask))]
        
        valid_targets.extend(batch["target_latent"][valid_idx])
        valid_masked.extend(batch["masked_latent"][valid_idx])
        valid_images.extend(batch["image"][valid_idx])
        valid_masks.extend(batch["mask"][valid_idx])
        
        if len(valid_targets) >= num_samples:
            break
    
    target_latent = torch.stack(valid_targets[:num_samples])
    to_transform = torch.stack(valid_masked[:num_samples]).to('cpu')
    images = torch.stack(valid_images[:num_samples]).cpu()
    masks = torch.stack(valid_masks[:num_samples]).cpu()
    
    corrupted = images * masks.unsqueeze(1)
    
    inpainter = inpainter.to('cpu')
    with torch.no_grad():
        inpainted_latent = inpainter(to_transform)
        reconstructed = inpainter.autoencoder.decode(inpainted_latent).cpu()
        target_imgs = inpainter.autoencoder.decode(target_latent).cpu()
    
    _, axes = plt.subplots(3, num_samples, figsize=(20, 8))
    for i in range(num_samples):
        display_data = [
            (target_imgs[i], "Target (AE)"),
            (corrupted[i], "Corrupted (Cut)"),
            (reconstructed[i], "Reconstructed")
        ]
        for row, (img, title) in enumerate(display_data):
            axes[row, i].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_results(
    model,
    data_module,
    num_samples: int = 8,
    output_path: str = "reconstruction_results.png",
):
    model.eval()

    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    images = batch["image"][:num_samples]
    targets = batch.get("target", images)[:num_samples]

    device = get_device()
    images = images.to(device)
    model = model.to(device)

    with torch.inference_mode():
        reconstructed = model(images)
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

    images = images.cpu()
    targets = targets.cpu()
    reconstructed = reconstructed.cpu()

    _, axes = plt.subplots(3, num_samples, figsize=(20, 8))

    for i in range(num_samples):
        display_data = [
            (targets[i], "Target"),
            (images[i], "Input"),
            (reconstructed[i], "Reconstructed")
        ]
        for row, (img, title) in enumerate(display_data):
            axes[row, i].imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def visualize_umap(latent_components_input_dir: str, clusters_input_dir: str, output_dir: str = "data/plots"):
    train_latent_components = load_latent_spaces(latent_components_input_dir, "train")["full"]
    val_latent_components = load_latent_spaces(latent_components_input_dir, "val")["full"]
    test_latent_components = load_latent_spaces(latent_components_input_dir, "test")["full"]
    latent_components = np.vstack((train_latent_components, val_latent_components, test_latent_components))

    train_clusters = load_clusters("train", clusters_input_dir)["full"]
    val_clusters = load_clusters("val", clusters_input_dir)["full"]
    test_clusters = load_clusters("test", clusters_input_dir)["full"]
    clusters = np.concatenate((train_clusters, val_clusters, test_clusters))

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_components)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, edgecolors="k", cmap="tab20")
    plt.title("UMAP Visualization", fontsize=14, fontweight="bold")
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / "umap-visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"UMAP visualization saved to {output_path}")

    plt.show()


def visualize_elbow_plot(
    input_dir: str,
    min_clusters: int = 2,
    max_clusters: int = 20,
    output_dir: str = "data/plots",
):
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
