import torch
import matplotlib.pyplot as plt
import numpy as np
import umap

from .latent_space import load_latent_spaces


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

    # Removing a mask if it exists
    if images.shape[1] == 4: 
        images = images[:, 1:, :, :]
        reconstructed = reconstructed[:, 1:, :, :]

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


def visualize_umap(input_dir: str, c: list[int] | None = None):
    """Visualize UMAP embeddings of latent spaces."""
    latent_spaces = load_latent_spaces(input_dir, splits=["train", "val", "test"])

    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(latent_spaces)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=c,
        edgecolors="k",
    )
    plt.title("UMAP Visualization of Latent Spaces")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()
