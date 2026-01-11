from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import pytorch_lightning as pl
from datasets import Dataset, load_dataset
from tqdm import tqdm
import torch.nn as nn
import joblib

from models import get_autoencoder, FeatureExtractor, Clusterizer
from data.module import WikiArtDataModule
from utils import load_config
from utils.cutting import apply_cut_reproducible


def _get_cutting_seed(config: dict, cutting_seed: Optional[int] = None) -> int:
    """Determine cutting seed from config or use provided value."""
    if cutting_seed is not None:
        return cutting_seed

    seed = config.get("cutting", {}).get("seed")
    if seed is None:
        seed = config["experiment"]["seed"]

    return seed


def _setup_data_module(config: dict, seed: int, cutting_seed: int, batch_size: int) -> WikiArtDataModule:
    """Create and prepare data module."""
    data_module = WikiArtDataModule(
        batch_size=batch_size,
        num_workers=0,
        image_size=config["data"]["image_size"],
        data_dir=config["data"]["data_dir"],
        seed=seed,
        splits_dir=config["data"]["splits_dir"],
        enable_cutting=False,
        cutting_seed=cutting_seed,
    )
    data_module.prepare_data()
    return data_module


def _load_model(checkpoint_path: str, config: dict, device: torch.device) -> pl.LightningModule:
    """Load model from checkpoint and move to device."""
    model = get_autoencoder(config["model"]["architecture"]).load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    return model


def _process_single_image(
    image_idx: int,
    full_dataset: Dataset,
    data_module: WikiArtDataModule,
    model: pl.LightningModule,
    device: torch.device,
    cutting_seed: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Process a single image and return its latent spaces.

    Returns:
        Tuple of (image_index, latent_full, latent_cut)
    """
    # Load image from dataset
    image_item = full_dataset[image_idx]
    image = image_item["image"]

    # Apply transform
    image_tensor = data_module.transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Encode full image
    latent_full = model.encode(image_tensor)
    latent_full = latent_full.squeeze(0).cpu().numpy()

    # Apply cut and encode cut image
    image_cut = apply_cut_reproducible(image_tensor.squeeze(0).cpu(), cutting_seed + image_idx)
    image_cut = image_cut.unsqueeze(0).to(device)
    latent_cut = model.encode(image_cut)
    latent_cut = latent_cut.squeeze(0).cpu().numpy()

    return image_idx, latent_full, latent_cut


def _process_split(
    split_name: str,
    indices: List[int],
    full_dataset: Dataset,
    data_module: WikiArtDataModule,
    model: nn.Module,
    device: torch.device,
    cutting_seed: int,
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], int]:
    """
    Process all images in a split and collect their latent spaces.

    Returns:
        Tuple of (image_indices, latent_full_list, latent_cut_list, error_count)
    """
    image_indices = []
    latent_full_list = []
    latent_cut_list = []
    error_count = 0

    with torch.inference_mode():
        for idx in tqdm(indices, desc=f"Processing {split_name}"):
            try:
                image_idx, latent_full, latent_cut = _process_single_image(
                    idx, full_dataset, data_module, model, device, cutting_seed
                )
                image_indices.append(image_idx)
                latent_full_list.append(latent_full)
                latent_cut_list.append(latent_cut)
            except Exception as e:
                print(f"\nError processing image {idx} in {split_name}: {e}")
                error_count += 1
                continue

    return image_indices, latent_full_list, latent_cut_list, error_count


def _compute_clusters(
    latent_full_list: List[np.ndarray],
    feature_extractor: FeatureExtractor,
    clusterizer: Clusterizer,
    scaler,
) -> np.ndarray:
    """Compute cluster assignments for latent spaces using feature extractor and clusterizer."""
    # Stack latent spaces: [num_images, latent_channels, 8, 8]
    full_array = np.stack(latent_full_list, axis=0)
    # Flatten for feature extraction: [num_images, latent_channels * 8 * 8]
    full_flat = full_array.reshape(full_array.shape[0], -1)
    # Extract features (PCA)
    latent_components = feature_extractor.transform(full_flat)
    # Scale features
    latent_components_scaled = scaler.transform(latent_components)
    # Predict clusters
    clusters = clusterizer.predict(latent_components_scaled)
    return clusters


def _save_split_latent_spaces(
    split_name: str,
    image_indices: List[int],
    latent_full_list: List[np.ndarray],
    latent_cut_list: List[np.ndarray],
    output_path: Path,
    clusters: Optional[np.ndarray] = None,
) -> None:
    """Save latent spaces for a split as a compressed numpy file."""
    print(f"Saving {len(image_indices)} latent spaces to {split_name}.npz...")

    # Stack arrays: [num_images, latent_channels, 8, 8]
    indices_array = np.array(image_indices, dtype=np.int64)
    target_latent_array = np.stack(latent_full_list, axis=0)
    masked_latent_array = np.stack(latent_cut_list, axis=0)

    output_file = output_path / f"{split_name}.npz"
    
    # Build save dict - clusters are optional
    save_dict = {
        "indices": indices_array,
        "target_latent": target_latent_array,
        "masked_latent": masked_latent_array,
    }
    if clusters is not None:
        save_dict["cluster"] = clusters
    
    np.savez_compressed(output_file, **save_dict)
    print(f"Saved {output_file} ({output_file.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Shape - target_latent: {target_latent_array.shape}, masked_latent: {masked_latent_array.shape}")
    if clusters is not None:
        print(f"  Clusters: {len(np.unique(clusters))} unique clusters")
    else:
        print("  Clusters: not included (no clusterizer provided)")


def generate_latent_spaces(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = "data/latent_spaces",
    cutting_seed: Optional[int] = None,
    batch_size: int = 1,
    feature_extractor_checkpoint: Optional[str] = None,
    clusterizer_checkpoint: Optional[str] = None,
):
    """
    Generate latent spaces for images in dataset splits.
    
    Args:
        config_path: Path to configuration YAML file
        checkpoint_path: Path to autoencoder checkpoint
        output_dir: Output directory for latent spaces
        cutting_seed: Seed for cutting operations
        batch_size: Batch size for processing
        feature_extractor_checkpoint: Optional path to feature extractor for cluster assignment
        clusterizer_checkpoint: Optional path to clusterizer for cluster assignment
        
    If feature_extractor_checkpoint and clusterizer_checkpoint are provided,
    cluster labels will be computed and saved. Otherwise, latent spaces are
    saved without cluster information.
    """
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)

    cutting_seed = _get_cutting_seed(config, cutting_seed)
    print(f"Using cutting seed: {cutting_seed}")

    data_module = _setup_data_module(config, seed, cutting_seed, batch_size)

    print("Loading splits...")
    _, train_indices, val_indices, test_indices = data_module._load_splits_from_csv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = _load_model(checkpoint_path, config, device)

    # Load feature extractor and clusterizer for cluster assignment (optional)
    enable_clustering = feature_extractor_checkpoint is not None and clusterizer_checkpoint is not None
    feature_extractor = None
    clusterizer = None
    scaler = None
    
    if enable_clustering:
        print(f"Loading feature extractor from: {feature_extractor_checkpoint}")
        feature_extractor = FeatureExtractor.load(feature_extractor_checkpoint)
        print(f"Loading clusterizer from: {clusterizer_checkpoint}")
        clusterizer = Clusterizer.load(clusterizer_checkpoint)
        # Load scaler (saved alongside clusterizer)
        scaler_path = Path(clusterizer_checkpoint).parent / "scaler.pkl"
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("No feature extractor/clusterizer provided - generating latent spaces without cluster labels")

    print("Loading full dataset...")
    full_dataset = load_dataset(
        "Artificio/WikiArt_Full",
        cache_dir=str(data_module.data_dir),
        split="train",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    total_processed = 0
    total_errors = 0

    for split_name, indices in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split ({len(indices)} images)")
        print(f"{'=' * 60}")

        image_indices, latent_full_list, latent_cut_list, split_errors = _process_split(
            split_name, indices, full_dataset, data_module, model, device, cutting_seed
        )

        # Compute cluster assignments if clustering is enabled
        clusters = None
        if enable_clustering:
            print("Computing cluster assignments...")
            clusters = _compute_clusters(latent_full_list, feature_extractor, clusterizer, scaler)

        _save_split_latent_spaces(
            split_name, image_indices, latent_full_list, latent_cut_list, output_path, clusters
        )

        total_processed += len(image_indices)
        total_errors += split_errors
        print(f"\n{split_name} split complete:")
        print(f"  Processed: {len(image_indices)}")
        print(f"  Errors: {split_errors}")

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Total processed: {total_processed}")
    print(f"  Total errors: {total_errors}")
    print(f"  Output directory: {output_path}")
    if enable_clustering:
        print("  Cluster labels: included")
    else:
        print("  Cluster labels: not included")
    print(f"{'=' * 60}\n")
