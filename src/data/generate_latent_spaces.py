from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import pytorch_lightning as pl
from datasets import Dataset, load_dataset
from tqdm import tqdm
import joblib

from models import get_autoencoder, FeatureExtractor, Clusterizer
from data.module import WikiArtDataModule
from utils import load_config, load_latent_spaces
from utils.cutting import apply_cut_reproducible
from utils.device import get_device

def _get_cutting_seed(config: dict, cutting_seed: Optional[int] = None) -> int:
    if cutting_seed is not None:
        return cutting_seed
    return config.get("cutting", {}).get("seed") or config["experiment"]["seed"]


def _setup_data_module(config: dict, seed: int, cutting_seed: int, batch_size: int) -> WikiArtDataModule:
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
    model = get_autoencoder(config["model"]["architecture"]).load_from_checkpoint(checkpoint_path)
    model.eval()
    return model.to(device)


def _load_clustering_models(
    feature_extractor_path: str,
    clusterizer_path: str,
) -> Tuple[FeatureExtractor, Clusterizer, object]:
    print(f"Loading feature extractor from: {feature_extractor_path}")
    feature_extractor = FeatureExtractor.load(feature_extractor_path)
    
    print(f"Loading clusterizer from: {clusterizer_path}")
    clusterizer = Clusterizer.load(clusterizer_path)
    
    scaler_path = Path(clusterizer_path).parent / "scaler.pkl"
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    return feature_extractor, clusterizer, scaler


def _compute_clusters(
    latent_array: np.ndarray,
    feature_extractor: FeatureExtractor,
    clusterizer: Clusterizer,
    scaler,
) -> np.ndarray:
    flat = latent_array.reshape(latent_array.shape[0], -1)
    components = feature_extractor.transform(flat)
    scaled = scaler.transform(components)
    return clusterizer.predict(scaled)


def _process_split(
    split_name: str,
    indices: List[int],
    full_dataset: Dataset,
    data_module: WikiArtDataModule,
    model: pl.LightningModule,
    device: torch.device,
    cutting_seed: int,
) -> Tuple[List[int], np.ndarray, np.ndarray, int]:
    image_indices = []
    latent_full_list = []
    latent_cut_list = []
    error_count = 0

    with torch.inference_mode():
        for idx in tqdm(indices, desc=f"Processing {split_name}"):
            try:
                image = full_dataset[idx]["image"]
                image_tensor = data_module.transform(image).unsqueeze(0).to(device)
                
                latent_full = model.encode(image_tensor).squeeze(0).cpu().numpy()
                
                image_cut = apply_cut_reproducible(image_tensor.squeeze(0).cpu(), cutting_seed + idx)
                latent_cut = model.encode(image_cut.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                
                image_indices.append(idx)
                latent_full_list.append(latent_full)
                latent_cut_list.append(latent_cut)
            except Exception as e:
                print(f"\nError processing image {idx} in {split_name}: {e}")
                error_count += 1

    return (
        image_indices,
        np.stack(latent_full_list, axis=0),
        np.stack(latent_cut_list, axis=0),
        error_count,
    )


def _save_latent_spaces(
    output_file: Path,
    indices: np.ndarray,
    target_latent: np.ndarray,
    masked_latent: np.ndarray,
    clusters: Optional[np.ndarray] = None,
) -> None:
    save_dict = {
        "indices": indices,
        "target_latent": target_latent,
        "masked_latent": masked_latent,
    }
    if clusters is not None:
        save_dict["cluster"] = clusters

    np.savez_compressed(output_file, **save_dict)
    
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"Saved {output_file} ({size_mb:.2f} MB)")
    print(f"  Shape: target_latent={target_latent.shape}, masked_latent={masked_latent.shape}")
    if clusters is not None:
        print(f"  Clusters: {len(np.unique(clusters))} unique")


def _add_clusters_to_existing_files(
    output_path: Path,
    feature_extractor: FeatureExtractor,
    clusterizer: Clusterizer,
    scaler,
) -> int:
    total_processed = 0
    
    for split_name in ["train", "val", "test"]:
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split")
        print(f"{'=' * 60}")
        
        split_file = output_path / f"{split_name}.npz"
        data = load_latent_spaces(str(output_path), split_name)
        
        target_latent = data["target_latent"]
        print(f"  Loaded {len(target_latent)} latent spaces, shape: {target_latent.shape}")
        
        print("  Computing cluster assignments...")
        clusters = _compute_clusters(target_latent, feature_extractor, clusterizer, scaler)
        
        save_dict = {key: data[key] for key in data.files}
        save_dict["cluster"] = clusters
        np.savez_compressed(split_file, **save_dict)
        
        size_mb = split_file.stat().st_size / 1024 / 1024
        print(f"  Saved ({size_mb:.2f} MB), {len(np.unique(clusters))} unique clusters")
        total_processed += len(target_latent)
    
    return total_processed


def generate_latent_spaces(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = "data/latent_spaces",
    cutting_seed: Optional[int] = None,
    batch_size: int = 1,
    feature_extractor_checkpoint: Optional[str] = None,
    clusterizer_checkpoint: Optional[str] = None,
):
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    seed = config["experiment"]["seed"]
    pl.seed_everything(seed, workers=True)
    
    cutting_seed = _get_cutting_seed(config, cutting_seed)
    print(f"Using cutting seed: {cutting_seed}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    enable_clustering = feature_extractor_checkpoint is not None and clusterizer_checkpoint is not None
    
    if enable_clustering:
        all_splits_exist = all((output_path / f"{s}.npz").exists() for s in ["train", "val", "test"])
        if all_splits_exist:
            print("Found existing latent space files. Adding cluster assignments...")
            feature_extractor, clusterizer, scaler = _load_clustering_models(
                feature_extractor_checkpoint, clusterizer_checkpoint
            )
            total = _add_clusters_to_existing_files(output_path, feature_extractor, clusterizer, scaler)
            print(f"\n{'=' * 60}")
            print(f"Cluster assignment complete! Total processed: {total}")
            print(f"{'=' * 60}\n")
            return
    
    data_module = _setup_data_module(config, seed, cutting_seed, batch_size)
    
    print("Loading splits...")
    _, train_indices, val_indices, test_indices = data_module._load_splits_from_csv()
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = _load_model(checkpoint_path, config, device)
    
    feature_extractor, clusterizer, scaler = None, None, None
    if enable_clustering:
        feature_extractor, clusterizer, scaler = _load_clustering_models(
            feature_extractor_checkpoint, clusterizer_checkpoint
        )
    else:
        print("No feature extractor/clusterizer provided - generating without cluster labels")
    
    print("Loading full dataset...")
    full_dataset = load_dataset(
        "Artificio/WikiArt_Full",
        cache_dir=str(data_module.data_dir),
        split="train",
    )
    
    total_processed = 0
    total_errors = 0
    
    splits = [("train", train_indices), ("val", val_indices), ("test", test_indices)]
    
    for split_name, indices in splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split ({len(indices)} images)")
        print(f"{'=' * 60}")
        
        image_indices, target_latent, masked_latent, errors = _process_split(
            split_name, indices, full_dataset, data_module, model, device, cutting_seed
        )
        
        clusters = None
        if enable_clustering:
            print("Computing cluster assignments...")
            clusters = _compute_clusters(target_latent, feature_extractor, clusterizer, scaler)
        
        _save_latent_spaces(
            output_path / f"{split_name}.npz",
            np.array(image_indices, dtype=np.int64),
            target_latent,
            masked_latent,
            clusters,
        )
        
        total_processed += len(image_indices)
        total_errors += errors
        print(f"  Completed: {len(image_indices)} processed, {errors} errors")
    
    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Total processed: {total_processed}, errors: {total_errors}")
    print(f"  Output: {output_path}")
    print(f"  Clusters: {'included' if enable_clustering else 'not included'}")
    print(f"{'=' * 60}\n")
