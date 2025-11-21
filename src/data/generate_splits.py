from pathlib import Path
from typing import Optional
import json
import random

from datasets import load_dataset


def generate_splits(
    seed: int,
    total_samples: Optional[int],
    val_split: float,
    test_split: float,
    data_dir: str,
    splits_dir: str = "splits",
):
    """
    Generate reproducible train/val/test splits and save them as CSV files.

    Args:
        seed: Random seed for reproducible split generation
        total_samples: Total number of samples to use (None for full dataset)
        val_split: Fraction of data for validation (e.g., 0.1 for 10%)
        test_split: Fraction of data for testing (e.g., 0.1 for 10%)
        data_dir: Base directory for dataset
        splits_dir: Directory name relative to data_dir where splits will be saved
    """
    data_dir_path = Path(data_dir)
    splits_path = data_dir_path / splits_dir

    # Ensure data directory exists
    data_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating splits with seed {seed}...")
    print(f"Splits will be saved to: {splits_path}")

    # Load full dataset (non-streaming) with fixed seed
    print("Loading dataset...")
    full_dataset = load_dataset(
        "Artificio/WikiArt_Full",
        cache_dir=str(data_dir_path),
        split="train",
    )

    if total_samples is None:
        total_samples = len(full_dataset)

    print(
        f"Using {total_samples} samples from dataset (total available: {len(full_dataset)})"
    )

    # Use fixed random seed for splitting
    random.seed(seed)

    # Create shuffled indices
    indices = list(range(total_samples))
    random.shuffle(indices)

    # Calculate split sizes
    test_size = int(total_samples * test_split)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size - test_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create splits directory
    splits_path.mkdir(parents=True, exist_ok=True)

    # Save splits to CSV files (one index per line)
    train_csv_path = splits_path / "train.csv"
    val_csv_path = splits_path / "val.csv"
    test_csv_path = splits_path / "test.csv"

    print(f"Writing splits to CSV files...")
    with train_csv_path.open("w") as f:
        for idx in train_indices:
            f.write(f"{idx}\n")

    with val_csv_path.open("w") as f:
        for idx in val_indices:
            f.write(f"{idx}\n")

    with test_csv_path.open("w") as f:
        for idx in test_indices:
            f.write(f"{idx}\n")

    # Save metadata
    metadata = {
        "seed": seed,
        "total_samples": total_samples,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }

    metadata_path = splits_path / "splits_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Split generation complete!")
    print(f"  Train: {train_size} samples -> {train_csv_path}")
    print(f"  Val:   {val_size} samples -> {val_csv_path}")
    print(f"  Test:  {test_size} samples -> {test_csv_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"{'=' * 60}\n")
