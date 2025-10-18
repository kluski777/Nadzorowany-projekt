from pathlib import Path
from typing import Optional
import json
import random

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset, Dataset
from torchvision import transforms


class WikiArtStreamingDataset(IterableDataset):
    """Streaming Dataset wrapper for WikiArt training data with epoch-based shuffling."""

    def __init__(self, base_dataset, transform=None, shuffle_buffer_size=None, base_seed=42):
        self.base_dataset = base_dataset
        self.transform = transform
        self.shuffle_buffer_size = shuffle_buffer_size
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Set the current epoch for shuffling."""
        self.epoch = epoch

    def __iter__(self):
        dataset = self.base_dataset
        
        if self.shuffle_buffer_size:
            epoch_seed = self.base_seed + self.epoch
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=epoch_seed)
        
        for item in dataset:
            image = item["image"]
            if self.transform:
                image = self.transform(image)
            yield {"image": image}


class WikiArtStaticDataset(Dataset):
    """Static Dataset for val/test - loads data into memory for deterministic splits."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        
        if self.transform:
            image = self.transform(image)
        
        return {"image": image}


class WikiArtDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for WikiArt with deterministic val/test splits.
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        image_size: int = 256,
        total_samples: Optional[int] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        data_dir: str = "/kaggle/working/data",
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        splits_cache_file: str = "dataset_splits.json",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.total_samples = total_samples
        self.val_split = val_split
        self.test_split = test_split
        self.data_dir = Path(data_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.splits_cache_file = self.data_dir / splits_cache_file

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download dataset to local directory if not already downloaded."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = self.data_dir / "Artificio___WikiArt_Full"
        if not dataset_path.exists():
            print(f"Downloading WikiArt dataset to {self.data_dir}...")
            load_dataset(
                "Artificio/WikiArt_Full",
                cache_dir=str(self.data_dir),
                keep_in_memory=False,
            )
            print("Dataset download completed!")
        else:
            print(f"Dataset already exists at {dataset_path}")

    def _create_deterministic_splits(self):
        """
        Create deterministic val/test splits and cache them.
        This ensures the same images are used across different model runs.
        """
        print("Creating deterministic val/test splits...")
        
        # Load full dataset (non-streaming) with fixed seed
        full_dataset = load_dataset(
            "Artificio/WikiArt_Full",
            cache_dir=str(self.data_dir),
            split="train",
        )
        
        if self.total_samples is None:
            self.total_samples = len(full_dataset)
        
        # Use fixed random seed for splitting
        random.seed(self.seed)
        
        # Create shuffled indices
        indices = list(range(self.total_samples))
        random.shuffle(indices)
        
        # Calculate split sizes
        self.test_size = int(self.total_samples * self.test_split)
        self.val_size = int(self.total_samples * self.val_split)
        self.train_size = self.total_samples - self.val_size - self.test_size
        
        # Split indices
        train_indices = indices[:self.train_size]
        val_indices = indices[self.train_size:self.train_size + self.val_size]
        test_indices = indices[self.train_size + self.val_size:]
        
        # Save splits to cache
        splits_info = {
            "seed": self.seed,
            "total_samples": self.total_samples,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }
        
        with Path(self.splits_cache_file).open('w') as f:
            json.dump(splits_info, f, indent=2)
        
        print(f"Splits cached to {self.splits_cache_file}")
        print(f"Train: {self.train_size}, Val: {self.val_size}, Test: {self.test_size}")
        
        return full_dataset, train_indices, val_indices, test_indices

    def _load_cached_splits(self):
        """Load previously cached splits."""
        with Path(self.splits_cache_file).open('r') as f:
            splits_info = json.load(f)
        
        if splits_info["seed"] != self.seed:
            raise ValueError(
                f"Cached splits use seed {splits_info['seed']}, but current seed is {self.seed}. "
                "Delete the cache file to regenerate splits."
            )
        
        print(f"Loaded cached splits from {self.splits_cache_file}")
        print(f"Train: {splits_info['train_size']}, Val: {splits_info['val_size']}, Test: {splits_info['test_size']}")
        
        return splits_info

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with deterministic val/test splits."""
        
        if self.splits_cache_file.exists():
            splits_info = self._load_cached_splits()
            val_indices = splits_info["val_indices"]
            test_indices = splits_info["test_indices"]
            self.train_size = splits_info["train_size"]
            self.val_size = splits_info["val_size"]
            self.test_size = splits_info["test_size"]
            
            full_dataset = load_dataset(
                "Artificio/WikiArt_Full",
                cache_dir=str(self.data_dir),
                split="train",
            )
        else:
            full_dataset, _, val_indices, test_indices = self._create_deterministic_splits()
        
        val_data = [full_dataset[int(i)] for i in val_indices]
        test_data = [full_dataset[int(i)] for i in test_indices]
        
        self.val_dataset = WikiArtStaticDataset(val_data, transform=self.transform)
        self.test_dataset = WikiArtStaticDataset(test_data, transform=self.transform)
        
        train_streaming = load_dataset(
            "Artificio/WikiArt_Full",
            cache_dir=str(self.data_dir),
            split="train",
            streaming=True
        ).take(self.train_size)
        
        self.train_dataset = WikiArtStreamingDataset(
            train_streaming,
            transform=self.transform,
            shuffle_buffer_size=self.shuffle_buffer_size,
            base_seed=self.seed
        )
        
        print(f"\n{'='*60}")
        print(f"Dataset Setup Complete:")
        print(f"  Training:   {self.train_size} samples (streaming)")
        print(f"  Validation: {self.val_size} samples (static)")
        print(f"  Test:       {self.test_size} samples (static)")
        print(f"  Val/Test splits are DETERMINISTIC and cached")
        print(f"{'='*60}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,  # Deterministic order for validation
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,  # Deterministic order for testing
        )
