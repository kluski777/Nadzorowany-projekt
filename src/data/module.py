from pathlib import Path
from typing import Optional
import json

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
        data_dir: str = "/kaggle/working/data",
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        splits_dir: str = "splits",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.splits_dir = self.data_dir / splits_dir

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

    def _load_splits_from_csv(self):
        """Load pre-generated splits from CSV files."""
        if not self.splits_dir.exists():
            raise ValueError(
                f"Splits directory does not exist: {self.splits_dir}\n"
                "Please generate splits first using: python main.py generate_splits"
            )
        
        train_csv = self.splits_dir / "train.csv"
        val_csv = self.splits_dir / "val.csv"
        test_csv = self.splits_dir / "test.csv"
        metadata_json = self.splits_dir / "splits_metadata.json"
        
        if not all([train_csv.exists(), val_csv.exists(), test_csv.exists(), metadata_json.exists()]):
            missing = [f for f in [train_csv, val_csv, test_csv, metadata_json] if not f.exists()]
            raise ValueError(
                f"Missing split files in {self.splits_dir}: {[f.name for f in missing]}\n"
                "Please generate splits first using: python main.py generate_splits"
            )
        
        # Load metadata
        with metadata_json.open('r') as f:
            metadata = json.load(f)
        
        # Validate seed
        if metadata["seed"] != self.seed:
            raise ValueError(
                f"Splits were generated with seed {metadata['seed']}, but current seed is {self.seed}. "
                "Either regenerate splits with the correct seed or use the matching seed for training."
            )
        
        # Load indices from CSV files
        train_indices = []
        with train_csv.open('r') as f:
            for line in f:
                train_indices.append(int(line.strip()))
        
        val_indices = []
        with val_csv.open('r') as f:
            for line in f:
                val_indices.append(int(line.strip()))
        
        test_indices = []
        with test_csv.open('r') as f:
            for line in f:
                test_indices.append(int(line.strip()))
        
        print(f"Loaded splits from {self.splits_dir}")
        print(f"Train: {metadata['train_size']}, Val: {metadata['val_size']}, Test: {metadata['test_size']}")
        
        return metadata, train_indices, val_indices, test_indices

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with pre-generated splits from CSV files."""
        
        # Load splits from CSV files
        metadata, train_indices, val_indices, test_indices = self._load_splits_from_csv()
        
        self.train_size = metadata["train_size"]
        self.val_size = metadata["val_size"]
        self.test_size = metadata["test_size"]
        
        # Load full dataset to access specific indices
        full_dataset = load_dataset(
            "Artificio/WikiArt_Full",
            cache_dir=str(self.data_dir),
            split="train",
        )
        
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
        print(f"  Val/Test splits loaded from pre-generated CSV files")
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
