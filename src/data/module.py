from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms


class WikiArtStreamingDataset(IterableDataset):
    """Streaming Dataset wrapper for WikiArt that applies transforms and loads from disk."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __iter__(self):
        for item in self.dataset:
            image = item["image"]

            if self.transform:
                image = self.transform(image)

            yield {"image": image}


class WikiArtDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for WikiArt dataset with local disk caching."""

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        image_size: int = 256,
        total_samples: Optional[int] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        data_dir: str = "/kaggle/working/data",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.total_samples = total_samples
        self.val_split = val_split
        self.test_split = test_split
        self.data_dir = Path(data_dir)

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

    def setup(self, stage: Optional[str] = None):
        """Load dataset from local disk using streaming."""
        
        dataset = load_dataset(
            "Artificio/WikiArt_Full",
            cache_dir=str(self.data_dir),
            split="train",
            streaming=True
        )
        
        if self.total_samples is None:
            self.total_samples = 103_250

        self.test_size = int(self.total_samples * self.test_split)
        self.val_size = int(self.total_samples * self.val_split)
        self.train_size = self.total_samples - self.val_size - self.test_size

        train_hf = dataset.take(self.train_size)
        val_hf = dataset.skip(self.train_size).take(self.val_size)
        test_hf = dataset.skip(self.train_size + self.val_size).take(self.test_size)

        self.train_dataset = WikiArtStreamingDataset(train_hf, transform=self.transform)
        self.val_dataset = WikiArtStreamingDataset(val_hf, transform=self.transform)
        self.test_dataset = WikiArtStreamingDataset(test_hf, transform=self.transform)

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
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
