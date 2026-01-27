from pathlib import Path
from typing import Optional
import json

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from utils.cutting import apply_cut, apply_cut_reproducible


class WikiArtDataset(IterableDataset):
    def __init__(
        self,
        base_dataset,
        transform=None,
        base_seed=42,
        shuffle_per_epoch=True,
        enable_cutting=False,
        cutting_mode="random",
        cutting_seed=42,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.base_seed = base_seed
        self.shuffle_per_epoch = shuffle_per_epoch
        self.epoch = 0
        self.enable_cutting = enable_cutting
        self.cutting_mode = cutting_mode
        self.cutting_seed = cutting_seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        dataset = self.base_dataset

        if self.shuffle_per_epoch:
            epoch_seed = self.base_seed + self.epoch
            dataset = dataset.shuffle(seed=epoch_seed)

        sample_index = 0
        for item in dataset:
            image = item["image"]
            if self.transform:
                image = self.transform(image)

            target = image
            if self.enable_cutting:
                if self.cutting_mode == "reproducible":
                    seed = self.cutting_seed + sample_index
                    image = apply_cut_reproducible(image, seed)
                else:
                    image = apply_cut(image)

            yield {"image": image, "target": target}
            sample_index += 1


class WikiArtDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        image_size: int = 256,
        data_dir: str = "/kaggle/working/data",
        seed: int = 42,
        splits_dir: str = "splits",
        enable_cutting: bool = False,
        cutting_mode_train: str = "random",
        cutting_mode_val: str = "reproducible",
        cutting_mode_test: str = "reproducible",
        cutting_seed: int = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.splits_dir = self.data_dir / splits_dir
        self.enable_cutting = enable_cutting
        self.cutting_mode_train = cutting_mode_train
        self.cutting_mode_val = cutting_mode_val
        self.cutting_mode_test = cutting_mode_test
        self.cutting_seed = cutting_seed if cutting_seed is not None else seed

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
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
        if not self.splits_dir.exists():
            raise ValueError(
                f"Splits directory does not exist: {self.splits_dir}\n"
                "Please generate splits first using: python main.py generate_splits"
            )

        train_csv = self.splits_dir / "train.csv"
        val_csv = self.splits_dir / "val.csv"
        test_csv = self.splits_dir / "test.csv"
        metadata_json = self.splits_dir / "splits_metadata.json"

        if not all(
            [
                train_csv.exists(),
                val_csv.exists(),
                test_csv.exists(),
                metadata_json.exists(),
            ]
        ):
            missing = [f for f in [train_csv, val_csv, test_csv, metadata_json] if not f.exists()]
            raise ValueError(
                f"Missing split files in {self.splits_dir}: {[f.name for f in missing]}\n"
                "Please generate splits first using: python main.py generate_splits"
            )

        with metadata_json.open("r") as f:
            metadata = json.load(f)

        if metadata["seed"] != self.seed:
            raise ValueError(
                f"Splits were generated with seed {metadata['seed']}, but current seed is {self.seed}. "
                "Either regenerate splits with the correct seed or use the matching seed for training."
            )

        train_indices = []
        with train_csv.open("r") as f:
            for line in f:
                train_indices.append(int(line.strip()))

        val_indices = []
        with val_csv.open("r") as f:
            for line in f:
                val_indices.append(int(line.strip()))

        test_indices = []
        with test_csv.open("r") as f:
            for line in f:
                test_indices.append(int(line.strip()))

        print(f"Loaded splits from {self.splits_dir}")
        print(f"Train: {metadata['train_size']}, Val: {metadata['val_size']}, Test: {metadata['test_size']}")

        return metadata, train_indices, val_indices, test_indices

    def setup(self, stage: Optional[str] = None):
        metadata, _, _, _ = self._load_splits_from_csv()

        self.train_size = metadata["train_size"]
        self.val_size = metadata["val_size"]
        self.test_size = metadata["test_size"]

        full_dataset = load_dataset(
            "Artificio/WikiArt_Full",
            cache_dir=str(self.data_dir),
            split="train",
        )

        train_subset = full_dataset.select(range(self.train_size))
        self.train_dataset = WikiArtDataset(
            train_subset,
            transform=self.transform,
            base_seed=self.seed,
            shuffle_per_epoch=True,
            enable_cutting=self.enable_cutting,
            cutting_mode=self.cutting_mode_train,
            cutting_seed=self.cutting_seed,
        )

        val_subset = full_dataset.select(range(self.train_size, self.train_size + self.val_size))
        self.val_dataset = WikiArtDataset(
            val_subset,
            transform=self.transform,
            base_seed=self.seed,
            shuffle_per_epoch=False,
            enable_cutting=self.enable_cutting,
            cutting_mode=self.cutting_mode_val,
            cutting_seed=self.cutting_seed,
        )

        test_subset = full_dataset.select(range(self.train_size + self.val_size, self.train_size + self.val_size + self.test_size))
        self.test_dataset = WikiArtDataset(
            test_subset,
            transform=self.transform,
            base_seed=self.seed,
            shuffle_per_epoch=False,
            enable_cutting=self.enable_cutting,
            cutting_mode=self.cutting_mode_test,
            cutting_seed=self.cutting_seed,
        )

        print(f"\n{'=' * 60}")
        print(f"Dataset Setup Complete:")
        print(f"  Training:   {self.train_size} samples")
        print(f"  Validation: {self.val_size} samples")
        print(f"  Test:       {self.test_size} samples")
        print(f"{'=' * 60}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )
