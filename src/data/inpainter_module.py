from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
# Chemy masek to jest wazne bo musimy wczytac oryginalne zdjecia i maski.


class LatentInpainterDataset(Dataset):
    def __init__(
        self,
        latent_dir: str,
        split: str,
        cluster_id: Optional[int] = None,
    ):
        self.latent_dir = Path(latent_dir)
        self.split = split
        self.cluster_id = cluster_id
        
        self.npz_path = self.latent_dir / f"{split}.npz"
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Latent space file not found: {self.npz_path}")
        
        # Open with mmap_mode='r' to avoid loading everything into RAM
        # Note: requires the npz to be saved without compression (np.savez instead of np.savez_compressed)
        self.data = np.load(self.npz_path, mmap_mode='r')

        self.latent_masked = self.data["masked_latent"]
        self.latent_target = self.data["target_latent"]
        self.images = self.data["images"]
        self.masks = self.data["masks"]
        
        # Filtering indices
        if cluster_id is not None and "cluster" in self.data.files:
            clusters = self.data["cluster"]
            self.valid_indices = np.where(clusters == cluster_id)[0]
        else:
            self.valid_indices = np.arange(len(self.latent_masked))
        
        cluster_info = f" (cluster {cluster_id})" if cluster_id is not None else ""
        print(f"Loaded {len(self)} samples from {split} split{cluster_info} (Memory Mapped)")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.valid_indices[idx]
        
        # These are read from disk on-the-fly due to mmap_mode to save RAM
        masked_latent = torch.from_numpy(self.latent_masked[real_idx]).float()
        target_latent = torch.from_numpy(self.latent_target[real_idx]).float()
        image = torch.from_numpy(self.images[real_idx]).float()
        mask = torch.from_numpy(self.masks[real_idx]).long()

        return {
            "masked_latent": masked_latent,
            "target_latent": target_latent,
            "image": image,
            "mask": mask
        }


class LatentInpainterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        latent_dir: str = "data/latent_spaces",
        cluster_id: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        
        self.latent_dir = latent_dir
        self.cluster_id = cluster_id
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[LatentInpainterDataset] = None
        self.val_dataset: Optional[LatentInpainterDataset] = None
        self.test_dataset: Optional[LatentInpainterDataset] = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LatentInpainterDataset(
            latent_dir=self.latent_dir,
            split="train",
            cluster_id=self.cluster_id,
        )
        self.val_dataset = LatentInpainterDataset(
            latent_dir=self.latent_dir,
            split="val",
            cluster_id=self.cluster_id,
        )
        self.test_dataset = LatentInpainterDataset(
            latent_dir=self.latent_dir,
            split="test",
            cluster_id=self.cluster_id,
        )

        if stage == "fit" or stage is None:
            print(f"\n{'=' * 60}")
            print(f"Inpainter Dataset Setup Complete:")
            print(f"  Cluster ID: {self.cluster_id if self.cluster_id is not None else 'All'}")
            print(f"  Training:   {len(self.train_dataset)} samples")
            print(f"  Validation: {len(self.val_dataset)} samples")
            print(f"{'=' * 60}\n")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
