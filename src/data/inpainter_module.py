from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


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
        
        npz_path = self.latent_dir / f"{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Latent space file not found: {npz_path}")
        
        data = np.load(npz_path)
        
        self.masked_latent = data["masked_latent"]
        self.target_latent = data["target_latent"]
        self.indices = data["indices"]
        
        has_clusters = "cluster" in data.files
        self.clusters = data["cluster"] if has_clusters else None
        
        if cluster_id is not None:
            if not has_clusters:
                raise ValueError(
                    f"cluster_id={cluster_id} specified, but the latent space file "
                    f"'{npz_path}' does not contain cluster labels. "
                    "Regenerate latent spaces with --feature-extractor-checkpoint and "
                    "--clusterizer-checkpoint to include cluster labels."
                )
            mask = self.clusters == cluster_id
            self.masked_latent = self.masked_latent[mask]
            self.target_latent = self.target_latent[mask]
            self.clusters = self.clusters[mask]
            self.indices = self.indices[mask]
        
        cluster_info = ""
        if cluster_id is not None:
            cluster_info = f" (cluster {cluster_id})"
        elif not has_clusters:
            cluster_info = " (no cluster labels)"
        
        print(f"Loaded {len(self)} samples from {split} split{cluster_info}")

    def __len__(self) -> int:
        return len(self.masked_latent)

    def __getitem__(self, idx: int) -> dict:
        masked = torch.from_numpy(self.masked_latent[idx]).float()
        target = torch.from_numpy(self.target_latent[idx]).float()
        
        return {
            "masked_latent": masked,
            "target_latent": target,
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
        if stage == "fit" or stage is None:
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
        
        if stage == "test" or stage is None:
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
