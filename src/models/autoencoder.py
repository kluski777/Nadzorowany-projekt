from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import pytorch_lightning as pl

if TYPE_CHECKING:
    from torch import Tensor


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 128,
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        # U-NET https://arxiv.org/pdf/1505.04597

        self.encoder = nn.Sequential(
            # (input_channels x 256 x 256) -> (64 x 128 x 128)
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # (64 x 128 x 128) -> (128 x 64 x 64)
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # (128 x 64 x 64) -> (256 x 32 x 32)
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # (256 x 32 x 32) -> (512 x 16 x 16)
            nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(512),
            nn.GELU(),

            # (512 x 16 x 16) -> (latent_channels x 8 x 8)
            nn.Conv2d(512, latent_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            # (latent_channels x 8 x 8) -> (512 x 16 x 16)
            nn.ConvTranspose2d(
                latent_channels, 512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.GELU(),

            # (512 x 16 x 16) -> (256 x 32 x 32)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # (256 x 32 x 32) -> (128 x 64 x 64)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # (128 x 64 x 64) -> (64 x 128 x 128)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # (64 x 128 x 128) -> (3 x 256 x 256)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: "Tensor") -> "Tensor":
        latent_space = self.encoder(x)
        reconstructed_image = self.decoder(latent_space)
        return reconstructed_image

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        reconstructed = self(images)

        loss = nn.functional.mse_loss(reconstructed, images)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        reconstructed = self(images)

        loss = nn.functional.mse_loss(reconstructed, images)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
