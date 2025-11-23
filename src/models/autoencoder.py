import torch
import torch.nn as nn
import pytorch_lightning as pl

from .losses import get_loss_function
from .encoder import Encoder
from .decoder import Decoder


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 128,
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        loss_type: str = "ssim",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.loss_type = loss_type

        self.loss_fn = get_loss_function(loss_type)

        self.encoder = Encoder(input_channels=input_channels, latent_channels=latent_channels)
        self.decoder = Decoder(latent_channels=latent_channels, output_channels=input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_space = self.encoder(x)
        reconstructed_image = self.decoder(latent_space)
        return reconstructed_image

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        reconstructed = self(images)

        loss = self.loss_fn(reconstructed, images)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        reconstructed = self(images)

        loss = self.loss_fn(reconstructed, images)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

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
                "monitor": "val_loss",
            },
        }
