import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import get_loss_function

from models.autoencoder.blocks import DownsampleBlock, ResidualBlock


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_channels: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            # (input_channels x 256 x 256) -> (64 x 128 x 128)
            DownsampleBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, use_residual=True),
            # (64 x 128 x 128) -> (128 x 64 x 64)
            DownsampleBlock(64, 128, kernel_size=7, stride=2, padding=3, use_residual=True),
            # (128 x 64 x 64) -> (256 x 32 x 32)
            DownsampleBlock(128, 256, kernel_size=5, stride=2, padding=2, use_residual=True),
            # (256 x 32 x 32) -> (512 x 16 x 16)
            DownsampleBlock(256, 512, kernel_size=5, stride=2, padding=2, use_residual=True),
            # (512 x 16 x 16) -> (latent_channels x 8 x 8)
            DownsampleBlock(512, latent_channels, kernel_size=3, stride=2, padding=1, use_residual=False),
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = 128, output_channels: int = 3):
        super().__init__()

        self.network = nn.Sequential(
            # (latent_channels x 8 x 8) -> (128 x 16 x 16)
            nn.Conv2d(latent_channels, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.GELU(),
            ResidualBlock(512),
            nn.PixelShuffle(2),
            # (128 x 16 x 16) -> (64 x 32 x 32)
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256),
            nn.PixelShuffle(2),
            # (64 x 32 x 32) -> (32 x 64 x 64)
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            nn.PixelShuffle(2),
            # (32 x 64 x 64) -> (16 x 128 x 128)
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),
            nn.PixelShuffle(2),
            # (16 x 128 x 128) -> (8 x 256 x 256)
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32),
            nn.PixelShuffle(2),
            # (8 x 256 x 256) -> (output_channels x 256 x 256)
            nn.Conv2d(8, output_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class ResK1UpsampleAutoEncoder(pl.LightningModule):
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
