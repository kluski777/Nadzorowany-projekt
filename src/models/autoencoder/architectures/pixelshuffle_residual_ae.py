import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.losses import get_loss_function
from models.autoencoder.blocks import ResidualBlock


class PixelShuffleEncoder(nn.Module):

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 32,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            # 256x256x3 -> 128x128x64
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),
            # 128x128x64 -> 64x64x128
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            # 64x64x128 -> 32x32x256
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256),
            # 32x32x256 -> 16x16x512
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            ResidualBlock(512),
            # 16x16x512 -> 8x8x32
            nn.Conv2d(512, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PixelShuffleDecoder(nn.Module):

    def __init__(
        self,
        latent_channels: int = 32,
        output_channels: int = 3,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            # 8x8x32 -> 16x16x512
            nn.Conv2d(latent_channels, 2048, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            ResidualBlock(512),
            # 16x16x512 -> 32x32x256
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256),
            # 32x32x256 -> 64x64x128
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            # 64x64x128 -> 128x128x64
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),
            # 128x128x64 -> 256x256x3
            nn.Conv2d(64, 12, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PixelShuffleResidualAE(pl.LightningModule):

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 32,
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.loss_type = loss_type
        
        self.loss_fn = get_loss_function(loss_type)
        
        self.encoder = PixelShuffleEncoder(
            input_channels=input_channels,
            latent_channels=latent_channels,
        )
        
        self.decoder = PixelShuffleDecoder(
            latent_channels=latent_channels,
            output_channels=input_channels,
        )

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
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
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
