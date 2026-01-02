import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import get_loss_function
from models.autoencoder.architectures.residual_convt import ResidualConvtAutoEncoder


class Compressor(nn.Module):
    """Compresses latent space using 1x1 convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Decompressor(nn.Module):
    """Decompresses latent space using 1x1 convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BottleneckAE4k(pl.LightningModule):
    """
    4096 latent dimensions (64 channels x 8x8).
    
    Loads base ResidualConvtAutoEncoder (8k latent) and adds 128->64 bottleneck.
    Freezes: encoder, decoder
    Trainable: compressor_128_64, decompressor_64_128
    """

    def __init__(
        self,
        base_checkpoint: str,
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

        # Load base model and extract encoder/decoder
        base_model = ResidualConvtAutoEncoder.load_from_checkpoint(
            base_checkpoint, strict=True
        )
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder

        # Freeze encoder and decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()

        # Trainable bottleneck: 128 -> 64 -> 128
        self.compressor_128_64 = Compressor(128, 64)
        self.decompressor_64_128 = Decompressor(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode: 256x256x3 -> 8x8x128
        latent = self.encoder(x)
        # Compress: 8x8x128 -> 8x8x64
        compressed = self.compressor_128_64(latent)
        # Decompress: 8x8x64 -> 8x8x128
        decompressed = self.decompressor_64_128(compressed)
        # Decode: 8x8x128 -> 256x256x3
        return self.decoder(decompressed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoding to 4k latent space."""
        latent = self.encoder(x)
        return self.compressor_128_64(latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Full decoding from 4k latent space."""
        decompressed = self.decompressor_64_128(z)
        return self.decoder(decompressed)

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
        trainable_params = list(self.compressor_128_64.parameters()) + list(
            self.decompressor_64_128.parameters()
        )
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_train_epoch_start(self):
        """Ensure frozen layers stay in eval mode."""
        self.encoder.eval()
        self.decoder.eval()


class BottleneckAE2k(pl.LightningModule):
    """
    2048 latent dimensions (32 channels x 8x8).
    
    Loads BottleneckAE4k checkpoint and adds 64->32 bottleneck.
    Freezes: encoder, decoder, compressor_128_64, decompressor_64_128
    Trainable: compressor_64_32, decompressor_32_64
    """

    def __init__(
        self,
        ae4k_checkpoint: str,
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

        # Load 4k model and extract frozen components
        ae4k = BottleneckAE4k.load_from_checkpoint(ae4k_checkpoint, strict=True)
        self.encoder = ae4k.encoder
        self.decoder = ae4k.decoder
        self.compressor_128_64 = ae4k.compressor_128_64
        self.decompressor_64_128 = ae4k.decompressor_64_128

        # Freeze all inherited components
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.compressor_128_64.parameters():
            param.requires_grad = False
        for param in self.decompressor_64_128.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()
        self.compressor_128_64.eval()
        self.decompressor_64_128.eval()

        # Trainable bottleneck: 64 -> 32 -> 64
        self.compressor_64_32 = Compressor(64, 32)
        self.decompressor_32_64 = Decompressor(32, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode: 256x256x3 -> 8x8x128 -> 8x8x64
        latent = self.encoder(x)
        latent = self.compressor_128_64(latent)
        # Compress: 8x8x64 -> 8x8x32
        compressed = self.compressor_64_32(latent)
        # Decompress: 8x8x32 -> 8x8x64 -> 8x8x128
        decompressed = self.decompressor_32_64(compressed)
        decompressed = self.decompressor_64_128(decompressed)
        # Decode: 8x8x128 -> 256x256x3
        return self.decoder(decompressed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoding to 2k latent space."""
        latent = self.encoder(x)
        latent = self.compressor_128_64(latent)
        return self.compressor_64_32(latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Full decoding from 2k latent space."""
        decompressed = self.decompressor_32_64(z)
        decompressed = self.decompressor_64_128(decompressed)
        return self.decoder(decompressed)

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
        trainable_params = list(self.compressor_64_32.parameters()) + list(
            self.decompressor_32_64.parameters()
        )
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_train_epoch_start(self):
        """Ensure frozen layers stay in eval mode."""
        self.encoder.eval()
        self.decoder.eval()
        self.compressor_128_64.eval()
        self.decompressor_64_128.eval()


class BottleneckAE1k(pl.LightningModule):
    """
    1024 latent dimensions (16 channels x 8x8).

    Loads BottleneckAE2k checkpoint and adds 32->16 bottleneck.
    Freezes: encoder, decoder, compressor_128_64, decompressor_64_128, 
             compressor_64_32, decompressor_32_64
    Trainable: compressor_32_16, decompressor_16_32
    """

    def __init__(
        self,
        ae2k_checkpoint: str,
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

        # Load 2k model and extract frozen components
        ae2k = BottleneckAE2k.load_from_checkpoint(ae2k_checkpoint, strict=True)
        self.encoder = ae2k.encoder
        self.decoder = ae2k.decoder
        self.compressor_128_64 = ae2k.compressor_128_64
        self.decompressor_64_128 = ae2k.decompressor_64_128
        self.compressor_64_32 = ae2k.compressor_64_32
        self.decompressor_32_64 = ae2k.decompressor_32_64

        # Freeze all inherited components
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.compressor_128_64.parameters():
            param.requires_grad = False
        for param in self.decompressor_64_128.parameters():
            param.requires_grad = False
        for param in self.compressor_64_32.parameters():
            param.requires_grad = False
        for param in self.decompressor_32_64.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()
        self.compressor_128_64.eval()
        self.decompressor_64_128.eval()
        self.compressor_64_32.eval()
        self.decompressor_32_64.eval()

        # Trainable bottleneck: 32 -> 16 -> 32
        self.compressor_32_16 = Compressor(32, 16)
        self.decompressor_16_32 = Decompressor(16, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode: 256x256x3 -> 8x8x128 -> 8x8x64 -> 8x8x32
        latent = self.encoder(x)
        latent = self.compressor_128_64(latent)
        latent = self.compressor_64_32(latent)
        # Compress: 8x8x32 -> 8x8x16
        compressed = self.compressor_32_16(latent)
        # Decompress: 8x8x16 -> 8x8x32 -> 8x8x64 -> 8x8x128
        decompressed = self.decompressor_16_32(compressed)
        decompressed = self.decompressor_32_64(decompressed)
        decompressed = self.decompressor_64_128(decompressed)
        # Decode: 8x8x128 -> 256x256x3
        return self.decoder(decompressed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoding to 1k latent space."""
        latent = self.encoder(x)
        latent = self.compressor_128_64(latent)
        latent = self.compressor_64_32(latent)
        return self.compressor_32_16(latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Full decoding from 1k latent space."""
        decompressed = self.decompressor_16_32(z)
        decompressed = self.decompressor_32_64(decompressed)
        decompressed = self.decompressor_64_128(decompressed)
        return self.decoder(decompressed)

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
        trainable_params = list(self.compressor_32_16.parameters()) + list(
            self.decompressor_16_32.parameters()
        )
        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_train_epoch_start(self):
        """Ensure frozen layers stay in eval mode."""
        self.encoder.eval()
        self.decoder.eval()
        self.compressor_128_64.eval()
        self.decompressor_64_128.eval()
        self.compressor_64_32.eval()
        self.decompressor_32_64.eval()
