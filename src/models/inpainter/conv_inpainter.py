import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import get_loss_function


class ResidualConvBlock(nn.Module):
    """Residual convolutional block that preserves spatial dimensions."""

    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ConvLatentInpainter(pl.LightningModule):
    """
    Convolutional inpainter for latent space reconstruction.
    
    Takes a masked latent space and outputs a reconstructed latent space
    that should match the original unmasked latent space.
    
    Uses residual learning: output = input + model(input)
    This allows the model to learn the "correction" needed to fix the masked regions.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_channels: int = 256,
        num_blocks: int = 4,
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

        # Initial projection to hidden channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stack of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualConvBlock(hidden_channels, hidden_channels * 2, kernel_size=3) for _ in range(num_blocks)]
        )

        # Final projection back to latent channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Masked latent space of shape (B, latent_channels, H, W)
            
        Returns:
            Reconstructed latent space of shape (B, latent_channels, H, W)
        """
        # Process through conv network
        h = self.input_conv(x)
        h = self.residual_blocks(h)
        correction = self.output_conv(h)
        
        # Residual learning: output = input + correction
        return x + correction

    def training_step(self, batch, batch_idx):
        masked_latent = batch["masked_latent"]
        target_latent = batch["target_latent"]

        predicted_latent = self(masked_latent)
        loss = self.loss_fn(predicted_latent, target_latent)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masked_latent = batch["masked_latent"]
        target_latent = batch["target_latent"]

        predicted_latent = self(masked_latent)
        loss = self.loss_fn(predicted_latent, target_latent)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        masked_latent = batch["masked_latent"]
        target_latent = batch["target_latent"]

        predicted_latent = self(masked_latent)
        loss = self.loss_fn(predicted_latent, target_latent)

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
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
