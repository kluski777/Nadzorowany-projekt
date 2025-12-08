import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.losses import get_loss_function
from models.autoencoder.blocks import ConvolutionBlock, IdentityBlock, ResNetUpsampleBlock


class ResNet18Encoder(nn.Module):
    """
    ResNet18-inspired encoder for autoencoder.
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Starting block: 256x256x3 -> 64x64x64
        self.starting_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Stage 1: 64x64x64 -> 64x64x64 (N identity blocks)
        self.stage1 = self._make_stage(64, 64, num_blocks=2, dropout=dropout, downsample=False)
        
        # Stage 2: 64x64x64 -> 32x32x128 (1 conv + N-1 identity)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, dropout=dropout, downsample=True)
        
        # Stage 3: 32x32x128 -> 16x16x256 (1 conv + N-1 identity)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, dropout=dropout, downsample=True)
        
        # Stage 4: 16x16x256 -> 8x8x512 (1 conv + N-1 identity)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, dropout=dropout, downsample=True)
        
        # Bottleneck: 8x8x512 -> 8x8x32 (2048 elements)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
        )

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, dropout: float, downsample: bool):
        """Build a stage: 1 ConvBlock (if downsampling) + remaining IdentityBlocks."""
        layers = []
        
        if downsample:
            # First block changes size/channels
            layers.append(ConvolutionBlock(in_channels, out_channels, dropout=dropout))
            # Rest refine features at the new size
            for _ in range(num_blocks - 1):
                layers.append(IdentityBlock(out_channels, dropout=dropout))
        else:
            # All blocks just refine features (no size change)
            for _ in range(num_blocks):
                layers.append(IdentityBlock(in_channels, dropout=dropout))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.starting_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bottleneck(x)
        return x


class ResNet18Decoder(nn.Module):
    """
    ResNet18-inspired decoder for autoencoder.
    """

    def __init__(
        self,
        latent_channels: int = 32,
        output_channels: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Stage 1: 8x8x32 -> 16x16x512
        self.stage1_upsample = ResNetUpsampleBlock(latent_channels, 512)
        self.stage1_refine = self._make_identity_stage(512, num_blocks=2, dropout=dropout)
        
        # Stage 2: 16x16x512 -> 32x32x256
        self.stage2_upsample = ResNetUpsampleBlock(512, 256)
        self.stage2_refine = self._make_identity_stage(256, num_blocks=2, dropout=dropout)
        
        # Stage 3: 32x32x256 -> 64x64x128
        self.stage3_upsample = ResNetUpsampleBlock(256, 128)
        self.stage3_refine = self._make_identity_stage(128, num_blocks=2, dropout=dropout)
        
        # Stage 4: 64x64x128 -> 128x128x64
        self.stage4_upsample = ResNetUpsampleBlock(128, 64)
        self.stage4_refine = self._make_identity_stage(64, num_blocks=2, dropout=dropout)
        
        # Stage 5: 128x128x64 -> 256x256x12 -> 256x256x3
        self.stage5_upsample = ResNetUpsampleBlock(64, 12)
        self.output_conv = nn.Sequential(
            nn.Conv2d(12, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def _make_identity_stage(self, channels: int, num_blocks: int, dropout: float):
        return nn.Sequential(*[IdentityBlock(channels, dropout=dropout) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        x = self.stage1_upsample(x)
        x = self.stage1_refine(x)
        
        # Stage 2
        x = self.stage2_upsample(x)
        x = self.stage2_refine(x)
        
        # Stage 3
        x = self.stage3_upsample(x)
        x = self.stage3_refine(x)
        
        # Stage 4
        x = self.stage4_upsample(x)
        x = self.stage4_refine(x)
        
        # Stage 5 (output)
        x = self.stage5_upsample(x)
        x = self.output_conv(x)
        
        return x


class ResNet18AutoEncoder(pl.LightningModule):
    """
    ResNet18-inspired autoencoder with configurable residual blocks.
    """

    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 32,
        dropout: float = 0.0,
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
        
        self.encoder = ResNet18Encoder(
            input_channels=input_channels,
            latent_channels=latent_channels,
            dropout=dropout,
        )
        
        self.decoder = ResNet18Decoder(
            latent_channels=latent_channels,
            output_channels=input_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_space = self.encoder(x)
        reconstructed_image = self.decoder(latent_space)
        return reconstructed_image
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

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

