import torch
import torch.nn as nn

import pytorch_lightning as pl

from models.losses import get_loss_function
from models.autoencoder import get_autoencoder

# class ResidualConvBlock(nn.Module): # chcialbym unet
#     def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3):
#         super().__init__()
#         padding = kernel_size // 2

#         self.block = nn.Sequential(
#             nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(hidden_channels), # dla include_batch_norma == True dziala tak samo dla False po prostu znikaja te warstwy
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(hidden_channels, channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#         )
#         self.activation = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.activation(x + self.block(x))

# To bylo w funkcji
# self.input_conv = nn.Sequential(
#     nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(hidden_channels),
#     nn.LeakyReLU(0.2, inplace=True),
# )

# self.residual_blocks = nn.Sequential(
#     *[ResidualConvBlock(hidden_channels, hidden_channels * 2, kernel_size=3) for _ in range(num_blocks)]
# )

# self.output_conv = nn.Sequential(
#     nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, stride=1, padding=1),
# )

class GatedUNet(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, depth: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depth = depth
        # ta linijka jest stad ze konkatenowana jest informacja o masce na poczatku do inpuu
        # i chcemy zejsc do wymiarow obrazka, dlatego // 2

        self.gated_conv_layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding*dilation, dilation=dilation),
            nn.Sigmoid(),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size, padding=padding * dilation, dilation=dilation),
            nn.GELU(),
            nn.GroupNorm(16, hidden_channels),
        )
        
        if depth > 0:
            self.inner_block = GatedUNet(hidden_channels, hidden_channels, depth - 1, kernel_size, dilation)
        else:
            self.inner_block = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding * dilation, dilation=dilation),
            nn.GELU(),
            nn.GroupNorm(16, hidden_channels),
            nn.Conv2d(hidden_channels, channels, kernel_size, padding=padding * dilation, dilation=dilation),
        )

    def forward(self, x):
        gated_out = self.gated_conv_layer(x)
        h = self.encoder(x)
        h = self.inner_block(h)
        out = self.decoder(h)
        return out * gated_out + x


class ConvLatentInpainter(pl.LightningModule):
    def __init__(
        self,
        latent_channels: int = 128,
        hidden_channels: int = 256,
        num_blocks: int = 4,
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5, 
        scheduler_factor: float = 0.5,
        reconstructed_loss_type: str = "charbonnier",
        reconstructed_loss_weight: float = 2.0,
        latent_loss_type: str = "charbonnier",
        ae_path: str = "",
        architecture: str = ""
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.recon_weight = reconstructed_loss_weight
        self.pixel_shuffle_val = 4

        self.decoded_loss = get_loss_function(reconstructed_loss_type) #TODO perceptual loss luczak mowil najlepiej na VGG
        self.latent_loss = get_loss_function(latent_loss_type)

        self.pixel_shuffle = nn.PixelShuffle(self.pixel_shuffle_val)
        self.network = GatedUNet(latent_channels // self.pixel_shuffle_val**2, hidden_channels, depth=num_blocks, kernel_size=3, dilation=1)
        self.unshuffle = nn.PixelUnshuffle(self.pixel_shuffle_val)
        
        if ae_path and ae_path != "No_argument":
            self.autoencoder = get_autoencoder(architecture).load_from_checkpoint(ae_path)
            self.autoencoder.requires_grad_(False)
            self.autoencoder.eval()
        else:
            self.autoencoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shuffled = self.pixel_shuffle(x)
        networked = self.network(shuffled)
        normalized_output = self.unshuffle(networked)
        return normalized_output

    def training_step(self, batch, batch_idx):
        z_corrupted = batch["masked_latent"]      # E(x ⊙ m^c)
        z_target = batch["target_latent"]         # E(x)
        image = batch["image"]
        mask = batch["mask"].long()
        b_idx, h_idx, w_idx = torch.where(mask == 0)
        
        original_image_cut_part = image[b_idx, :, h_idx, w_idx]

        predicted_latent = self(z_corrupted)
        latent_loss = self.latent_loss(predicted_latent, z_target)
        
        if abs(self.recon_weight) > 1e-4:
            reconstructed_image = self.autoencoder.decoder(predicted_latent)
            reconstructed_image_cut = reconstructed_image[b_idx, :, h_idx, w_idx]
            reconstructed_loss = self.decoded_loss(original_image_cut_part, reconstructed_image_cut)
            loss = latent_loss + self.recon_weight * reconstructed_loss
        else:
            loss = latent_loss
            
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z_corrupted = batch["masked_latent"]      # E(x ⊙ m^c)
        z_target = batch["target_latent"]         # E(x)
        image = batch["image"]
        mask = batch["mask"].long()
        b_idx, h_idx, w_idx = torch.where(mask == 0)
        
        # to jest ten wyciety obraz kretynie - i wszystko jasne
        original_image_cut_part = image[b_idx, :, h_idx, w_idx]

        predicted_latent = self(z_corrupted)
        latent_loss = self.latent_loss(predicted_latent, z_target)
        
        if abs(self.recon_weight) > 1e-4:
            reconstructed_image = self.autoencoder.decoder(predicted_latent)
            reconstructed_image_cut = reconstructed_image[b_idx, :, h_idx, w_idx]
            reconstructed_loss = self.decoded_loss(original_image_cut_part, reconstructed_image_cut)
            loss = latent_loss + self.recon_weight * reconstructed_loss
        else:
            loss = latent_loss
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        z_corrupted = batch["masked_latent"]      # E(x ⊙ m^c)
        z_target = batch["target_latent"]         # E(x)
        image = batch["image"]
        mask = batch["mask"].long()
        b_idx, h_idx, w_idx = torch.where(mask == 0)
        
        original_image_cut_part = image[b_idx, :, h_idx, w_idx]

        predicted_latent = self(z_corrupted)
        latent_loss = self.latent_loss(predicted_latent, z_target)
        
        if abs(self.recon_weight) > 1e-4:
            reconstructed_image = self.autoencoder.decoder(predicted_latent)
            reconstructed_image_cut = reconstructed_image[b_idx, :, h_idx, w_idx]
            reconstructed_loss = self.decoded_loss(original_image_cut_part, reconstructed_image_cut)
            loss = latent_loss + self.recon_weight * reconstructed_loss
        else:
            loss = latent_loss
            
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