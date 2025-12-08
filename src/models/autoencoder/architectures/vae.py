import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple

from models.losses import vae_bce_kl_loss


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_channels: int = 512, base_channels: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        
        self.conv1 = self._make_conv_block(input_channels, base_channels)
        self.conv2 = self._make_conv_block(base_channels, base_channels * 2)
        self.conv3 = self._make_conv_block(base_channels * 2, base_channels * 4)
        self.conv4 = self._make_conv_block(base_channels * 4, base_channels * 8)
        self.conv5 = self._make_conv_block(base_channels * 8, base_channels * 8)
        self.conv6 = self._make_conv_block(base_channels * 8, base_channels * 8)
        
        self.flatten_size = 4 * 4 * base_channels * 8
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_channels)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_channels)
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = 512, output_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        self.fc = nn.Linear(latent_channels, 4 * 4 * base_channels * 8)
        
        self.deconv1 = self._make_deconv_block(base_channels * 8, base_channels * 8)
        self.deconv2 = self._make_deconv_block(base_channels * 8, base_channels * 8)
        self.deconv3 = self._make_deconv_block(base_channels * 8, base_channels * 4)
        self.deconv4 = self._make_deconv_block(base_channels * 4, base_channels * 2)
        self.deconv5 = self._make_deconv_block(base_channels * 2, base_channels)
        
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def _make_deconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.base_channels * 8, 4, 4)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        
        return x


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 512,
        base_channels: int = 32,
        learning_rate: float = 1e-3,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        loss_type: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_channels = latent_channels
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        
        self.encoder = Encoder(input_channels, latent_channels, base_channels)
        self.decoder = Decoder(latent_channels, input_channels, base_channels)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_channels).to(device)
        samples = self.decoder(z)
        return samples
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        recon, mu, logvar = self(images)
        
        losses = vae_bce_kl_loss(recon, images, mu, logvar)
        
        self.log("train_loss", losses['loss'], prog_bar=True, sync_dist=True)
        self.log("train_recon_loss", losses['recon_loss'], sync_dist=True)
        self.log("train_kl_loss", losses['kl_loss'], sync_dist=True)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        recon, mu, logvar = self(images)
        
        losses = vae_bce_kl_loss(recon, images, mu, logvar)
        
        self.log("val_loss", losses['loss'], prog_bar=True, sync_dist=True)
        self.log("val_recon_loss", losses['recon_loss'], sync_dist=True)
        self.log("val_kl_loss", losses['kl_loss'], sync_dist=True)
        
        return losses['loss']
    
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
