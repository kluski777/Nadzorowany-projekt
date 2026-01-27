import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import numpy as np

from models.losses import get_loss_function
# No w chuj dlugo sie to trenuje duzy problem na ten moment


class ResidualConvESDR(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x)) # reset nie pojdzie
    

class EDSR(pl.LightningModule):
    def __init__(self, 
            loss_type: str, 
            hidden_size: list[int] | np.ndarray, 
            learning_rate: float, 
            scheduler_factor: float = 1.0, 
            scheduler_patience: int = 1_000, 
            training_image_size: int = 96, 
            blurring: str = 'Gaussian'
        ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.scheduler_factor  = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.training_image_size = training_image_size
        self.blurring = blurring
        # state dac z fourierem nieco
        def custom_loss_function(pred, tar):
            # trzeba abs zeby na rzeczywiste liczby przejsc z imaginowanych
            fft_pred, fft_tar = torch.fft.rfft2(pred).abs(), torch.fft.rfft2(tar).abs()
            charbonnier_losss = torch.sqrt((pred - tar).square() + 1e-6).mean()
            return charbonnier_losss + 0.2 * F.l1_loss(fft_pred, fft_tar)

        self.loss_fn = custom_loss_function

        outline = [ nn.Conv2d(in_channels=3, out_channels=hidden_size[0], kernel_size=3, padding=1), nn.GELU() ]
        for i in range(1, len(hidden_size) - 2):
            outline.append(
                ResidualConvESDR(channels=hidden_size[i], hidden_channels=hidden_size[i+1], kernel_size=3)
            )
        
        outline.extend([
            nn.Conv2d(in_channels=hidden_size[-2], out_channels=4*hidden_size[-1], kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=hidden_size[-1], out_channels=3, kernel_size=3, padding=1)
        ])
        outline.append( nn.Hardsigmoid(inplace=True) ) # zeby w dobrym range'u byly nasze zmienne
        self.network = nn.Sequential(*outline)


    def forward(self, x): # to mi wystarczy jak już się nauczy i tego ędzie się używać
        return self.network(x)
    
    
    def step(self, batch):
        # tutaj jakos zepsuc to zdjecie / najlepiej byloby te zepsute zdjecia jakos zapisac
        # normalnie obraz jest 256 x 256 wybierzmy sobie jakies indexy tak zeby byl 96 x 96
        original_image = batch["image"] 

        x_index, y_index = torch.randint(low=0, high=256-self.training_image_size, size=(2,))
        to_downsample = original_image[:, :, x_index:x_index+self.training_image_size, y_index:y_index+self.training_image_size] # B, C, i tu tniemy

        # Gaussian blur to be added in the future I guess   
        if self.blurring == 'Gaussian':
            to_downsample = TF.gaussian_blur(img=to_downsample, kernel_size=[5], sigma=[1])

        downsampled_image = F.interpolate(
            to_downsample,
            scale_factor=0.5,
            mode='bicubic',
            align_corners=False,
            antialias=True # nie wiem czy az tak potrzebne
        )
        up_sampled = self(downsampled_image)
        loss = self.loss_fn(up_sampled, to_downsample)
        return loss
        

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            }
        }