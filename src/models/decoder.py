import torch.nn as nn

from .residual_block import ResidualBlock


class Decoder(nn.Module):
    
    def __init__(self, latent_channels: int = 128, output_channels: int = 3):
        super().__init__()
        
        self.network = nn.Sequential(
            # (latent_channels x 8 x 8) -> (512 x 16 x 16)
            nn.ConvTranspose2d(latent_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            ResidualBlock(512),
            
            # (512 x 16 x 16) -> (256 x 32 x 32)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256),
            
            # (256 x 32 x 32) -> (128 x 64 x 64)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            
            # (128 x 64 x 64) -> (64 x 128 x 128)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),
            
            # (64 x 128 x 128) -> (32 x 256 x 256)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # (32 x 256 x 256) -> (output_channels x 256 x 256)
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.network(x)

