import torch.nn as nn

from .residual_block import ResidualBlock


class Encoder(nn.Module):
    
    def __init__(self, input_channels: int = 3, latent_channels: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            # (input_channels x 256 x 256) -> (64 x 128 x 128)
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64),
            
            # (64 x 128 x 128) -> (128 x 64 x 64)s
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            
            # (128 x 64 x 64) -> (256 x 32 x 32)
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256),
            
            # (256 x 32 x 32) -> (512 x 16 x 16)
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            ResidualBlock(512),
            
            # (512 x 16 x 16) -> (latent_channels x 8 x 8)
            nn.Conv2d(512, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.network(x)

