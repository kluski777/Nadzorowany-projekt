import torch.nn as nn

from .downsample_block import DownsampleBlock


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

