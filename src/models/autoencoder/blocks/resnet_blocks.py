import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    """
    Downsampling block that changes dimensions (for encoder).
    
    Two convolutions with a skip connection that gets adjusted to match the new size.
    Input: (batch, in_channels, H, W)
    Output: (batch, out_channels, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        
        # Two 3x3 convolutions (first one downsamples with stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Adjust the skip connection: 1x1 conv to match channels and size
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for skip connection
        skip = self.skip_conv(x)
        
        # Main path: two convolutions
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        
        # Add skip and activate
        return self.activation(x + skip)


class IdentityBlock(nn.Module):
    """
    Block that keeps the same dimensions (for feature refinement).
    
    Two convolutions with a direct skip connection.
    Input: (batch, channels, H, W)
    Output: (batch, channels, H, W)  [same size]
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        
        # Two 3x3 convolutions (both with stride=1, no downsampling)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for skip connection (no adjustment needed)
        skip = x
        
        # Main path: two convolutions
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        
        # Add skip and activate
        return self.activation(x + skip)


class UpsampleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle (for decoder).
    
    Expands channels then rearranges them to increase spatial size.
    Input: (batch, in_channels, H, W)
    Output: (batch, out_channels, H*2, W*2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Expand to 4x channels (PixelShuffle will convert 4 channels -> 2x2 spatial)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)  # Rearrange: (C*4, H, W) -> (C, H*2, W*2)
        x = self.bn(x)
        return self.activation(x)

