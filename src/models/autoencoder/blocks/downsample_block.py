import torch.nn as nn

from .residual_block import ResidualBlock


class DownsampleBlock(nn.Module):
    """
    Downsampling block for encoder architecture.

    Consists of:
    - Conv2d layer for downsampling
    - BatchNorm2d for normalization
    - GELU activation
    - Optional ResidualBlock for feature refinement
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_residual: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]

        if use_residual:
            layers.append(ResidualBlock(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
