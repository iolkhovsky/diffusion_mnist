import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        hidden = self.block1(x)
        output = self.block2(hidden)
        if return_hidden:
            return output, hidden
        else:
            return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvolutionBlock(in_channels, out_channels)
        self.same_depth = in_channels == out_channels
        self.norm_k = torch.sqrt(torch.tensor(2.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.conv(x, True)
        if self.same_depth:
            return output + x
        else:
            return (output + hidden) / self.norm_k
