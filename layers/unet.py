import torch
import torch.nn as nn

from layers.conv import ConvolutionBlock, ResidualBlock


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            ConvolutionBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
            ),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        stacked_input = torch.cat((x, s), dim=1)
        return self.layers(stacked_input)


class UnetBridge(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                stride=7,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UnetOutputBlock(nn.Module):
    def __init__(self, hidden_channels: int, in_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_channels
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
