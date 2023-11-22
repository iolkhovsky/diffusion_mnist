import torch
import torch.nn as nn

from layers.conv import ConvolutionBlock, ResidualBlock


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvolutionBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        self.conv1 = ResidualBlock(out_channels, out_channels)
        self.conv2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        stacked_input = torch.cat((x, s), dim=1)
        return self.conv2(
            self.conv1(self.upscale(stacked_input))
        )


class UnetBridge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            stride=7,
        )
        self.norm = nn.GroupNorm(
            num_groups=8,
            num_channels=channels,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UnetOutputBlock(nn.Module):
    def __init__(self, hidden_channels, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2 * hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm = nn.GroupNorm(
            num_groups=8,
            num_channels=hidden_channels
        )
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(
            self.act(self.norm(self.conv1(x)))
        )
