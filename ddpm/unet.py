import torch
import torch.nn as nn

from layers import (
    EmbeddingLayer, ResidualBlock, UnetEncoderBlock,
    UnetDecoderBlock, UnetBridge, UnetOutputBlock
)


class ContextUnet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, n_classes: int):
        super().__init__()

        self.dims = hidden_channels
        self.n_classes = n_classes

        self.conv = ResidualBlock(in_channels, hidden_channels)
        self.enc1 = UnetEncoderBlock(hidden_channels, hidden_channels)
        self.enc2 = UnetEncoderBlock(hidden_channels, 2 * hidden_channels)

        self.time_emb1 = EmbeddingLayer(1, 2 * hidden_channels)
        self.time_emb2 = EmbeddingLayer(1, hidden_channels)
        self.cls_emb1 = EmbeddingLayer(n_classes, 2 * hidden_channels)
        self.cls_emb2 = EmbeddingLayer(n_classes, hidden_channels)

        self.bridge = UnetBridge(2 * hidden_channels)
        self.dec1 = UnetDecoderBlock(4 * hidden_channels, hidden_channels)
        self.dec2 = UnetDecoderBlock(2 * hidden_channels, hidden_channels)

        self.output = UnetOutputBlock(hidden_channels, in_channels)

    def forward(
        self, x: torch.Tensor, c: torch.LongTensor, t: torch.Tensor,
        mask: torch.LongTensor
    ) -> torch.Tensor:
        assert len(x.shape) == 4, f'x.shape = {x.shape}'
        batch_size = x.shape[0]
        assert len(c.shape) == 1 and c.shape[0] == batch_size, f'c.shape = {c.shape}'
        assert len(t.shape) == 1 and t.shape[0] == batch_size, f't.shape = {t.shape}'
        assert len(mask.shape) == 1 and mask.shape[0] == batch_size, f'mask.shape = {mask.shape}'

        in_map = self.conv(x)
        down1 = self.enc1(in_map)
        down2 = self.enc2(down1)

        hidden = nn.Sequential(nn.AvgPool2d(7), nn.GELU())(down2)
        up1 = self.bridge(hidden)

        c_one_hot = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        mask = mask[:, None].repeat(1, self.n_classes)
        c_one_hot = c_one_hot * (1 - mask)

        t = torch.unsqueeze(t, 1)
        cemb1 = self.cls_emb1(c_one_hot.float()).view(-1, self.dims * 2, 1, 1)
        temb1 = self.time_emb1(t.float()).view(-1, self.dims * 2, 1, 1)
        cemb2 = self.cls_emb2(c_one_hot.float()).view(-1, self.dims, 1, 1)
        temb2 = self.time_emb2(t.float()).view(-1, self.dims, 1, 1)

        up2 = self.dec1(cemb1 * up1 + temb1, down2)
        up3 = self.dec2(cemb2 * up2 + temb2, down1)
        out = self.output(torch.cat((up3, in_map), 1))
        return out
