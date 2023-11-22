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
        self, x: torch.Tensor, c: torch.LongTensor, t: torch.LongTensor,
        context_mask: torch.LongTensor
    ) -> torch.Tensor:
        assert x.shape[0] == c.shape[0], f'Context (target class) tensor shape mismatch {c.shape}'
        assert t.shape[0] == t.shape[0], f'Timestamp tensor shape mismatch {t.shape}'
        assert len(context_mask.shape) == 0, f'Incorrectn ask shape {context_mask.shape}'

        x = self.conv(x)

        down1 = self.enc1(x)
        down2 = self.enc2(down1)

        hidden = nn.Sequential(nn.AvgPool2d(7), nn.GELU())(down2)

        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        context_mask = torch.unsqueeze(context_mask, 0)[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1*(1-context_mask)) 
        c = c * context_mask

        t = torch.unsqueeze(t, 1)
        cemb1 = self.cls_emb1(c.float()).view(-1, self.dims * 2, 1, 1)
        temb1 = self.time_emb1(t.float()).view(-1, self.dims * 2, 1, 1)
        cemb2 = self.cls_emb2(c.float()).view(-1, self.dims, 1, 1)
        temb2 = self.time_emb2(t.float()).view(-1, self.dims, 1, 1)

        up1 = self.bridge(hidden)

        up2 = self.dec1(cemb1*up1+ temb1, down2)
        up3 = self.dec2(cemb2*up2+ temb2, down1)

        out = self.output(torch.cat((up3, x), 1))
        return out
