import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.proj1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2 and x.shape[1] == self.in_features, f'{x.shape}'
        return self.proj2(self.act(self.proj1(x)))
