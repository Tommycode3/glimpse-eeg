"""EEG encoder: temporal conv -> spatial conv -> projection to the CLIP space.

The layout follows the temporal-spatial convolution idea used by Song et al.
(2024), sized here for 17 channels at 100 Hz and any window length.
"""

import math

import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    def __init__(self, n_channels: int = 17, n_times: int = 100, emb_dim: int = 512,
                 n_filters: int = 40, dropout: float = 0.5):
        super().__init__()
        k_t = max(1, min(25, n_times // 3))
        length = n_times - k_t + 1
        n_bins = max(1, min(6, length))
        stride = max(1, length // n_bins)
        pool = length - stride * (n_bins - 1)  # gives exactly n_bins outputs
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, k_t)),
            nn.AvgPool2d((1, pool), (1, stride)),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.Conv2d(n_filters, n_filters, (n_channels, 1)),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        feat = n_filters * n_bins
        self.proj = nn.Sequential(nn.Linear(feat, emb_dim), _Residual(emb_dim, dropout), nn.LayerNorm(emb_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, emb_dim)."""
        h = self.conv(x.unsqueeze(1))
        return self.proj(h.flatten(1))


class _Residual(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.f = nn.Sequential(nn.GELU(), nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.f(x)
