"""Per-level projection of backbone multi-scale features into a common
token dimension, then concatenation into per-view image features Q_IMG
(paper Sec. 3.3, Fig. 3).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiScaleProjector(nn.Module):
    def __init__(self, in_dims: dict[str, int], out_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.level_keys = list(in_dims.keys())
        self.projs = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(in_dims[key], hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim),
                )
                for key in self.level_keys
            }
        )

    def forward(self, features: dict[str, Tensor]) -> tuple[Tensor, list[tuple[int, int]]]:
        """
        Args:
            features: dict mapping each level key (matching `in_dims` at
                construction) to a (N, C_level, H, W) tensor, where N is the
                per-view batch dimension.

        Returns:
            tokens: (N, sum(H*W for all levels), out_dim) -- Q_IMG.
            spatial_shapes: list of (H, W) per level, in `self.level_keys` order.
        """
        tokens_per_level = []
        spatial_shapes = []
        for key in self.level_keys:
            feat = features[key]
            _, _, h, w = feat.shape
            flat = feat.flatten(2).transpose(1, 2)  # (N, H*W, C)
            tokens_per_level.append(self.projs[key](flat))
            spatial_shapes.append((h, w))
        tokens = torch.cat(tokens_per_level, dim=1)
        return tokens, spatial_shapes
