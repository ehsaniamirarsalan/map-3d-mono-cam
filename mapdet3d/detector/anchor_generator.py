"""Dense 2D anchor proposals on a regular grid per feature level, with
scale-dependent default sizes (paper Sec. 3.3).
"""

from __future__ import annotations

import torch
from torch import Tensor


def generate_anchors(
    spatial_shapes: list[tuple[int, int]],
    base_sizes: list[float],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Args:
        spatial_shapes: list of (H, W) per level.
        base_sizes: default (square) anchor side length per level, normalized
            to [0, 1] image coordinates. Must be the same length as
            `spatial_shapes`, one size per level (coarser levels typically
            get larger anchors).

    Returns:
        (sum(H*W for all levels), 4) anchors in normalized cxcywh, ordered
        level-major then row-major within each level (matching
        `MultiScaleProjector`'s token concatenation order).
    """
    assert len(spatial_shapes) == len(base_sizes)
    anchors = []
    for (h, w), size in zip(spatial_shapes, base_sizes):
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cx = grid_x.reshape(-1)
        cy = grid_y.reshape(-1)
        wh = torch.full_like(cx, size)
        anchors.append(torch.stack([cx, cy, wh, wh], dim=-1))
    return torch.cat(anchors, dim=0)
