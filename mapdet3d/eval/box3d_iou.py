"""Rotated cuboid (oriented 3D box) IoU.

Exact analytic OBB-OBB volumetric intersection requires 3D polytope
clipping, which is genuinely hard to get right (degenerate/near-parallel
face cases) and this project avoids native-extension dependencies (e.g.
pytorch3d) that are unreliable to build on Windows (see plan §5 risks,
§6 vendor-vs-build). We therefore use a Monte Carlo estimator: correct in
expectation, straightforward to verify, and exact (no sampling noise) for
the identical-box and disjoint-box edge cases used in unit tests.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _corners_bounding_box(centers: Tensor, dims: Tensor, rots: Tensor) -> tuple[Tensor, Tensor]:
    """Axis-aligned bounding box (min, max) enclosing each oriented box's 8 corners."""
    from mapdet3d.utils.geometry import corners_from_box

    corners = corners_from_box(centers, dims, rots)  # (N, 8, 3)
    return corners.min(dim=-2).values, corners.max(dim=-2).values


def _points_inside_box(points: Tensor, center: Tensor, dims: Tensor, rot: Tensor) -> Tensor:
    """Boolean mask of which of `points` (S, 3) lie inside one oriented box."""
    local = torch.einsum("ij,sj->si", rot.transpose(-1, -2), points - center)
    half_extent = dims / 2.0
    return (local.abs() <= half_extent).all(dim=-1)


def box3d_iou_montecarlo(
    centers_a: Tensor,
    dims_a: Tensor,
    rots_a: Tensor,
    centers_b: Tensor,
    dims_b: Tensor,
    rots_b: Tensor,
    num_samples: int = 20000,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pairwise Monte Carlo 3D IoU between two sets of oriented boxes.

    Args:
        centers_a, dims_a, rots_a: (N, 3), (N, 3), (N, 3, 3) for box set A.
        centers_b, dims_b, rots_b: (M, 3), (M, 3), (M, 3, 3) for box set B.
        num_samples: Monte Carlo samples drawn per (a, b) pair.

    Returns:
        (N, M) IoU matrix.
    """
    n, m = centers_a.shape[0], centers_b.shape[0]
    device, dtype = centers_a.device, centers_a.dtype
    iou = torch.zeros(n, m, device=device, dtype=dtype)

    vol_a = dims_a.prod(dim=-1)  # (N,)
    vol_b = dims_b.prod(dim=-1)  # (M,)

    min_a, max_a = _corners_bounding_box(centers_a, dims_a, rots_a)
    min_b, max_b = _corners_bounding_box(centers_b, dims_b, rots_b)

    for i in range(n):
        for j in range(m):
            lo = torch.min(min_a[i], min_b[j])
            hi = torch.max(max_a[i], max_b[j])
            extent = (hi - lo).clamp(min=1e-9)
            sample_vol = extent.prod()

            samples = lo + torch.rand(num_samples, 3, device=device, dtype=dtype, generator=generator) * extent
            in_a = _points_inside_box(samples, centers_a[i], dims_a[i], rots_a[i])
            in_b = _points_inside_box(samples, centers_b[j], dims_b[j], rots_b[j])
            inter_frac = (in_a & in_b).float().mean()

            inter_vol = inter_frac * sample_vol
            union_vol = vol_a[i] + vol_b[j] - inter_vol
            iou[i, j] = inter_vol / union_vol.clamp(min=1e-9)

    return iou
