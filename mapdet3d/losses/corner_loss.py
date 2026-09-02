"""Disentangled 3D corner loss (Cube R-CNN style).

For each of the four geometric attribute groups (xy-center, z-depth,
dimensions, rotation), construct a "disentangled" box that takes the
prediction for exactly that one attribute and ground truth for all others,
then compares its 8 corners against the full ground-truth box's corners.
Because only one attribute differs between the two corner sets, the
resulting distance isolates the error contributed by that attribute alone.

Rotation uses a (bidirectional) Chamfer distance between corner sets instead
of L1, since a cuboid's corners are invariant under certain rotations (e.g.
180 degrees about the vertical axis for any box, or 90 degrees for a
square-cross-section box) — plain L1 on ordered corners would wrongly
penalize these symmetric rotations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapdet3d.utils.geometry import corners_from_box


def corner_l1_loss(corners_a: Tensor, corners_b: Tensor) -> Tensor:
    """Mean L1 distance between two equally-ordered (..., 8, 3) corner sets."""
    return (corners_a - corners_b).abs().mean(dim=(-2, -1))


def chamfer_distance(corners_a: Tensor, corners_b: Tensor) -> Tensor:
    """Bidirectional Chamfer distance between two (..., 8, 3) point sets.

    Order-invariant: does not assume matching corner indices, so it is
    unaffected by rotational symmetries that permute the corner ordering.
    """
    dist = torch.cdist(corners_a, corners_b)  # (..., 8, 8)
    a_to_b = dist.min(dim=-1).values.mean(dim=-1)
    b_to_a = dist.min(dim=-2).values.mean(dim=-1)
    return a_to_b + b_to_a


def disentangled_corner_losses(
    pred_center: Tensor,
    pred_dims: Tensor,
    pred_rot: Tensor,
    gt_center: Tensor,
    gt_dims: Tensor,
    gt_rot: Tensor,
) -> dict[str, Tensor]:
    """Compute the four disentangled corner-loss terms.

    All inputs are (N, 3) / (N, 3, 3) for N matched (prediction, GT) pairs.

    Returns:
        dict with per-pair (shape (N,)) losses: "xy", "z", "dim" (L1) and
        "rot" (Chamfer distance).
    """
    gt_corners = corners_from_box(gt_center, gt_dims, gt_rot)

    xy_center = torch.cat([pred_center[..., :2], gt_center[..., 2:3]], dim=-1)
    xy_corners = corners_from_box(xy_center, gt_dims, gt_rot)
    loss_xy = corner_l1_loss(xy_corners, gt_corners)

    z_center = torch.cat([gt_center[..., :2], pred_center[..., 2:3]], dim=-1)
    z_corners = corners_from_box(z_center, gt_dims, gt_rot)
    loss_z = corner_l1_loss(z_corners, gt_corners)

    dim_corners = corners_from_box(gt_center, pred_dims, gt_rot)
    loss_dim = corner_l1_loss(dim_corners, gt_corners)

    rot_corners = corners_from_box(gt_center, gt_dims, pred_rot)
    loss_rot = chamfer_distance(rot_corners, gt_corners)

    return {"xy": loss_xy, "z": loss_z, "dim": loss_dim, "rot": loss_rot}
