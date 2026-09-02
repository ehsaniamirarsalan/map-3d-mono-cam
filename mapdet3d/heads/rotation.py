"""Rotation representation utilities.

Implements the continuous 6D rotation representation of Zhou et al. (2019,
"On the Continuity of Rotation Representations in Neural Networks") and the
allocentric-to-egocentric conversion of Kundu et al. (2018, 3D-RCNN), as used
by Cube R-CNN and Map-Det3D for the up-to-scale 3D box head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """Convert a 6D rotation representation to a 3x3 rotation matrix.

    Args:
        d6: (..., 6) tensor, the first 3 and last 3 components are treated
            as two (possibly non-orthogonal, non-unit) 3D vectors.

    Returns:
        (..., 3, 3) proper rotation matrices (right-handed, orthonormal).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Inverse of `rotation_6d_to_matrix`: take the first two columns.

    `rotation_6d_to_matrix` stacks (b1, b2, b3) as *columns*, i.e.
    `matrix[..., :, 0] == b1` and `matrix[..., :, 1] == b2`. The 6D
    representation is the concatenation `[b1, b2]` (each a 3-vector), so we
    must select and concatenate columns explicitly rather than reshape a
    (..., 3, 2) slice (whose row-major flatten interleaves b1/b2 components).
    """
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _ray_alignment_rotation(ray_dir: Tensor) -> Tensor:
    """Rotation matrix that rotates the +Z axis onto `ray_dir`.

    `ray_dir` (..., 3) must be a unit vector (the direction from the camera
    center to the object's predicted 3D center). Used to convert an
    allocentric rotation (defined relative to the viewing ray) into an
    egocentric rotation (defined relative to the camera's optical axis),
    following Kundu et al. (2018): R_ego = R_align(ray_dir) @ R_allo.

    Implemented via Rodrigues' rotation formula aligning the camera's
    forward axis e_z = (0, 0, 1) with `ray_dir`.
    """
    device, dtype = ray_dir.device, ray_dir.dtype
    ez = torch.zeros_like(ray_dir)
    ez[..., 2] = 1.0

    axis = torch.cross(ez, ray_dir, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True)
    cos_angle = (ez * ray_dir).sum(-1, keepdim=True).clamp(-1.0, 1.0)

    # Degenerate case: ray_dir parallel (or anti-parallel) to ez, so the
    # cross-product axis vanishes and Rodrigues' formula is undefined. Guard
    # 0/0 in axis normalization by substituting a dummy axis (result unused
    # for degenerate entries, overwritten below).
    is_degenerate = (axis_norm < 1e-8).squeeze(-1)
    safe_axis_norm = torch.where(axis_norm < 1e-8, torch.ones_like(axis_norm), axis_norm)
    axis_unit = axis / safe_axis_norm
    angle = torch.atan2(axis_norm.squeeze(-1), cos_angle.squeeze(-1))

    kx, ky, kz = axis_unit[..., 0], axis_unit[..., 1], axis_unit[..., 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zeros, -kz, ky], dim=-1),
            torch.stack([kz, zeros, -kx], dim=-1),
            torch.stack([-ky, kx, zeros], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=device, dtype=dtype).expand(*ray_dir.shape[:-1], 3, 3)
    sin_a = torch.sin(angle)[..., None, None]
    cos_a = torch.cos(angle)[..., None, None]
    R = eye + sin_a * K + (1.0 - cos_a) * (K @ K)

    if is_degenerate.any():
        # Parallel (cos_angle ~ +1): identity. Anti-parallel (cos_angle ~ -1):
        # a 180-degree rotation about any axis perpendicular to ez, e.g. the
        # x-axis, which maps (0,0,1) -> (0,0,-1).
        rot_180_about_x = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)
        ).expand(*ray_dir.shape[:-1], 3, 3)
        is_antiparallel = is_degenerate & (cos_angle.squeeze(-1) < 0.0)
        degenerate_fallback = torch.where(is_antiparallel[..., None, None], rot_180_about_x, eye)
        R = torch.where(is_degenerate[..., None, None], degenerate_fallback, R)
    return R


def allocentric_to_egocentric(rot_allo: Tensor, center: Tensor) -> Tensor:
    """Convert an allocentric rotation to egocentric using the ray to `center`.

    Args:
        rot_allo: (..., 3, 3) rotation matrices in allocentric (viewing-ray
            relative) form.
        center: (..., 3) predicted 3D box centers in camera coordinates
            (NOT ground truth) — the egocentric conversion must use the
            model's own predicted center direction.

    Returns:
        (..., 3, 3) rotation matrices in egocentric (camera-axis relative)
        form: R_ego = R_align(ray_dir) @ R_allo.
    """
    ray_dir = F.normalize(center, dim=-1)
    R_align = _ray_alignment_rotation(ray_dir)
    return R_align @ rot_allo
