"""3D box corner geometry shared by losses and evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

# Corner ordering: 8 corners of a unit box centered at origin, axes aligned
# with (x=length/2, y=width/2, z=height/2) before rotation. Order is fixed
# and must stay identical between prediction and GT corner construction so
# that L1 / Chamfer distances between corner sets are meaningful.
_UNIT_CORNERS = torch.tensor(
    [
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ]
)


def corners_from_box(center: Tensor, dims: Tensor, rot: Tensor) -> Tensor:
    """Compute the 8 corners of oriented 3D boxes.

    Args:
        center: (..., 3) box centers in camera/world coordinates.
        dims: (..., 3) box (width, length, height) or (dx, dy, dz) extents.
        rot: (..., 3, 3) rotation matrices mapping box-local axes to the
            same frame as `center`.

    Returns:
        (..., 8, 3) corner coordinates, in the fixed `_UNIT_CORNERS` order.
    """
    unit = _UNIT_CORNERS.to(device=center.device, dtype=center.dtype)
    local = unit.unsqueeze(0) * dims.unsqueeze(-2)  # (..., 8, 3)
    rotated = torch.einsum("...ij,...kj->...ki", rot, local)  # (..., 8, 3)
    return rotated + center.unsqueeze(-2)


def box_volume(dims: Tensor) -> Tensor:
    """Volume of oriented boxes given their (w, l, h)-style extents."""
    return dims[..., 0] * dims[..., 1] * dims[..., 2]
