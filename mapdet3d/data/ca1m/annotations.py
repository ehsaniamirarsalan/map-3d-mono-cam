"""Converts CA-1M's `Instances3D` (from the vendored `ml-cubifyanything`
toolkit, `third_party/ml-cubifyanything`) into this project's target format:
{"boxes2d": (N,4) cxcywh normalized, "center": (N,3), "dims": (N,3), "rot": (N,3,3)}.

CA-1M's per-frame `wide/instances.json` boxes are already expressed in that
frame's own camera coordinates (the toolkit sets `wide.RT = eye(4)` and
documents these boxes as "independent of any pose" -- see
`cubifyanything/dataset.py::_map_sample`), so no extra camera transform is
needed here. `GeneralInstance3DBoxes.dims` follows Apple's local box-frame
axis convention (x=length, y=height, z=width, i.e. `.gravity_center`/`.dims`/
`.R` already correspond exactly to this project's (center, dims, rot)
before-rotation-then-translation convention in `mapdet3d/utils/geometry.py`)
-- no axis permutation is applied.

CA-1M provides no direct 2D box annotation for non-empty instance lists
(`cubifyanything/dataset.py::read_instances`, non-empty branch), so 2D boxes
are derived here by projecting each 3D box's 8 corners with the camera
intrinsics and taking their axis-aligned bounding rectangle.
"""

from __future__ import annotations

import torch
from torch import Tensor


def project_corners_to_2d_box(
    corners_cam: Tensor, K: Tensor, img_w: int, img_h: int
) -> Tensor:
    """Project one box's 8 camera-frame corners with pinhole intrinsics `K`
    and return the normalized cxcywh bounding rectangle of their projections,
    clipped to the image bounds.

    Args:
        corners_cam: (8, 3) corners in camera coordinates (z > 0 in front).
        K: (3, 3) camera intrinsics.
        img_w, img_h: image dimensions in pixels.

    Returns:
        (4,) normalized cxcywh box.
    """
    # Clip the convex box against a positive near plane before projection.
    # All corner pairs include the box edges; extra diagonal intersections
    # lie inside the clipped hull and cannot expand its projected bounds.
    near = 1e-4
    points = [corners_cam[corners_cam[:, 2] >= near]]
    for i in range(len(corners_cam)):
        for j in range(i):
            a, b = corners_cam[i], corners_cam[j]
            if bool((a[2] >= near) != (b[2] >= near)):
                points.append((a + (b-a) * ((near-a[2])/(b[2]-a[2])))[None])
    corners_cam = torch.cat(points)
    if not len(corners_cam):
        return K.new_zeros(4)  # Retain its 3D annotation; no image footprint.
    z = corners_cam[:, 2]
    uv = (corners_cam[:, :2] / z.unsqueeze(-1)) @ K[:2, :2].T + K[:2, 2]
    u = uv[:, 0].clamp(0, img_w)
    v = uv[:, 1].clamp(0, img_h)

    x0, x1 = u.min(), u.max()
    y0, y1 = v.min(), v.max()

    cx = (x0 + x1) / 2.0 / img_w
    cy = (y0 + y1) / 2.0 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return torch.stack([cx, cy, w, h])


def instances_to_target(instances, K: Tensor, img_w: int, img_h: int) -> dict[str, Tensor]:
    """
    Args:
        instances: a `cubifyanything.instances.Instances3D` with a
            `gt_boxes_3d` field (a `GeneralInstance3DBoxes`), as returned by
            CA-1M's `wide/instances`.
        K: (3, 3) camera intrinsics for this frame.
        img_w, img_h: image dimensions in pixels.

    Returns:
        dict with "boxes2d" (N,4), "center" (N,3), "dims" (N,3), "rot" (N,3,3).
        N may be 0 (frame with no annotated objects).
    """
    boxes_3d = instances.get("gt_boxes_3d")
    center = torch.as_tensor(boxes_3d.gravity_center, dtype=torch.float32)
    dims = torch.as_tensor(boxes_3d.dims, dtype=torch.float32)
    rot = torch.as_tensor(boxes_3d.R, dtype=torch.float32)

    n = center.shape[0]
    if n == 0:
        return {
            "boxes2d": torch.zeros(0, 4),
            "center": torch.zeros(0, 3),
            "dims": torch.zeros(0, 3),
            "rot": torch.zeros(0, 3, 3),
        }

    corners = torch.as_tensor(boxes_3d.corners, dtype=torch.float32)  # (N, 8, 3)
    boxes2d = torch.stack(
        [project_corners_to_2d_box(corners[i], K, img_w, img_h) for i in range(n)]
    )

    return {"boxes2d": boxes2d, "center": center, "dims": dims, "rot": rot}
