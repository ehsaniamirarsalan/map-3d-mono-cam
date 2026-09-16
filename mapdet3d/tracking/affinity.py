"""Camera-to-world coordinate transform and 3D-IoU affinity matrix used for
the online tracking-by-detection algorithm (paper Sec. 4.5).
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapdet3d.eval.box3d_iou import box3d_iou_montecarlo


def transform_to_world(
    center_cam: Tensor, rot_cam: Tensor, pose_cam2world: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Args:
        center_cam: (N, 3) box centers in camera coordinates.
        rot_cam: (N, 3, 3) box rotations in camera coordinates.
        pose_cam2world: (4, 4) camera-to-world homogeneous transform.

    Returns:
        center_world: (N, 3).
        rot_world: (N, 3, 3).
    """
    R = pose_cam2world[:3, :3]
    t = pose_cam2world[:3, 3]
    center_world = torch.einsum("ij,nj->ni", R, center_cam) + t
    rot_world = torch.einsum("ij,njk->nik", R, rot_cam)
    return center_world, rot_world


def compute_affinity(
    track_centers: Tensor,
    track_dims: Tensor,
    track_rots: Tensor,
    det_centers: Tensor,
    det_dims: Tensor,
    det_rots: Tensor,
    num_samples: int = 5000,
) -> Tensor:
    """Pairwise 3D IoU affinity matrix between existing tracks and new
    detections, both already in world coordinates. Returns (num_tracks,
    num_dets); empty (0-sized) inputs produce an appropriately empty matrix.
    """
    if track_centers.shape[0] == 0 or det_centers.shape[0] == 0:
        return det_centers.new_zeros(track_centers.shape[0], det_centers.shape[0])
    return box3d_iou_montecarlo(
        track_centers, track_dims, track_rots, det_centers, det_dims, det_rots, num_samples=num_samples
    )
