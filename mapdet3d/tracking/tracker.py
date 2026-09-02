"""Online tracking-by-detection (paper Sec. 4.5): maintains a track memory
in world coordinates, matches new per-frame detections against existing
tracks via greedy 3D-IoU assignment, and replaces a matched track's box
with the new observation's box if it is larger (larger box ~ more complete
view of the object as the camera moves, e.g. Fig. 4 in the paper).
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapdet3d.tracking.affinity import compute_affinity, transform_to_world
from mapdet3d.tracking.track import Track
from mapdet3d.utils.geometry import box_volume


class Tracker:
    def __init__(self, iou_threshold: float = 0.1, num_samples_iou: int = 5000):
        self.iou_threshold = iou_threshold
        self.num_samples_iou = num_samples_iou
        self.tracks: list[Track] = []
        self._next_id = 0

    def step(
        self,
        det_centers_cam: Tensor,
        det_dims: Tensor,
        det_rots_cam: Tensor,
        pose_cam2world: Tensor,
    ) -> list[Track]:
        """
        Args:
            det_centers_cam: (N, 3) new detections' centers, camera coords.
            det_dims: (N, 3).
            det_rots_cam: (N, 3, 3), camera coords.
            pose_cam2world: (4, 4) current frame's camera-to-world transform.

        Returns:
            The updated list of all tracks (including newly created ones).
        """
        det_centers, det_rots = transform_to_world(det_centers_cam, det_rots_cam, pose_cam2world)
        num_dets = det_centers.shape[0]

        if self.tracks:
            track_centers = torch.stack([t.center for t in self.tracks])
            track_dims = torch.stack([t.dims for t in self.tracks])
            track_rots = torch.stack([t.rot for t in self.tracks])
        else:
            track_centers = torch.zeros(0, 3)
            track_dims = torch.zeros(0, 3)
            track_rots = torch.zeros(0, 3, 3)

        affinity = compute_affinity(
            track_centers, track_dims, track_rots, det_centers, det_dims, det_rots,
            num_samples=self.num_samples_iou,
        )

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if affinity.numel() > 0:
            # Greedy: repeatedly take the highest remaining affinity pair.
            flat_order = torch.argsort(affinity.flatten(), descending=True)
            num_track_cols = affinity.shape[1]
            for flat_idx in flat_order.tolist():
                t_idx, d_idx = divmod(flat_idx, num_track_cols)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                if affinity[t_idx, d_idx].item() < self.iou_threshold:
                    break
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                self._update_track(t_idx, det_centers[d_idx], det_dims[d_idx], det_rots[d_idx])

        for d_idx in range(num_dets):
            if d_idx not in matched_dets:
                self._create_track(det_centers[d_idx], det_dims[d_idx], det_rots[d_idx])

        for track in self.tracks:
            track.age += 1

        return self.tracks

    def _update_track(self, track_idx: int, center: Tensor, dims: Tensor, rot: Tensor) -> None:
        track = self.tracks[track_idx]
        if box_volume(dims) > box_volume(track.dims):
            track.center, track.dims, track.rot = center, dims, rot

    def _create_track(self, center: Tensor, dims: Tensor, rot: Tensor) -> None:
        self.tracks.append(Track(id=self._next_id, center=center, dims=dims, rot=rot))
        self._next_id += 1
