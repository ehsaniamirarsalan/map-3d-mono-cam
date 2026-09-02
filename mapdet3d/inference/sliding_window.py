"""Online causal sliding-window inference (paper Sec. 3.5): at time t, feed
the current frame plus the previous T-1 frames as multi-view input, but
output detections only for the current (most recent) frame.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapdet3d.models.mapdet3d import MapDet3D


class SlidingWindowInference:
    def __init__(self, model: MapDet3D, window_size: int):
        self.model = model
        self.window_size = window_size
        self.buffer: list[dict] = []

    def reset(self) -> None:
        self.buffer = []

    @torch.no_grad()
    def step(self, view: dict) -> dict[str, Tensor]:
        """
        Args:
            view: a single new frame's per-view dict (MapAnything's `forward()`
                input format -- see MapAnythingBackbone).

        Returns:
            The final-layer predictions (LayerPred dict: "logits", "boxes2d",
            "center", "dims", "rot") for the CURRENT frame only.
        """
        self.buffer.append(view)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size :]

        self.model.eval()
        out = self.model(self.buffer)

        num_views = len(self.buffer)
        last_pred = out["layer_preds"][-1]
        batch_size = last_pred["logits"].shape[0] // num_views
        # Features are concatenated view-major (view 0's batch block, then
        # view 1's, ...), so the current (last) view's block is the final
        # `batch_size` rows.
        return {k: v[-batch_size:] for k, v in last_pred.items()}
