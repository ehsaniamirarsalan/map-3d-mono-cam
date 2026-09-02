"""Wraps MapAnything to expose the paper's multi-scale features {F_E, F_7,
F_11, F_15} and metric scale factor rho, none of which MapAnything's public
`forward()`/`infer()` return (they only return final decoded outputs like
points, poses, and the metric scaling factor per view).

Confirmed by direct inspection of the vendored `third_party/map-anything`
source (pinned commit) and the `uniception` package it depends on:

- `self.model.encoder` is called once with all views concatenated along the
  batch dimension and returns a `ViTEncoderOutput` with `.features`
  (batch*num_views, C, H, W); MapAnything itself splits this via
  `.chunk(num_views, dim=0)` into the paper's F_E (one tensor per view)
  (`mapanything/models/mapanything/model.py::_encode_n_views`).
- `self.model.info_sharing` is the 16-layer
  `MultiViewAlternatingAttentionTransformerIFR`, configured with
  `indices=[7, 11]`. Its forward returns
  `(final_output, intermediate_outputs)`, both `MultiViewTransformerOutput`
  dataclasses with `.features` (list of per-view tensors) and
  `.additional_token_features` (the scale token's transformer-updated
  hidden state, i.e. the paper's pre-scale-head rho input).
  `intermediate_outputs[0].features` == F_7, `intermediate_outputs[1].features`
  == F_11, `final_output.features` == F_15.

Rather than duplicating MapAnything's large `forward()` method, we register
forward hooks on these two submodules and capture their return values
directly during a normal `self.model.forward(views)` call.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MapAnythingBackbone(nn.Module):
    def __init__(
        self,
        pretrained: str = "facebook/map-anything-apache",
        freeze_encoder: bool = True,
    ):
        super().__init__()
        from mapanything.models import MapAnything  # heavy optional dependency

        self.model = MapAnything.from_pretrained(pretrained)
        self._captured: dict = {}
        self.model.encoder.register_forward_hook(self._capture("encoder_output"))
        self.model.info_sharing.register_forward_hook(self._capture("info_sharing_output"))

        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad_(False)

    def _capture(self, key: str):
        def _hook(module, inputs, output):
            self._captured[key] = output

        return _hook

    def forward(self, views: list[dict]) -> dict[str, Tensor | list[Tensor]]:
        """
        Args:
            views: list of T per-view dicts in MapAnything's `forward()` input
                format (each with "img" and "data_norm_type", optionally
                "ray_directions_cam", "camera_pose_quats"/"camera_pose_trans" —
                see `mapanything/utils/inference.py` for the full contract).

        Returns:
            dict with:
              "F_E":  List[T] of (B, C_enc, H, W) — raw per-view encoder features.
              "F_7":  List[T] of (B, C, H, W) — layer-7 multi-view transformer features.
              "F_11": List[T] of (B, C, H, W) — layer-11 multi-view transformer features.
              "F_15": List[T] of (B, C, H, W) — final (layer-16) transformer features.
              "rho":  (B,) metric scale factor, from MapAnything's own scale head.
              "scale_token_features": (B, C, 1) pre-head scale token hidden state.
        """
        self._captured.clear()
        num_views = len(views)
        per_view_outputs = self.model.forward(views)

        encoder_output = self._captured["encoder_output"]
        f_e = list(encoder_output.features.chunk(num_views, dim=0))

        final_output, intermediate_outputs = self._captured["info_sharing_output"]

        return {
            "F_E": f_e,
            "F_7": intermediate_outputs[0].features,
            "F_11": intermediate_outputs[1].features,
            "F_15": final_output.features,
            # Real MapAnything returns this as (B, 1); flatten to (B,) to
            # match this wrapper's documented contract and the rest of the
            # pipeline (mapdet3d.models.mapdet3d.MapDet3D expects (B,)).
            "rho": per_view_outputs[0]["metric_scaling_factor"].reshape(-1),
            "scale_token_features": final_output.additional_token_features,
        }
