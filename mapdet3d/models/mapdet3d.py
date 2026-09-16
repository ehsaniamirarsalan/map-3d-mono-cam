"""Top-level Map-Det3D model: backbone -> detection transformer -> per-layer
up-to-scale 3D box heads (paper Fig. 2-3).

Split into two classes so the detector + 3D-head assembly (`MapDet3DCore`)
is testable on synthetic multi-scale features without requiring the real
(multi-GB) pretrained MapAnything checkpoint; `MapDet3D` adds the real
backbone on top.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone
from mapdet3d.detector.detr_detector import DetrDetector
from mapdet3d.heads.box3d_head import Box3DHead

FEATURE_KEYS = ("F_E", "F_7", "F_11", "F_15")


class MapDet3DCore(nn.Module):
    def __init__(
        self,
        in_dims: dict[str, int],
        num_queries: int,
        num_decoder_layers: int,
        base_sizes: list[float],
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        d_ffn: int = 1024,
    ):
        super().__init__()
        self.detector = DetrDetector(
            in_dims=in_dims,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            base_sizes=base_sizes,
            d_model=d_model,
            n_heads=n_heads,
            n_points=n_points,
            d_ffn=d_ffn,
        )
        # One 3D box head per decoder stage (k=0..num_decoder_layers), per the
        # paper's layer-specific notation Phi_3D^k.
        self.box3d_heads = nn.ModuleList(
            [Box3DHead(in_dim=d_model) for _ in range(num_decoder_layers + 1)]
        )

    def forward(self, features: dict[str, Tensor], rho: Tensor) -> dict[str, Tensor | list]:
        """
        Args:
            features: dict of per-view-batched multi-scale features, keys
                matching `in_dims` (each (N, C_level, H, W), N = num_views *
                batch_size, view-major concatenation order).
            rho: (N,) per-query-batch-element metric scale factor, already
                expanded/repeated to match `features`' leading dimension.

        Returns:
            dict with:
              "layer_preds": list of LayerPred dicts (see
                  mapdet3d.losses.criterion), one per decoder stage.
              "proposal_logits", "proposal_boxes": encoder-proposal outputs.
        """
        det_out = self.detector(features)

        layer_preds = []
        for k, (head, (query_k, ref_boxes_k)) in enumerate(zip(self.box3d_heads, det_out["layer_outputs"])):
            rho_per_box = rho.view(-1, 1).expand(-1, query_k.shape[1])
            box3d_out = head(query_k, rho_per_box)
            layer_preds.append(
                {
                    "logits": det_out["selected_logits"] if k == 0 else box3d_out["conf"],
                    "boxes2d": det_out["selected_boxes"] if k == 0 else ref_boxes_k,
                    "center": box3d_out["center"],
                    "dims": box3d_out["dims"],
                    "rot": box3d_out["rot"],
                }
            )

        return {
            "layer_preds": layer_preds,
            "proposal_logits": det_out["proposal_logits"],
            "proposal_boxes": det_out["proposal_boxes"],
        }


class MapDet3D(nn.Module):
    def __init__(self, backbone: MapAnythingBackbone, core: MapDet3DCore):
        super().__init__()
        self.backbone = backbone
        self.core = core

    def forward(self, views: list[dict]) -> dict[str, Tensor | list]:
        backbone_out = self.backbone(views)
        num_views = len(views)

        features = {key: torch.cat(backbone_out[key], dim=0) for key in FEATURE_KEYS}
        # backbone_out["rho"] is (B,), one value per window, shared across all
        # T views; repeat view-major to match `features`' (T*B, ...) ordering.
        rho = backbone_out["rho"].repeat(num_views)

        out = self.core(features, rho)
        out["rho"] = backbone_out["rho"]
        return out
