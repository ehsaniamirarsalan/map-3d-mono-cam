"""Wires the multi-scale projection, dense anchor proposals, proposal head,
and deformable decoder into the full detection-transformer pipeline (paper
Sec. 3.3, Fig. 3, Eq. 2): Phi_DET(F) -> {(Q_k, B2D_k)}_{k=0}^{L}.

Per-layer 3D box heads (Phi_3D^k, one per k, since the paper's notation is
layer-specific) are intentionally NOT part of this module -- they belong to
the top-level model (mapdet3d/models/mapdet3d.py), which applies a separate
Box3DHead to each (query_k, ref_boxes_k) pair returned here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mapdet3d.detector.anchor_generator import generate_anchors
from mapdet3d.detector.decoder import DeformableDecoder
from mapdet3d.detector.multiscale_proj import MultiScaleProjector
from mapdet3d.detector.proposal_head import ProposalHead, select_top_m


class DetrDetector(nn.Module):
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
        assert len(base_sizes) == len(in_dims), "one base anchor size required per feature level"
        self.num_queries = num_queries
        self.base_sizes = base_sizes

        self.proj = MultiScaleProjector(in_dims, out_dim=d_model)
        self.proposal_head = ProposalHead(in_dim=d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.decoder = DeformableDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_levels=len(in_dims),
            n_points=n_points,
            d_ffn=d_ffn,
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor | list]:
        """
        Args:
            features: dict of per-view multi-scale feature maps, keys matching
                the `in_dims` passed at construction (paper's F = {F_E, F_7,
                F_11, F_15}), each (N, C_level, H, W).

        Returns:
            dict with:
              "proposal_logits": (N, num_anchors, 1) -- dense proposal scores,
                  supervised directly (paper's "deep supervision ... to the
                  encoder proposal layer").
              "proposal_boxes": (N, num_anchors, 4) -- dense proposal boxes.
              "layer_outputs": list of (query_k, ref_boxes_k) pairs, length
                  num_decoder_layers + 1 (k=0 is the initial top-M selection).
        """
        tokens, spatial_shapes = self.proj(features)
        anchors = generate_anchors(
            spatial_shapes, self.base_sizes, device=tokens.device, dtype=tokens.dtype
        )

        proposal_logits, proposal_boxes = self.proposal_head(tokens, anchors)
        query_feats, ref_boxes0, indices = select_top_m(tokens, proposal_logits, proposal_boxes, self.num_queries)
        query0 = self.query_proj(query_feats)

        layer_outputs = self.decoder(query0, ref_boxes0, tokens, spatial_shapes)

        return {
            "proposal_logits": proposal_logits,
            "proposal_boxes": proposal_boxes,
            "selected_logits": torch.gather(proposal_logits, 1, indices[..., None]),
            "selected_boxes": torch.gather(proposal_boxes, 1, indices[..., None].expand(-1, -1, 4)),
            "layer_outputs": layer_outputs,
        }
