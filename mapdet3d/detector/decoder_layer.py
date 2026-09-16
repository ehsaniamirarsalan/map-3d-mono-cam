"""A single deformable decoder layer: self-attention among queries,
deformable cross-attention to the image tokens, and residual 2D box
refinement of the query's reference box (paper Sec. 3.3).

Box refinement here uses a simple additive-delta-then-clamp update rather
than the inverse-sigmoid parameterization some Deformable-DETR variants use;
functionally equivalent (differentiable, iterative refinement of a
normalized cxcywh box) and simpler to reason about/test.
"""

from __future__ import annotations

from torch import Tensor, nn

from mapdet3d.detector.ms_deform_attn import MSDeformAttn
from mapdet3d.detector.box_ops import inverse_sigmoid


class DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        d_ffn: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.query_position = nn.Sequential(nn.Linear(4, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.bbox_refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        # Zero-init the final refinement layer so a freshly constructed
        # layer starts as an identity box update (delta == 0).
        nn.init.constant_(self.bbox_refine[-1].weight, 0.0)
        nn.init.constant_(self.bbox_refine[-1].bias, 0.0)

    def forward(
        self,
        query: Tensor,
        ref_boxes: Tensor,
        value_input: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            query: (N, M, d_model) box queries.
            ref_boxes: (N, M, 4) normalized cxcywh reference boxes.
            value_input: (N, Len_in, d_model) -- Q_IMG.
            spatial_shapes: list of (h, w) per level.

        Returns:
            updated_query: (N, M, d_model).
            updated_ref_boxes: (N, M, 4), refined and clamped to [0, 1].
        """
        position = self.query_position(inverse_sigmoid(ref_boxes))
        cross_out = self.cross_attn(query + position, ref_boxes, value_input, spatial_shapes)
        query = self.norm1(query + cross_out)
        attn_out, _ = self.self_attn(query + position, query + position, query, need_weights=False)
        query = self.norm2(query + attn_out)

        ffn_out = self.ffn(query)
        query = self.norm3(query + ffn_out)

        delta = self.bbox_refine(query)
        updated_ref_boxes = (inverse_sigmoid(ref_boxes) + delta).sigmoid()

        return query, updated_ref_boxes
