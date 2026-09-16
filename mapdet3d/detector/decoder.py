"""Stack of deformable decoder layers, producing the paper's
{(Q_k, B2D_k)}_{k=0}^{L} sequence (Eq. 2): k=0 is the initial (encoder
proposal) query/box pair, k=1..L are the outputs of each decoder layer.
"""

from __future__ import annotations

from torch import Tensor, nn

from mapdet3d.detector.decoder_layer import DeformableDecoderLayer


class DeformableDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        d_ffn: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeformableDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_levels=n_levels,
                    n_points=n_points,
                    d_ffn=d_ffn,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query0: Tensor,
        ref_boxes0: Tensor,
        value_input: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Args:
            query0: (N, M, d_model) initial object queries (from proposal
                top-M selection).
            ref_boxes0: (N, M, 4) initial reference boxes.
            value_input: (N, Len_in, d_model) -- Q_IMG.
            spatial_shapes: list of (h, w) per level.

        Returns:
            List of length `num_layers + 1` of (query_k, ref_boxes_k) pairs,
            index 0 being the (unrefined) input pair.
        """
        query, ref_boxes = query0, ref_boxes0
        outputs = [(query, ref_boxes)]
        for layer in self.layers:
            query, ref_boxes = layer(query, ref_boxes, value_input, spatial_shapes)
            outputs.append((query, ref_boxes))
            ref_boxes = ref_boxes.detach()
        return outputs
