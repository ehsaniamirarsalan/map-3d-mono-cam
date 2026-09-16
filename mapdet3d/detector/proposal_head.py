"""Shared classification + regression head over encoder tokens, scoring dense
anchor proposals; top-M selection forms the initial object queries and
reference boxes (paper Sec. 3.3).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from mapdet3d.detector.box_ops import inverse_sigmoid


class ProposalHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.class_head = nn.Linear(in_dim, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, tokens: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            tokens: (N, num_anchors, in_dim) -- Q_IMG.
            anchors: (num_anchors, 4) normalized cxcywh, shared across the batch.

        Returns:
            logits: (N, num_anchors, 1) objectness logits.
            boxes: (N, num_anchors, 4) anchors refined by a regressed delta,
                clamped to valid normalized [0, 1] coordinates.
        """
        logits = self.class_head(tokens)
        deltas = self.bbox_head(tokens)
        boxes = (inverse_sigmoid(anchors.unsqueeze(0)) + deltas).sigmoid()
        return logits, boxes


def select_top_m(
    tokens: Tensor, logits: Tensor, boxes: Tensor, num_queries: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Select the top-M scoring anchor positions to initialize object queries.

    The returned query features are detached (stop-gradient, paper Sec.
    3.3: "corresponding encoder features, after stop-gradient ... form the
    initial object queries") — proposal-head gradients flow only through
    its own auxiliary 2D loss, not back through the decoder.

    Args:
        tokens: (N, num_anchors, in_dim).
        logits: (N, num_anchors, 1).
        boxes: (N, num_anchors, 4).
        num_queries: M, the fixed output box count.

    Returns:
        query_feats: (N, M, in_dim), detached.
        ref_boxes: (N, M, 4), detached.
        topk_idx: (N, M) indices into the anchor dimension.
    """
    if not 0 < num_queries <= tokens.shape[1]:
        raise ValueError("num_queries must be positive and no larger than the number of feature positions")
    scores = logits.squeeze(-1)
    topk_idx = scores.topk(num_queries, dim=1).indices
    query_feats = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    ref_boxes = torch.gather(boxes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    return query_feats.detach(), ref_boxes.detach(), topk_idx
