"""Hungarian bipartite matcher for DETR-style set prediction.

Matching cost is computed from 2D box predictions ONLY (classification +
L1 + GIoU on the auxiliary 2D boxes), per paper Sec. 3.6. 3D quantities must
never enter the matching cost.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from mapdet3d.losses.loss_2d import generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        target_boxes: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Args:
            pred_logits: (B, M, 1) objectness logits (class-agnostic).
            pred_boxes: (B, M, 4) predicted boxes, normalized cxcywh.
            target_boxes: list of length B, each (N_b, 4) normalized cxcywh
                ground-truth boxes for that batch element (N_b may be 0).

        Returns:
            List of length B of (pred_idx, gt_idx) index tensors giving the
            matched pairs for each batch element.
        """
        bs, num_queries = pred_logits.shape[:2]
        out_prob = pred_logits.sigmoid()  # (B, M, 1)

        indices = []
        for b in range(bs):
            tgt = target_boxes[b]
            if tgt.numel() == 0:
                indices.append(
                    (
                        torch.as_tensor([], dtype=torch.int64),
                        torch.as_tensor([], dtype=torch.int64),
                    )
                )
                continue

            prob_b = out_prob[b]  # (M, 1), single class -> cost is -prob
            cost_class = -prob_b.expand(-1, tgt.shape[0])  # (M, N_b)

            pred_boxes_b = pred_boxes[b]  # (M, 4)
            cost_bbox = torch.cdist(pred_boxes_b, tgt, p=1)  # (M, N_b)
            cost_giou = -generalized_box_iou(pred_boxes_b, tgt)  # (M, N_b)

            cost = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            cost = cost.cpu()

            row_idx, col_idx = linear_sum_assignment(cost.numpy())
            indices.append(
                (torch.as_tensor(row_idx, dtype=torch.int64), torch.as_tensor(col_idx, dtype=torch.int64))
            )
        return indices
