"""2D box format conversion, GIoU, and focal loss used for DETR-style matching.

Matching in Map-Det3D is explicitly 2D-only (paper Sec. 3.6): the Hungarian
matcher and these losses must never receive 3D cost terms.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_area(boxes_xyxy: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def box_iou(boxes1_xyxy: Tensor, boxes2_xyxy: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU. boxes1: (N, 4), boxes2: (M, 4) -> iou (N, M), union (N, M)."""
    area1 = box_area(boxes1_xyxy)
    area2 = box_area(boxes2_xyxy)

    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    return iou, union


def generalized_box_iou(boxes1_cxcywh: Tensor, boxes2_cxcywh: Tensor) -> Tensor:
    """Pairwise GIoU between normalized cxcywh boxes. Returns (N, M)."""
    boxes1 = box_cxcywh_to_xyxy(boxes1_cxcywh)
    boxes2 = box_cxcywh_to_xyxy(boxes2_cxcywh)

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing_area = wh[..., 0] * wh[..., 1]

    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-7)


def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    """Elementwise focal loss (Lin et al. 2017). No reduction applied."""
    prob = torch.sigmoid(logits)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss
