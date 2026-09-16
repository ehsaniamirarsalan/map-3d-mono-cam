"""Combined training criterion: 2D-only Hungarian matching, focal + L1 + GIoU
for the auxiliary 2D boxes, and the disentangled corner loss for 3D geometry.

Deep supervision (paper Sec. 3.6): the total per-frame loss sums a per-layer
loss (`layer_loss`) over the encoder-proposal layer and every decoder layer.
This module exposes `layer_loss` (single layer) and `SetCriterion.forward`
(sums it over a list of per-layer prediction dicts).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mapdet3d.losses.corner_loss import disentangled_corner_losses
from mapdet3d.losses.loss_2d import generalized_box_iou, sigmoid_focal_loss
from mapdet3d.losses.matcher import HungarianMatcher

# A single target (per image/frame): 2D boxes drive matching; 3D attributes
# supervise the disentangled corner loss for matched pairs only.
Target = dict[str, Tensor]  # keys: "boxes2d" (N,4) cxcywh, "center"/"dims" (N,3), "rot" (N,3,3)

# A single layer's predictions (per image/frame, batched over B).
LayerPred = dict[str, Tensor]  # keys: "logits" (B,M,1), "boxes2d" (B,M,4),
# "center"/"dims" (B,M,3), "rot" (B,M,3,3)


def layer_loss(
    pred: LayerPred,
    targets: list[Target],
    matcher: HungarianMatcher,
    weight_class: float = 1.0,
    weight_bbox: float = 5.0,
    weight_giou: float = 2.0,
    weight_3d: float = 1.0,
) -> dict[str, Tensor]:
    """Compute one decoder layer's (2D + 3D) losses, averaged over matched boxes."""
    bs, num_queries = pred["logits"].shape[:2]
    if len(targets) != bs:
        raise ValueError(f"Expected {bs} per-view targets, received {len(targets)}")
    device = pred["logits"].device

    target_boxes2d = [t["boxes2d"] for t in targets]
    indices = matcher(pred["logits"], pred["boxes2d"], target_boxes2d)

    num_matched = sum(len(idx[0]) for idx in indices)
    num_matched = max(num_matched, 1)

    # Classification: focal loss over ALL queries (positive at matched, else negative).
    class_targets = torch.zeros(bs, num_queries, 1, device=device)
    for b, (pred_idx, _) in enumerate(indices):
        class_targets[b, pred_idx, 0] = 1.0
    loss_class = sigmoid_focal_loss(pred["logits"], class_targets).sum() / num_matched

    # 2D + 3D losses on matched pairs only.
    loss_bbox_terms, loss_giou_terms = [], []
    corner_terms = {"xy": [], "z": [], "dim": [], "rot": []}
    for b, (pred_idx, gt_idx) in enumerate(indices):
        if pred_idx.numel() == 0:
            continue
        p_boxes2d = pred["boxes2d"][b, pred_idx]
        t_boxes2d = targets[b]["boxes2d"][gt_idx]
        loss_bbox_terms.append((p_boxes2d - t_boxes2d).abs().sum(dim=-1))
        loss_giou_terms.append(1.0 - generalized_box_iou(p_boxes2d, t_boxes2d).diag())

        corners = disentangled_corner_losses(
            pred["center"][b, pred_idx],
            pred["dims"][b, pred_idx],
            pred["rot"][b, pred_idx],
            targets[b]["center"][gt_idx],
            targets[b]["dims"][gt_idx],
            targets[b]["rot"][gt_idx],
        )
        for k, v in corners.items():
            corner_terms[k].append(v)

    def _sum_or_zero(terms: list[Tensor]) -> Tensor:
        if not terms:
            return torch.zeros((), device=device)
        return torch.cat(terms).sum()

    loss_bbox = _sum_or_zero(loss_bbox_terms) / num_matched
    loss_giou = _sum_or_zero(loss_giou_terms) / num_matched
    loss_xy = _sum_or_zero(corner_terms["xy"]) / num_matched
    loss_z = _sum_or_zero(corner_terms["z"]) / num_matched
    loss_dim = _sum_or_zero(corner_terms["dim"]) / num_matched
    loss_rot = _sum_or_zero(corner_terms["rot"]) / num_matched

    loss_2d = weight_class * loss_class + weight_bbox * loss_bbox + weight_giou * loss_giou
    loss_3d = weight_3d * (loss_xy + loss_z + loss_dim + loss_rot)

    return {
        "loss_class": loss_class,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
        "loss_xy": loss_xy,
        "loss_z": loss_z,
        "loss_dim": loss_dim,
        "loss_rot": loss_rot,
        "loss_2d": loss_2d,
        "loss_3d": loss_3d,
        "loss_total": loss_2d + loss_3d,
    }


class SetCriterion(nn.Module):
    """Sums `layer_loss` over all supervised layers (deep supervision, Eq. 4)."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        weight_3d: float = 1.0,
    ):
        super().__init__()
        self.matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.weight_class = cost_class
        self.weight_bbox = cost_bbox
        self.weight_giou = cost_giou
        self.weight_3d = weight_3d

    def forward(self, outputs: dict | list[LayerPred], targets: list[Target]) -> dict[str, Tensor]:
        # The encoder stage in layer_preds[0] uses connected proposal outputs.
        # Lists remain accepted for callers computing isolated layer losses.
        layer_preds = outputs["layer_preds"] if isinstance(outputs, dict) else outputs
        if not layer_preds:
            raise ValueError("At least one supervised stage is required")
        per_layer = [
            layer_loss(
                pred,
                targets,
                self.matcher,
                weight_class=self.weight_class,
                weight_bbox=self.weight_bbox,
                weight_giou=self.weight_giou,
                weight_3d=self.weight_3d,
            )
            for pred in layer_preds
        ]
        total = {
            k: torch.stack([layer[k] for layer in per_layer]).sum()
            for k in per_layer[0]
        }
        return total
