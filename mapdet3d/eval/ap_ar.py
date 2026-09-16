"""Class-agnostic 3D AP/AR at a fixed cuboid-IoU threshold (paper Sec. 4.2):
matches ALL ground-truth boxes regardless of visibility/truncation (an
explicit stated deviation from prior benchmarks -- do not add any
visibility/truncation filtering here).
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapdet3d.eval.box3d_iou import box3d_iou_montecarlo

FramePred = dict[str, Tensor]  # "center"/"dims" (N,3), "rot" (N,3,3), "scores" (N,)
FrameGT = dict[str, Tensor]  # "center"/"dims" (M,3), "rot" (M,3,3)


def _greedy_match(
    frame_preds: list[FramePred], frame_gts: list[FrameGT], iou_threshold: float, num_samples: int
) -> tuple[Tensor, Tensor, int]:
    """Returns (tp, fp) boolean arrays aligned with predictions sorted by
    descending score across the whole dataset, plus the total GT count."""
    # Flatten predictions with (frame_idx, local_idx, score) for global score sorting.
    all_scores, all_frame_idx, all_local_idx = [], [], []
    for f, pred in enumerate(frame_preds):
        n = pred["center"].shape[0]
        all_scores.append(pred["scores"])
        all_frame_idx.extend([f] * n)
        all_local_idx.extend(range(n))
    if all_scores:
        all_scores = torch.cat(all_scores)
    else:
        all_scores = torch.zeros(0)

    order = torch.argsort(all_scores, descending=True, stable=True)

    # Precompute per-frame IoU matrices and a "matched" flag per GT box.
    iou_matrices = []
    gt_matched = []
    total_gt = 0
    for pred, gt in zip(frame_preds, frame_gts):
        n_gt = gt["center"].shape[0]
        total_gt += n_gt
        if pred["center"].shape[0] == 0 or n_gt == 0:
            iou_matrices.append(None)
        else:
            iou_matrices.append(
                box3d_iou_montecarlo(
                    pred["center"], pred["dims"], pred["rot"],
                    gt["center"], gt["dims"], gt["rot"],
                    num_samples=num_samples,
                )
            )
        gt_matched.append(torch.zeros(n_gt, dtype=torch.bool))

    num_preds = len(order)
    tp = torch.zeros(num_preds, dtype=torch.bool)
    fp = torch.zeros(num_preds, dtype=torch.bool)

    for rank, idx in enumerate(order.tolist()):
        f = all_frame_idx[idx]
        local_i = all_local_idx[idx]
        iou_mat = iou_matrices[f]
        if iou_mat is None:
            fp[rank] = True
            continue
        ious = iou_mat[local_i]
        # Only consider still-unmatched GT boxes in this frame.
        ious = ious.clone()
        ious[gt_matched[f]] = -1.0
        best_iou, best_gt = ious.max(dim=0)
        if best_iou.item() >= iou_threshold:
            tp[rank] = True
            gt_matched[f][best_gt] = True
        else:
            fp[rank] = True

    return tp, fp, total_gt


def _average_precision(precision: Tensor, recall: Tensor) -> float:
    """101-point interpolated AP (COCO/VOC2012-style continuous method)."""
    if recall.numel() == 0:
        return 0.0
    ap = 0.0
    for t in torch.linspace(0.0, 1.0, 101):
        mask = recall >= t
        p = precision[mask].max().item() if mask.any() else 0.0
        ap += p
    return ap / 101.0


def compute_ap_ar(
    frame_preds: list[FramePred],
    frame_gts: list[FrameGT],
    iou_threshold: float,
    num_samples: int = 20000,
) -> dict[str, float]:
    """
    Args:
        frame_preds: per-frame predicted boxes + scores.
        frame_gts: per-frame ground-truth boxes (ALL boxes, unfiltered).
        iou_threshold: cuboid-IoU threshold defining a true positive.
        num_samples: Monte Carlo samples per IoU pair (see box3d_iou.py).

    Returns:
        {"AP": float, "AR": float}
    """
    if len(frame_preds) != len(frame_gts):
        raise ValueError("Prediction and ground-truth frame counts differ")
    frame_preds = [{k: v.detach().cpu() for k,v in p.items()} for p in frame_preds]
    frame_gts = [{k: v.detach().cpu() for k,v in p.items()} for p in frame_gts]
    tp, fp, total_gt = _greedy_match(frame_preds, frame_gts, iou_threshold, num_samples)

    if total_gt == 0:
        return {"AP": 0.0, "AR": 0.0}

    tp_cumsum = torch.cumsum(tp.float(), dim=0)
    fp_cumsum = torch.cumsum(fp.float(), dim=0)

    recall = tp_cumsum / total_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-9)

    ap = _average_precision(precision, recall)
    ar = recall[-1].item() if recall.numel() > 0 else 0.0

    return {"AP": ap, "AR": ar}
