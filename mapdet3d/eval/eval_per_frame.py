"""Per-frame evaluation harness: runs a model over a DataLoader and computes
class-agnostic 3D AP/AR at each requested IoU threshold, using the FINAL
decoder layer's predictions (paper: B3D_t := B3D_L, the last-layer output).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mapdet3d.eval.ap_ar import compute_ap_ar


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    iou_thresholds: list[float],
    device: torch.device | None = None,
    num_samples_iou: int = 20000,
) -> dict[float, dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    frame_preds, frame_gts = [], []
    for features, rho, targets in dataloader:
        features = {k: v.to(device) for k, v in features.items()}
        rho = rho.to(device)

        out = model(features, rho)
        last = out["layer_preds"][-1]
        scores = torch.sigmoid(last["logits"]).squeeze(-1)  # (B, M)

        for b in range(len(targets)):
            frame_preds.append(
                {
                    "center": last["center"][b].cpu(),
                    "dims": last["dims"][b].cpu(),
                    "rot": last["rot"][b].cpu(),
                    "scores": scores[b].cpu(),
                }
            )
            frame_gts.append({k: v.cpu() for k, v in targets[b].items()})

    return {
        thr: compute_ap_ar(frame_preds, frame_gts, thr, num_samples=num_samples_iou)
        for thr in iou_thresholds
    }
