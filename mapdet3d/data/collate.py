"""Batches a list of (features, rho, target) samples. Targets are kept as a
list of variable-length dicts (standard DETR-style set-prediction batching)
since each frame has a different number of ground-truth boxes.
"""

from __future__ import annotations

import torch


def collate_feature_samples(batch: list[dict]) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict]]:
    features = {key: torch.stack([sample["features"][key] for sample in batch], dim=0) for key in batch[0]["features"]}
    rho = torch.tensor([sample["rho"] for sample in batch], dtype=torch.float32)
    targets = [sample["target"] for sample in batch]
    return features, rho, targets
