"""Batches CA1MWindowDataset samples into MapAnything's `forward(views)`
input format, using MapAnything's own `preprocess_inputs` (resize/normalize)
rather than reimplementing its aspect-ratio-bucket logic (plan §6: reuse
MapAnything wherever possible instead of duplicating its internals).

Each sample's window is preprocessed independently (aspect-ratio bucketing
is a function of that window's own images), then concatenated across the
batch dimension per view slot. This assumes every sample in a batch shares
the same window length and resolution/aspect ratio -- true for CA-1M, whose
"wide" stream uses a fixed camera resolution throughout.
"""

from __future__ import annotations

import torch


def collate_ca1m_batch(batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Args:
        batch: list of {"views": [{"img", "intrinsics"}, ...], "target": {...}}
            samples, all with the same number of views (window_size).

    Returns:
        views: list of length T (window_size), each a dict with "img"
            (B, 3, H, W), "intrinsics" (B, 3, 3), and "data_norm_type" --
            ready for `MapAnythingBackbone.forward`.
        targets: list of length B target dicts (unbatched, variable N per frame).
    """
    from mapanything.utils.image import preprocess_inputs

    window_size = len(batch[0]["views"])
    assert all(len(sample["views"]) == window_size for sample in batch), (
        "all samples in a batch must share the same window size"
    )

    processed_per_sample = [preprocess_inputs(sample["views"]) for sample in batch]

    views = []
    for t in range(window_size):
        img_t = torch.cat([processed_per_sample[b][t]["img"] for b in range(len(batch))], dim=0)
        intrinsics_t = torch.cat(
            [processed_per_sample[b][t]["intrinsics"] for b in range(len(batch))], dim=0
        )
        views.append(
            {
                "img": img_t,
                "data_norm_type": processed_per_sample[0][t]["data_norm_type"],
                "intrinsics": intrinsics_t,
            }
        )

    targets = [sample["target"] for sample in batch]
    return views, targets
