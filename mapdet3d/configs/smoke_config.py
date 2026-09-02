"""Shared model/feature configuration for the synthetic smoke-test pipeline,
used by both tools/train.py and tools/eval.py so they build an identical
model architecture (required for `--checkpoint` state_dict compatibility).
"""

from __future__ import annotations

from mapdet3d.models.mapdet3d import MapDet3DCore

FEATURE_SHAPES: dict[str, tuple[int, int, int]] = {
    "F_E": (32, 8, 8),
    "F_7": (32, 8, 8),
    "F_11": (32, 4, 4),
    "F_15": (32, 4, 4),
}


def build_model(num_queries: int, num_decoder_layers: int) -> MapDet3DCore:
    return MapDet3DCore(
        in_dims={k: shape[0] for k, shape in FEATURE_SHAPES.items()},
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        base_sizes=[0.4, 0.3, 0.2, 0.1],
        d_model=64,
        n_heads=4,
        n_points=4,
        d_ffn=128,
    )
