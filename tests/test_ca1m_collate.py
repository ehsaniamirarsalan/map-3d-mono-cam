"""Tests collate_ca1m_batch against the real mapanything.utils.image.preprocess_inputs
(no model weights needed -- this is pure image resize/normalize logic)."""

import numpy as np
import pytest
import torch

pytest.importorskip("mapanything.utils.image")

from mapdet3d.data.ca1m.collate import collate_ca1m_batch  # noqa: E402


def _make_sample(window_size: int, img_hw=(64, 48), num_boxes: int = 1) -> dict:
    h, w = img_hw
    views = []
    for _ in range(window_size):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        K = np.array([[50.0, 0.0, w / 2], [0.0, 50.0, h / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
        views.append({"img": img, "intrinsics": K})

    target = {
        "boxes2d": torch.rand(num_boxes, 4),
        "center": torch.randn(num_boxes, 3),
        "dims": torch.rand(num_boxes, 3) + 0.5,
        "rot": torch.eye(3).unsqueeze(0).expand(num_boxes, 3, 3).clone(),
    }
    return {"views": views, "targets": [{k:v.clone() for k,v in target.items()} for _ in range(window_size)]}


def test_collate_produces_batched_views_and_unbatched_targets():
    batch_size, window_size = 3, 4
    batch = [_make_sample(window_size) for _ in range(batch_size)]

    views, targets = collate_ca1m_batch(batch)

    assert len(views) == window_size
    for view in views:
        assert view["img"].shape[0] == batch_size
        assert view["img"].ndim == 4  # (B, C, H, W)
        assert view["intrinsics"].shape == (batch_size, 3, 3)

    assert len(targets) == batch_size * window_size


def test_collate_rejects_mismatched_window_sizes():
    batch = [_make_sample(4), _make_sample(5)]
    with pytest.raises(ValueError):
        collate_ca1m_batch(batch)
