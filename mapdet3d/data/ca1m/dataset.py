"""Sliding-window training sample generator over the real CA-1M dataset,
wrapping Apple's `CubifyAnythingDataset` (vendored at
`third_party/ml-cubifyanything`) rather than re-parsing its WebDataset tar
format from scratch (see the plan's data-pipeline component notes).

`CubifyAnythingDataset` streams per-timestamp samples (potentially
interleaving multiple video captures depending on shard ordering), not
pre-grouped sliding windows, so this module buffers frames per video_id and
emits a T-frame training window each time a video's buffer has enough
frames -- mirroring the paper's causal sliding-window training setup
(Sec. 3.5): the window's LAST frame is always the one currently supervised.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from torch.utils.data import IterableDataset

from mapdet3d.data.ca1m.annotations import instances_to_target

_TOOLKIT_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "ml-cubifyanything"
if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))


class CA1MWindowDataset(IterableDataset):
    def __init__(
        self,
        source: str | list[str],
        window_size: int = 5,
        stride_range: tuple[int, int] = (2, 10),
        max_buffer_per_video: int = 64,
    ):
        """
        Args:
            source: a tar path/URI, list of tar paths/URIs, or a `.txt`
                link-list file (e.g. `data/train.txt` from the CA-1M repo),
                passed straight through to `CubifyAnythingDataset`.
            window_size: T, the number of views per training sample.
            stride_range: (min, max) frame stride sampled per window,
                matching the paper's randomized 2-10 FPS sampling augmentation.
            max_buffer_per_video: bounds per-video memory use; older frames
                are dropped once a video's buffer exceeds this size.
        """
        self.source = source
        self.window_size = window_size
        self.stride_range = stride_range
        self.max_buffer_per_video = max_buffer_per_video

    def _underlying(self):
        from cubifyanything.dataset import CubifyAnythingDataset

        return CubifyAnythingDataset(self.source, load_arkit_depth=False)

    def __iter__(self):
        buffers: dict[int, list] = {}
        for sample in self._underlying():
            if "wide" not in sample:
                continue  # skip timeless "world" instance samples

            video_id = sample["meta"]["video_id"]
            buf = buffers.setdefault(video_id, [])
            buf.append(sample)
            if len(buf) > self.max_buffer_per_video:
                buf.pop(0)

            if len(buf) < self.window_size:
                continue

            stride = random.randint(*self.stride_range)
            start = len(buf) - 1 - (self.window_size - 1) * stride
            if start < 0:
                stride = 1
                start = len(buf) - self.window_size

            indices = list(range(start, len(buf), stride))[-self.window_size :]
            yield self._build_sample([buf[i] for i in indices])

    def _build_sample(self, window: list[dict]) -> dict:
        views = []
        for s in window:
            img_chw = s["wide"]["image"][0]  # (C, H, W) uint8
            img_hwc = img_chw.permute(1, 2, 0).numpy()
            K = s["sensor_info"].wide.image.K[0]  # (3, 3)
            views.append({"img": img_hwc, "intrinsics": K})

        last_sample = window[-1]
        last_K = last_sample["sensor_info"].wide.image.K[0]
        img_h, img_w = last_sample["wide"]["image"].shape[-2:]
        target = instances_to_target(last_sample["wide"]["instances"], last_K, img_w, img_h)

        return {"views": views, "target": target}
