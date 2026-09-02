"""Synthetic "fake CA-1M" dataset (plan §4 Phase 5 fallback): generates
procedural multi-object scenes directly in feature space, bypassing image
rendering and the real MapAnything backbone entirely.

This exercises the full training/eval/tracking infrastructure we control
end-to-end without requiring the real (multi-GB, license-gated) CA-1M/
ScanNet datasets or the real pretrained MapAnything checkpoint -- both are
external prerequisites outside this project's control (plan §2, §5 risks).
Real dataset integration plugs in at `mapdet3d/data/ca1m/` and
`mapdet3d/data/scannet/` behind the same (features, rho, targets) sample
interface used here, once those become available.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


def _rot_z(theta: float) -> list[list[float]]:
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


class SyntheticCuboidDataset(Dataset):
    """Each sample is a single synthetic frame: random multi-scale "feature"
    tensors (uncorrelated noise -- this dataset validates the training/eval/
    tracking *infrastructure*, not detection accuracy) plus a random number
    of ground-truth cuboids with plausible 2D/3D box parameters.
    """

    def __init__(
        self,
        num_scenes: int,
        feature_shapes: dict[str, tuple[int, int, int]],
        max_objects: int = 3,
        seed: int = 0,
    ):
        self.num_scenes = num_scenes
        self.feature_shapes = feature_shapes
        self.max_objects = max_objects
        generator = torch.Generator().manual_seed(seed)
        self._samples = [self._generate_one(generator) for _ in range(num_scenes)]

    def _generate_one(self, generator: torch.Generator) -> dict:
        num_objects = int(torch.randint(1, self.max_objects + 1, (1,), generator=generator).item())
        features = {
            key: torch.randn(*shape, generator=generator) for key, shape in self.feature_shapes.items()
        }
        rho = float((torch.rand(1, generator=generator) * 2.0 + 0.5).item())

        boxes2d, centers, dims, rots = [], [], [], []
        for _ in range(num_objects):
            cx, cy = torch.rand(2, generator=generator).tolist()
            w, h = (torch.rand(2, generator=generator) * 0.3 + 0.05).tolist()
            boxes2d.append([cx, cy, w, h])

            center = (torch.rand(3, generator=generator) * 4.0 - 2.0).tolist()
            center[2] = abs(center[2]) + 2.0  # positive depth in front of camera
            centers.append(center)

            dims.append((torch.rand(3, generator=generator) * 1.5 + 0.3).tolist())

            angle = float(torch.rand(1, generator=generator).item()) * 2.0 * math.pi
            rots.append(_rot_z(angle))

        target = {
            "boxes2d": torch.tensor(boxes2d),
            "center": torch.tensor(centers),
            "dims": torch.tensor(dims),
            "rot": torch.tensor(rots),
        }
        return {"features": features, "rho": rho, "target": target}

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]
