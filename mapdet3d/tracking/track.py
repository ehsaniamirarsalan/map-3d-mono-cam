"""A single tracked 3D object (world coordinates)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class Track:
    id: int
    center: Tensor  # (3,) world coordinates
    dims: Tensor  # (3,)
    rot: Tensor  # (3, 3) world-frame rotation
    score: float = 0.0
    age: int = 0
