"""Up-to-scale 3D bounding box head (paper Sec. 3.4).

Regresses each geometric attribute (center, dimensions, rotation,
objectness) from a decoder query in an up-to-scale parameterization, then
converts center/dimensions to metric units via the backbone's predicted
per-window scale factor rho: x = rho * x~, y = rho * y~, z = rho * exp(d~),
and (w, l, h) = rho * exp(log-size).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mapdet3d.heads.rotation import allocentric_to_egocentric, rotation_6d_to_matrix


class _TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Box3DHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.center_mlp = _TwoLayerMLP(in_dim, hidden_dim, 3)  # x~, y~, log-depth d~
        self.dim_mlp = _TwoLayerMLP(in_dim, hidden_dim, 3)  # log-sizes s~w, s~l, s~h
        self.rot_mlp = _TwoLayerMLP(in_dim, hidden_dim, 6)  # allocentric 6D rotation
        self.conf_mlp = nn.Linear(in_dim, 1)  # binary objectness logit

    def forward(self, query: Tensor, rho: Tensor) -> dict[str, Tensor]:
        """
        Args:
            query: (..., in_dim) decoder query features, one per candidate box.
            rho: (...) per-box metric scale factor (the backbone's per-window
                rho, already broadcast/expanded to one value per box query by
                the caller) — same leading shape as `query` minus its last
                (feature) dimension.

        Returns:
            dict with:
              center: (..., 3) metric (x, y, z) in camera coordinates.
              dims:   (..., 3) metric (w, l, h), all positive.
              rot:    (..., 3, 3) egocentric rotation matrix.
              conf:   (..., 1) objectness logit (apply sigmoid for a score).
        """
        rho = rho.unsqueeze(-1)  # (..., 1), broadcasts against (..., 3)

        raw_center = self.center_mlp(query)  # (..., 3): x~, y~, log-depth
        xy_tilde = raw_center[..., :2]
        log_depth = raw_center[..., 2:3]
        xy = rho * xy_tilde
        z = rho * torch.exp(log_depth)
        center = torch.cat([xy, z], dim=-1)

        raw_dims = self.dim_mlp(query)  # (..., 3): log-sizes
        dims = rho * torch.exp(raw_dims)

        rot6d = self.rot_mlp(query)
        rot_allo = rotation_6d_to_matrix(rot6d)
        # Ray direction only depends on the direction of `center`, not its
        # magnitude, so using the metric center here is equivalent to using
        # the up-to-scale center (rho > 0 never flips direction).
        rot_ego = allocentric_to_egocentric(rot_allo, center)

        conf = self.conf_mlp(query)

        return {"center": center, "dims": dims, "rot": rot_ego, "conf": conf}
