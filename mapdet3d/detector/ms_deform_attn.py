"""Pure-PyTorch multi-scale deformable attention (Zhu et al., Deformable DETR).

Implemented with `F.grid_sample` instead of a custom CUDA kernel, per the
plan's decision to avoid fragile native-extension builds (particularly
brittle on Windows) — see plan §6 "Vendor vs. Build-from-Scratch".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """
    Args:
        value: (bs, num_value, num_heads, head_dim) flattened multi-level value,
            where num_value == sum(h * w for h, w in value_spatial_shapes).
        value_spatial_shapes: list of (h, w) per feature level, in the same
            order as `value`'s concatenation along dim=1.
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2),
            normalized to [0, 1] in (x, y) order.
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points).

    Returns:
        (bs, num_queries, num_heads * head_dim)
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=1)
    # grid_sample expects grid coordinates in [-1, 1]; our locations are [0, 1].
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level_id, (h, w) in enumerate(value_spatial_shapes):
        # (bs, h*w, num_heads, head_dim) -> (bs*num_heads, head_dim, h, w)
        value_l = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, head_dim, h, w)
        )
        # (bs, num_queries, num_heads, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # (bs*num_heads, head_dim, num_queries, num_points)
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * head_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Learned multi-scale deformable cross-attention module.

    Predicts per-query sampling offsets and attention weights from the query
    features, converts offsets to normalized sampling locations relative to
    each query's reference point, and delegates the actual sampling to
    `multi_scale_deformable_attention`.
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(n_heads) * (2 * torch.pi / n_heads)
        radial = torch.stack([angles.cos(), angles.sin()], -1)
        radial = radial / radial.abs().amax(-1, keepdim=True)
        radial = radial[:, None, None, :].repeat(1, n_levels, n_points, 1)
        radial *= torch.arange(1, n_points + 1)[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(radial.flatten())
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value_input: Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            query: (N, Len_q, d_model).
            reference_points: (N, Len_q, 2), normalized (x, y) in [0, 1],
                shared across levels (a single reference point per query;
                the same location is used to anchor sampling at every level).
            value_input: (N, Len_in, d_model) -- Q_IMG, Len_in == sum(h*w).
            spatial_shapes: list of (h, w) per level, matching `value_input`'s
                concatenation order.

        Returns:
            (N, Len_q, d_model)
        """
        n, len_q, _ = query.shape
        len_in = value_input.shape[1]

        value = self.value_proj(value_input).view(n, len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = torch.softmax(attention_weights, dim=-1).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Per-level offset normalizer: offsets are in pixel units of each
        # level's own resolution, converted to normalized [0,1] coordinates.
        offset_normalizer = torch.as_tensor(
            [[w, h] for h, w in spatial_shapes], device=query.device, dtype=query.dtype
        )  # (n_levels, 2), (x, y) order to match (w, h) normalization.

        if reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, None, None, :2]
                + sampling_offsets / self.n_points
                * reference_points[:, :, None, None, None, 2:] * 0.5
            )
        elif reference_points.shape[-1] == 2:
            sampling_locations = (
                reference_points[:, :, None, None, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError("Reference points must have two or four coordinates")

        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)
