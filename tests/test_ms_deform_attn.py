import torch

from mapdet3d.detector.ms_deform_attn import multi_scale_deformable_attention


def _naive_bilinear_sample(feat: torch.Tensor, x: float, y: float) -> torch.Tensor:
    """Independent, non-vectorized bilinear sample matching grid_sample's
    align_corners=False pixel-coordinate convention, with zero padding.

    feat: (H, W, C). x, y: normalized sample location in [0, 1].
    """
    H, W, C = feat.shape
    # align_corners=False mapping: pixel = coord_norm_in_[-1,1] transform.
    # Equivalent closed form for coord in [0, 1]: pixel = coord * size - 0.5.
    px = x * W - 0.5
    py = y * H - 0.5

    x0 = int(torch.floor(torch.tensor(px)).item())
    y0 = int(torch.floor(torch.tensor(py)).item())
    x1, y1 = x0 + 1, y0 + 1
    wx1, wy1 = px - x0, py - y0
    wx0, wy0 = 1 - wx1, 1 - wy1

    def _get(xi: int, yi: int) -> torch.Tensor:
        if 0 <= xi < W and 0 <= yi < H:
            return feat[yi, xi]
        return torch.zeros(C, dtype=feat.dtype)

    return (
        _get(x0, y0) * wx0 * wy0
        + _get(x1, y0) * wx1 * wy0
        + _get(x0, y1) * wx0 * wy1
        + _get(x1, y1) * wx1 * wy1
    )


def naive_ms_deform_attn(
    value: torch.Tensor,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Fully independent nested-loop reference implementation."""
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=1)
    level_feats = [
        value_list[lvl].reshape(bs, h, w, num_heads, head_dim)
        for lvl, (h, w) in enumerate(value_spatial_shapes)
    ]

    output = torch.zeros(bs, num_queries, num_heads, head_dim, dtype=value.dtype)
    for b in range(bs):
        for q in range(num_queries):
            for h in range(num_heads):
                acc = torch.zeros(head_dim, dtype=value.dtype)
                for lvl in range(num_levels):
                    feat = level_feats[lvl][b, :, :, h, :]  # (H, W, head_dim)
                    for p in range(num_points):
                        x, y = sampling_locations[b, q, h, lvl, p].tolist()
                        w_attn = attention_weights[b, q, h, lvl, p].item()
                        acc = acc + _naive_bilinear_sample(feat, x, y) * w_attn
                output[b, q, h] = acc
    return output.reshape(bs, num_queries, num_heads * head_dim)


def test_ms_deform_attn_matches_naive_reference():
    torch.manual_seed(0)
    bs, num_heads, head_dim = 2, 2, 4
    spatial_shapes = [(4, 4), (2, 2)]
    num_value = sum(h * w for h, w in spatial_shapes)
    num_queries, num_levels, num_points = 3, len(spatial_shapes), 2

    value = torch.randn(bs, num_value, num_heads, head_dim)
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    raw_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
    attention_weights = raw_weights / raw_weights.sum(dim=(-2, -1), keepdim=True)

    fast_out = multi_scale_deformable_attention(
        value, spatial_shapes, sampling_locations, attention_weights
    )
    naive_out = naive_ms_deform_attn(
        value, spatial_shapes, sampling_locations, attention_weights
    )

    assert fast_out.shape == (bs, num_queries, num_heads * head_dim)
    assert torch.allclose(fast_out, naive_out, atol=1e-5)


def test_ms_deform_attn_out_of_bounds_sample_uses_zero_padding():
    bs, num_heads, head_dim = 1, 1, 2
    spatial_shapes = [(2, 2)]
    value = torch.ones(1, 4, 1, 2)
    # Sample far outside [0, 1] -> should be zero-padded, contributing nothing.
    sampling_locations = torch.tensor([[[[[[5.0, 5.0]]]]]])
    attention_weights = torch.ones(1, 1, 1, 1, 1)

    out = multi_scale_deformable_attention(
        value, spatial_shapes, sampling_locations, attention_weights
    )
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_ms_deform_attn_center_sample_matches_feature_value():
    # Sampling at the exact center of a single pixel in a 1x1 feature map
    # (any (x, y) in (0, 1) maps to that single pixel) should recover it exactly.
    value = torch.tensor([[[[3.0, -2.0]]]])  # (bs=1, num_value=1, heads=1, dim=2)
    sampling_locations = torch.tensor([[[[[[0.5, 0.5]]]]]])
    attention_weights = torch.ones(1, 1, 1, 1, 1)

    out = multi_scale_deformable_attention(value, [(1, 1)], sampling_locations, attention_weights)
    assert torch.allclose(out[0, 0], torch.tensor([3.0, -2.0]), atol=1e-6)
