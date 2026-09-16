import torch

from mapdet3d.detector.ms_deform_attn import MSDeformAttn


def test_output_shape_and_gradient_flow():
    torch.manual_seed(0)
    d_model, n_levels, n_heads, n_points = 16, 2, 2, 3
    attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)

    spatial_shapes = [(4, 4), (2, 2)]
    len_in = sum(h * w for h, w in spatial_shapes)
    bs, len_q = 2, 5

    query = torch.randn(bs, len_q, d_model, requires_grad=True)
    reference_points = torch.rand(bs, len_q, 2)
    value_input = torch.randn(bs, len_in, d_model, requires_grad=True)

    out = attn(query, reference_points, value_input, spatial_shapes)
    assert out.shape == (bs, len_q, d_model)

    out.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert value_input.grad is not None and torch.isfinite(value_input.grad).all()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"


def test_radial_offsets_and_uniform_weights_at_init():
    attn = MSDeformAttn(d_model=8, n_levels=2, n_heads=2, n_points=2)
    query = torch.zeros(1, 1, 8)
    offsets = attn.sampling_offsets(query)
    weights_logits = attn.attention_weights(query)
    radial = offsets.view(1,1,2,2,2,2)
    assert torch.allclose(radial[...,1,:],2*radial[...,0,:])
    assert (radial.abs().sum(-1)>0).all()
    weights = torch.softmax(weights_logits.view(1, 1, 2, 2 * 2), dim=-1)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 4.0))
