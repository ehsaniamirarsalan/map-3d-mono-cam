import torch

from mapdet3d.detector.decoder import DeformableDecoder
from mapdet3d.detector.decoder_layer import DeformableDecoderLayer


def test_decoder_layer_is_identity_box_update_at_init():
    # bbox_refine's final layer is zero-initialized, so a fresh layer must
    # leave the reference box unchanged (delta == 0) on the first call.
    torch.manual_seed(0)
    layer = DeformableDecoderLayer(d_model=16, n_heads=2, n_levels=1, n_points=2, d_ffn=32)
    query = torch.randn(1, 3, 16)
    ref_boxes = torch.rand(1, 3, 4)
    value_input = torch.randn(1, 9, 16)

    _, updated_ref_boxes = layer(query, ref_boxes, value_input, [(3, 3)])
    assert torch.allclose(updated_ref_boxes, ref_boxes, atol=1e-6)


def test_decoder_layer_shapes_and_box_bounds():
    layer = DeformableDecoderLayer(d_model=16, n_heads=2, n_levels=2, n_points=2, d_ffn=32)
    query = torch.randn(2, 4, 16)
    ref_boxes = torch.rand(2, 4, 4)
    value_input = torch.randn(2, 20, 16)
    spatial_shapes = [(4, 4), (2, 2)]

    new_query, new_ref_boxes = layer(query, ref_boxes, value_input, spatial_shapes)
    assert new_query.shape == query.shape
    assert new_ref_boxes.shape == ref_boxes.shape
    assert (new_ref_boxes >= 0.0).all() and (new_ref_boxes <= 1.0).all()


def test_decoder_produces_num_layers_plus_one_outputs():
    torch.manual_seed(1)
    num_layers = 3
    decoder = DeformableDecoder(num_layers=num_layers, d_model=16, n_heads=2, n_levels=1, n_points=2, d_ffn=32)

    query0 = torch.randn(1, 5, 16)
    ref_boxes0 = torch.rand(1, 5, 4)
    value_input = torch.randn(1, 16, 16)

    outputs = decoder(query0, ref_boxes0, value_input, [(4, 4)])
    assert len(outputs) == num_layers + 1
    assert outputs[0][0] is query0
    assert outputs[0][1] is ref_boxes0
    for query_k, ref_boxes_k in outputs:
        assert query_k.shape == (1, 5, 16)
        assert ref_boxes_k.shape == (1, 5, 4)


def test_decoder_gradients_flow_to_all_layers():
    decoder = DeformableDecoder(num_layers=2, d_model=8, n_heads=2, n_levels=1, n_points=2, d_ffn=16)
    query0 = torch.randn(1, 2, 8, requires_grad=True)
    ref_boxes0 = torch.rand(1, 2, 4)
    value_input = torch.randn(1, 4, 8)

    outputs = decoder(query0, ref_boxes0, value_input, [(2, 2)])
    loss = sum(q.sum() + b.sum() for q, b in outputs[1:])
    loss.backward()

    assert query0.grad is not None and torch.isfinite(query0.grad).all()
    for name, p in decoder.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
