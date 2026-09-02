import torch

from mapdet3d.detector.detr_detector import DetrDetector


def test_end_to_end_shapes_and_query_count():
    torch.manual_seed(0)
    in_dims = {"F_E": 8, "F_7": 8, "F_11": 8, "F_15": 8}
    num_queries = 5
    num_layers = 2
    detector = DetrDetector(
        in_dims=in_dims,
        num_queries=num_queries,
        num_decoder_layers=num_layers,
        base_sizes=[0.4, 0.3, 0.2, 0.1],
        d_model=16,
        n_heads=2,
        n_points=2,
        d_ffn=32,
    )

    features = {
        "F_E": torch.randn(1, 8, 4, 4),
        "F_7": torch.randn(1, 8, 4, 4),
        "F_11": torch.randn(1, 8, 2, 2),
        "F_15": torch.randn(1, 8, 2, 2),
    }
    out = detector(features)

    num_anchors = 4 * 4 + 4 * 4 + 2 * 2 + 2 * 2
    assert out["proposal_logits"].shape == (1, num_anchors, 1)
    assert out["proposal_boxes"].shape == (1, num_anchors, 4)
    assert len(out["layer_outputs"]) == num_layers + 1
    for query_k, ref_boxes_k in out["layer_outputs"]:
        assert query_k.shape == (1, num_queries, 16)
        assert ref_boxes_k.shape == (1, num_queries, 4)
        assert (ref_boxes_k >= 0.0).all() and (ref_boxes_k <= 1.0).all()


def test_gradients_flow_through_full_pipeline():
    torch.manual_seed(1)
    in_dims = {"F_E": 4, "F_7": 4}
    detector = DetrDetector(
        in_dims=in_dims,
        num_queries=3,
        num_decoder_layers=1,
        base_sizes=[0.3, 0.2],
        d_model=8,
        n_heads=2,
        n_points=2,
        d_ffn=16,
    )
    features = {
        "F_E": torch.randn(1, 4, 2, 2, requires_grad=True),
        "F_7": torch.randn(1, 4, 2, 2, requires_grad=True),
    }

    out = detector(features)
    loss = out["proposal_logits"].sum() + out["proposal_boxes"].sum()
    for query_k, ref_boxes_k in out["layer_outputs"]:
        loss = loss + query_k.sum() + ref_boxes_k.sum()
    loss.backward()

    assert features["F_E"].grad is not None and torch.isfinite(features["F_E"].grad).all()
    for name, p in detector.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
