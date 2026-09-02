import torch

from mapdet3d.detector.anchor_generator import generate_anchors
from mapdet3d.detector.multiscale_proj import MultiScaleProjector
from mapdet3d.detector.proposal_head import ProposalHead, select_top_m


def test_multiscale_projector_shapes_and_order():
    torch.manual_seed(0)
    in_dims = {"F_E": 8, "F_7": 8, "F_11": 8, "F_15": 8}
    proj = MultiScaleProjector(in_dims, out_dim=16, hidden_dim=16)

    features = {
        "F_E": torch.randn(2, 8, 4, 4),
        "F_7": torch.randn(2, 8, 4, 4),
        "F_11": torch.randn(2, 8, 2, 2),
        "F_15": torch.randn(2, 8, 2, 2),
    }
    tokens, spatial_shapes = proj(features)

    expected_len = 4 * 4 + 4 * 4 + 2 * 2 + 2 * 2
    assert tokens.shape == (2, expected_len, 16)
    assert spatial_shapes == [(4, 4), (4, 4), (2, 2), (2, 2)]


def test_generate_anchors_shape_and_bounds():
    spatial_shapes = [(2, 2), (1, 1)]
    base_sizes = [0.1, 0.5]
    anchors = generate_anchors(spatial_shapes, base_sizes)

    assert anchors.shape == (2 * 2 + 1 * 1, 4)
    assert (anchors[:, :2] >= 0).all() and (anchors[:, :2] <= 1).all()
    # First 4 anchors (level 0) all have size 0.1; last anchor (level 1) has size 0.5.
    assert torch.allclose(anchors[:4, 2:], torch.full((4, 2), 0.1))
    assert torch.allclose(anchors[4:, 2:], torch.full((1, 2), 0.5))


def test_generate_anchors_grid_centers_are_evenly_spaced():
    anchors = generate_anchors([(2, 2)], [0.2])
    # A 2x2 grid over [0,1] with cell-center convention -> centers at 0.25, 0.75.
    expected_centers = {0.25, 0.75}
    cx_values = set(round(v, 4) for v in anchors[:, 0].tolist())
    cy_values = set(round(v, 4) for v in anchors[:, 1].tolist())
    assert cx_values == expected_centers
    assert cy_values == expected_centers


def test_proposal_head_output_shapes_and_box_clamping():
    torch.manual_seed(1)
    head = ProposalHead(in_dim=16, hidden_dim=16)
    tokens = torch.randn(3, 10, 16)
    anchors = generate_anchors([(2, 5)], [0.3])

    logits, boxes = head(tokens, anchors)
    assert logits.shape == (3, 10, 1)
    assert boxes.shape == (3, 10, 4)
    assert (boxes >= 0.0).all() and (boxes <= 1.0).all()


def test_select_top_m_picks_highest_scoring_and_detaches():
    tokens = torch.randn(1, 5, 4, requires_grad=True)
    boxes = torch.rand(1, 5, 4, requires_grad=True)
    logits = torch.tensor([[[0.0], [5.0], [-1.0], [3.0], [1.0]]])  # index 1 highest, then 3, then 4

    query_feats, ref_boxes, topk_idx = select_top_m(tokens, logits, boxes, num_queries=2)

    assert topk_idx.tolist() == [[1, 3]]
    assert torch.allclose(query_feats[0, 0], tokens[0, 1].detach())
    assert torch.allclose(query_feats[0, 1], tokens[0, 3].detach())
    assert not query_feats.requires_grad
    assert not ref_boxes.requires_grad
