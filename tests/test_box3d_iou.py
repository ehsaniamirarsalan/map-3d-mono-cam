import math

import torch

from mapdet3d.eval.box3d_iou import box3d_iou_montecarlo


def _rot_z(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_identical_axis_aligned_boxes_have_iou_exactly_one():
    # For an axis-aligned box, its sampling bounding box coincides exactly
    # with the box itself, so every sample lands inside -> zero sampling
    # noise, exact IoU = 1.
    center = torch.tensor([[1.0, 2.0, 3.0]])
    dims = torch.tensor([[2.0, 3.0, 4.0]])
    rot = torch.eye(3).unsqueeze(0)

    iou = box3d_iou_montecarlo(center, dims, rot, center, dims, rot, num_samples=2000)
    assert torch.allclose(iou, torch.ones(1, 1), atol=1e-6)


def test_identical_rotated_boxes_have_iou_close_to_one():
    # A rotated box's axis-aligned bounding box is strictly larger than the
    # box itself, so this is a genuine (noisy) Monte Carlo estimate, not an
    # exact result — verify it converges close to 1 with enough samples.
    center = torch.tensor([[1.0, 2.0, 3.0]])
    dims = torch.tensor([[2.0, 3.0, 4.0]])
    rot = _rot_z(0.37).unsqueeze(0)

    iou = box3d_iou_montecarlo(center, dims, rot, center, dims, rot, num_samples=200000)
    assert abs(iou.item() - 1.0) < 0.02


def test_disjoint_boxes_have_iou_exactly_zero():
    center_a = torch.tensor([[0.0, 0.0, 0.0]])
    center_b = torch.tensor([[100.0, 100.0, 100.0]])
    dims = torch.tensor([[1.0, 1.0, 1.0]])
    rot = torch.eye(3).unsqueeze(0)

    iou = box3d_iou_montecarlo(center_a, dims, rot, center_b, dims, rot, num_samples=2000)
    assert torch.allclose(iou, torch.zeros(1, 1), atol=1e-9)


def test_axis_aligned_partial_overlap_matches_analytic_value():
    # Two axis-aligned unit-ish cubes overlapping in a 1x1x1 region.
    center_a = torch.tensor([[0.0, 0.0, 0.0]])
    dims_a = torch.tensor([[2.0, 2.0, 2.0]])
    center_b = torch.tensor([[1.0, 1.0, 1.0]])
    dims_b = torch.tensor([[2.0, 2.0, 2.0]])
    rot = torch.eye(3).unsqueeze(0)

    # A occupies [-1,1]^3, B occupies [0,2]^3 -> intersection [0,1]^3, volume=1.
    # union = 8 + 8 - 1 = 15. iou = 1/15.
    expected = 1.0 / 15.0

    iou = box3d_iou_montecarlo(
        center_a, dims_a, rot, center_b, dims_b, rot, num_samples=200000
    )
    assert abs(iou.item() - expected) < 0.01


def test_rotated_box_overlapping_with_itself_shifted_and_rotated():
    # A cube rotated by 45 degrees about z, compared to an identical cube
    # shifted along x. Cross-check the Monte Carlo estimate is self-
    # consistent: IoU must lie in [0, 1] and be symmetric under swapping A/B.
    center_a = torch.tensor([[0.0, 0.0, 0.0]])
    dims = torch.tensor([[1.0, 1.0, 1.0]])
    rot_a = _rot_z(math.pi / 4).unsqueeze(0)
    center_b = torch.tensor([[0.3, 0.0, 0.0]])
    rot_b = _rot_z(0.0).unsqueeze(0)

    iou_ab = box3d_iou_montecarlo(center_a, dims, rot_a, center_b, dims, rot_b, num_samples=50000)
    iou_ba = box3d_iou_montecarlo(center_b, dims, rot_b, center_a, dims, rot_a, num_samples=50000)

    assert 0.0 < iou_ab.item() < 1.0
    assert abs(iou_ab.item() - iou_ba.item()) < 0.02
