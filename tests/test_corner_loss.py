import math

import torch

from mapdet3d.losses.corner_loss import chamfer_distance, disentangled_corner_losses
from mapdet3d.utils.geometry import corners_from_box


def _rot_z(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_all_losses_zero_when_prediction_equals_ground_truth():
    torch.manual_seed(0)
    n = 5
    center = torch.randn(n, 3)
    dims = torch.rand(n, 3) + 0.5
    rot = torch.eye(3).expand(n, 3, 3)

    losses = disentangled_corner_losses(center, dims, rot, center, dims, rot)
    for name, val in losses.items():
        assert torch.allclose(val, torch.zeros(n), atol=1e-6), f"{name} not zero"


def test_xy_loss_isolated_from_other_attribute_errors():
    # Only the xy attribute is wrong; z/dim/rot match GT exactly.
    gt_center = torch.tensor([[0.0, 0.0, 5.0]])
    gt_dims = torch.tensor([[2.0, 2.0, 2.0]])
    gt_rot = torch.eye(3).unsqueeze(0)

    pred_center = torch.tensor([[1.0, 0.0, 5.0]])  # off by 1 in x
    pred_dims = gt_dims.clone()
    pred_rot = gt_rot.clone()

    losses = disentangled_corner_losses(
        pred_center, pred_dims, pred_rot, gt_center, gt_dims, gt_rot
    )
    assert losses["xy"].item() > 0.0
    assert torch.allclose(losses["z"], torch.zeros(1), atol=1e-6)
    assert torch.allclose(losses["dim"], torch.zeros(1), atol=1e-6)
    assert torch.allclose(losses["rot"], torch.zeros(1), atol=1e-6)


def test_chamfer_zero_for_identical_point_sets():
    pts = torch.randn(1, 8, 3)
    assert torch.allclose(chamfer_distance(pts, pts), torch.zeros(1), atol=1e-6)


def test_rotation_loss_invariant_to_180_degree_yaw_for_any_box():
    # A cuboid's corner SET is unchanged by a 180-degree rotation about the
    # vertical (z) axis, regardless of its (w, l, h) — negating x and y maps
    # each corner to another corner of the same box. Plain L1 on ordered
    # corners would see a large discrepancy here; Chamfer must report ~0,
    # which is exactly why rotation uses Chamfer instead of L1.
    gt_center = torch.tensor([[0.0, 0.0, 5.0]])
    gt_dims = torch.tensor([[2.0, 3.0, 4.0]])  # deliberately non-square
    gt_rot = torch.eye(3).unsqueeze(0)
    pred_rot = _rot_z(math.pi).unsqueeze(0)

    # Sanity: the rotation matrices themselves are very different.
    assert not torch.allclose(pred_rot, gt_rot, atol=1e-3)

    losses = disentangled_corner_losses(
        gt_center, gt_dims, pred_rot, gt_center, gt_dims, gt_rot
    )
    assert losses["rot"].item() < 1e-4


def test_rotation_loss_nonzero_for_a_genuine_90_degree_error_on_non_square_box():
    # A 90-degree yaw error on a non-square-cross-section box is a real
    # error (swaps length and width) and must NOT be treated as symmetric.
    gt_center = torch.tensor([[0.0, 0.0, 5.0]])
    gt_dims = torch.tensor([[2.0, 4.0, 1.0]])
    gt_rot = torch.eye(3).unsqueeze(0)
    pred_rot = _rot_z(math.pi / 2).unsqueeze(0)

    losses = disentangled_corner_losses(
        gt_center, gt_dims, pred_rot, gt_center, gt_dims, gt_rot
    )
    assert losses["rot"].item() > 0.5
