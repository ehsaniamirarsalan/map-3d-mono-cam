import torch

from mapdet3d.losses.criterion import SetCriterion


def _make_perfect_prediction(gt_boxes2d, gt_center, gt_dims, gt_rot, num_extra=2):
    """Build a LayerPred that exactly reproduces GT at the first N queries,
    with low-confidence "background" queries appended after."""
    n = gt_boxes2d.shape[0]
    total = n + num_extra

    logits = torch.full((1, total, 1), -10.0)
    logits[0, :n, 0] = 10.0

    boxes2d = torch.rand(1, total, 4) * 0.1 + 0.5
    boxes2d[0, :n] = gt_boxes2d

    center = torch.randn(1, total, 3)
    center[0, :n] = gt_center
    dims = torch.rand(1, total, 3) + 0.5
    dims[0, :n] = gt_dims
    rot = torch.eye(3).expand(1, total, 3, 3).clone()
    rot[0, :n] = gt_rot

    return {"logits": logits, "boxes2d": boxes2d, "center": center, "dims": dims, "rot": rot}


def _make_target(gt_boxes2d, gt_center, gt_dims, gt_rot):
    return {"boxes2d": gt_boxes2d, "center": gt_center, "dims": gt_dims, "rot": gt_rot}


def test_loss_near_zero_for_perfect_predictions_across_all_layers():
    torch.manual_seed(0)
    gt_boxes2d = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.6, 0.15, 0.2]])
    gt_center = torch.tensor([[0.0, 0.0, 5.0], [1.0, -1.0, 6.0]])
    gt_dims = torch.tensor([[1.0, 2.0, 1.5], [0.8, 0.8, 2.0]])
    gt_rot = torch.eye(3).unsqueeze(0).expand(2, 3, 3).clone()

    target = _make_target(gt_boxes2d, gt_center, gt_dims, gt_rot)
    num_layers = 3
    layer_preds = [
        _make_perfect_prediction(gt_boxes2d, gt_center, gt_dims, gt_rot) for _ in range(num_layers)
    ]

    criterion = SetCriterion()
    losses = criterion(layer_preds, [target])

    assert losses["loss_bbox"].item() < 1e-4
    assert losses["loss_xy"].item() < 1e-4
    assert losses["loss_z"].item() < 1e-4
    assert losses["loss_dim"].item() < 1e-4
    assert losses["loss_rot"].item() < 1e-4
    assert losses["loss_class"].item() < 0.1  # confident correct classification


def test_loss_is_higher_for_wrong_predictions():
    torch.manual_seed(1)
    gt_boxes2d = torch.tensor([[0.2, 0.2, 0.1, 0.1]])
    gt_center = torch.tensor([[0.0, 0.0, 5.0]])
    gt_dims = torch.tensor([[1.0, 2.0, 1.5]])
    gt_rot = torch.eye(3).unsqueeze(0)
    target = _make_target(gt_boxes2d, gt_center, gt_dims, gt_rot)

    good_pred = _make_perfect_prediction(gt_boxes2d, gt_center, gt_dims, gt_rot)
    bad_pred = {
        "logits": torch.full((1, 3, 1), -10.0),
        "boxes2d": torch.rand(1, 3, 4) * 0.1 + 0.8,
        "center": torch.randn(1, 3, 3) + 10.0,
        "dims": torch.rand(1, 3, 3) + 0.1,
        "rot": torch.eye(3).expand(1, 3, 3).clone(),
    }

    criterion = SetCriterion()
    good_loss = criterion([good_pred], [target])["loss_total"]
    bad_loss = criterion([bad_pred], [target])["loss_total"]

    assert bad_loss.item() > good_loss.item()


def test_criterion_handles_empty_targets_without_crashing():
    pred = {
        "logits": torch.zeros(1, 4, 1),
        "boxes2d": torch.rand(1, 4, 4),
        "center": torch.randn(1, 4, 3),
        "dims": torch.rand(1, 4, 3) + 0.1,
        "rot": torch.eye(3).expand(1, 4, 3, 3).clone(),
    }
    target = {
        "boxes2d": torch.zeros(0, 4),
        "center": torch.zeros(0, 3),
        "dims": torch.zeros(0, 3),
        "rot": torch.zeros(0, 3, 3),
    }
    criterion = SetCriterion()
    losses = criterion([pred], [target])
    assert torch.isfinite(losses["loss_total"])
