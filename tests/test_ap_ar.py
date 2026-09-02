import torch

from mapdet3d.eval.ap_ar import compute_ap_ar


def _box(center, dims):
    return {
        "center": torch.tensor([center]),
        "dims": torch.tensor([dims]),
        "rot": torch.eye(3).unsqueeze(0),
    }


def test_perfect_predictions_give_ap_and_ar_of_one():
    gt = _box([0.0, 0.0, 5.0], [1.0, 1.0, 1.0])
    pred = {**gt, "scores": torch.tensor([0.9])}

    result = compute_ap_ar([pred], [gt], iou_threshold=0.5, num_samples=2000)
    assert result["AP"] > 0.99
    assert result["AR"] > 0.99


def test_no_predictions_gives_zero_ap_and_ar():
    gt = _box([0.0, 0.0, 5.0], [1.0, 1.0, 1.0])
    empty_pred = {
        "center": torch.zeros(0, 3),
        "dims": torch.zeros(0, 3),
        "rot": torch.zeros(0, 3, 3),
        "scores": torch.zeros(0),
    }

    result = compute_ap_ar([empty_pred], [gt], iou_threshold=0.5, num_samples=2000)
    assert result["AP"] == 0.0
    assert result["AR"] == 0.0


def test_no_ground_truth_gives_zero_by_convention():
    empty_gt = {"center": torch.zeros(0, 3), "dims": torch.zeros(0, 3), "rot": torch.zeros(0, 3, 3)}
    pred = _box([0.0, 0.0, 5.0], [1.0, 1.0, 1.0])
    pred["scores"] = torch.tensor([0.9])

    result = compute_ap_ar([pred], [empty_gt], iou_threshold=0.5, num_samples=2000)
    assert result["AP"] == 0.0
    assert result["AR"] == 0.0


def test_false_positive_scored_above_the_true_positive_hurts_ap_not_ar():
    # A false positive ranked BEFORE the true match drags down precision at
    # the recall level where the true positive occurs (rank 1: 1 FP, 0 TP ->
    # precision 0; rank 2: 1 FP, 1 TP -> precision 0.5), so interpolated AP
    # must be < 1 even though full recall is still eventually achieved.
    # (A false positive scored AFTER full recall is reached does NOT lower
    # interpolated AP, since the max-precision-at-recall=1.0 point already
    # satisfies every recall threshold -- that is standard AP behavior, not
    # a bug, and is intentionally not asserted here.)
    gt = _box([0.0, 0.0, 5.0], [1.0, 1.0, 1.0])
    pred = {
        "center": torch.tensor([[100.0, 100.0, 100.0], [0.0, 0.0, 5.0]]),
        "dims": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        "rot": torch.eye(3).unsqueeze(0).expand(2, 3, 3),
        "scores": torch.tensor([0.9, 0.5]),  # FP scored higher than the TP
    }

    result = compute_ap_ar([pred], [gt], iou_threshold=0.5, num_samples=2000)
    assert result["AR"] > 0.99  # the true GT was still found
    assert result["AP"] < 1.0  # but precision-recall curve is hurt by the FP


def test_iou_threshold_controls_strictness():
    # A prediction offset just enough to fail a strict IoU threshold but
    # pass a lenient one.
    gt = _box([0.0, 0.0, 5.0], [2.0, 2.0, 2.0])
    pred = {**_box([1.5, 0.0, 5.0], [2.0, 2.0, 2.0]), "scores": torch.tensor([0.9])}

    lenient = compute_ap_ar([pred], [gt], iou_threshold=0.05, num_samples=20000)
    strict = compute_ap_ar([pred], [gt], iou_threshold=0.9, num_samples=20000)
    assert lenient["AR"] > strict["AR"]
