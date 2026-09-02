import torch

from mapdet3d.losses.loss_2d import box_iou, generalized_box_iou, sigmoid_focal_loss
from mapdet3d.losses.matcher import HungarianMatcher


def test_giou_is_one_for_identical_boxes():
    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    giou = generalized_box_iou(box, box)
    assert torch.allclose(giou, torch.ones(1, 1), atol=1e-6)


def test_giou_is_negative_for_disjoint_boxes():
    box_a = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    box_b = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    giou = generalized_box_iou(box_a, box_b)
    assert giou.item() < 0.0


def test_box_iou_matches_hand_computed_overlap():
    # Two unit-side boxes in xyxy, overlapping in a 1x1 square out of area 4 each.
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    iou, union = box_iou(a, b)
    # intersection = 1, union = 4+4-1=7
    assert torch.allclose(iou, torch.tensor([[1.0 / 7.0]]), atol=1e-6)


def test_focal_loss_zero_when_confident_and_correct():
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    loss = sigmoid_focal_loss(logits, targets)
    assert (loss < 1e-3).all()


def test_matcher_recovers_permuted_gt_assignment():
    torch.manual_seed(0)
    # 3 GT boxes; predictions include exact copies of GT boxes but shuffled,
    # plus extra low-confidence "background" queries.
    gt_boxes = torch.tensor(
        [
            [0.2, 0.2, 0.1, 0.1],
            [0.5, 0.5, 0.2, 0.2],
            [0.8, 0.3, 0.15, 0.1],
        ]
    )
    perm = torch.tensor([2, 0, 1])
    pred_boxes_matched = gt_boxes[perm]
    extra = torch.rand(4, 4) * 0.05 + 0.5  # background queries, far from GT
    pred_boxes = torch.cat([pred_boxes_matched, extra], dim=0).unsqueeze(0)  # (1, 7, 4)

    logits = torch.full((1, 7, 1), -5.0)
    logits[0, :3, 0] = 5.0  # high confidence on the true-positive queries

    matcher = HungarianMatcher()
    (pred_idx, gt_idx) = matcher(logits, pred_boxes, [gt_boxes])[0]

    # Each GT index should be matched to the query holding its exact copy.
    matched = dict(zip(gt_idx.tolist(), pred_idx.tolist()))
    for gt_i, pred_i in matched.items():
        assert torch.allclose(pred_boxes[0, pred_i], gt_boxes[gt_i], atol=1e-6)


def test_matcher_handles_empty_targets():
    logits = torch.zeros(1, 5, 1)
    boxes = torch.rand(1, 5, 4)
    matcher = HungarianMatcher()
    (pred_idx, gt_idx) = matcher(logits, boxes, [torch.zeros(0, 4)])[0]
    assert pred_idx.numel() == 0
    assert gt_idx.numel() == 0
