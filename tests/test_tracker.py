import torch

from mapdet3d.tracking.affinity import transform_to_world
from mapdet3d.tracking.tracker import Tracker


def test_transform_to_world_identity_pose():
    center_cam = torch.tensor([[1.0, 2.0, 3.0]])
    rot_cam = torch.eye(3).unsqueeze(0)
    identity_pose = torch.eye(4)

    center_world, rot_world = transform_to_world(center_cam, rot_cam, identity_pose)
    assert torch.allclose(center_world, center_cam)
    assert torch.allclose(rot_world, rot_cam)


def test_transform_to_world_applies_translation():
    center_cam = torch.tensor([[0.0, 0.0, 0.0]])
    rot_cam = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([5.0, -2.0, 1.0])

    center_world, _ = transform_to_world(center_cam, rot_cam, pose)
    assert torch.allclose(center_world, torch.tensor([[5.0, -2.0, 1.0]]))


def test_first_frame_creates_new_tracks():
    tracker = Tracker(iou_threshold=0.1)
    dets_center = torch.tensor([[0.0, 0.0, 5.0], [3.0, 0.0, 5.0]])
    dets_dims = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    dets_rot = torch.eye(3).unsqueeze(0).expand(2, 3, 3)

    tracks = tracker.step(dets_center, dets_dims, dets_rot, torch.eye(4))
    assert len(tracks) == 2
    assert {t.id for t in tracks} == {0, 1}


def test_same_object_across_frames_reuses_track_id():
    tracker = Tracker(iou_threshold=0.1)
    center = torch.tensor([[0.0, 0.0, 5.0]])
    dims = torch.tensor([[1.0, 1.0, 1.0]])
    rot = torch.eye(3).unsqueeze(0)

    tracker.step(center, dims, rot, torch.eye(4))
    # Slightly shifted detection in the next frame (still high IoU overlap).
    center2 = torch.tensor([[0.05, 0.0, 5.0]])
    tracks = tracker.step(center2, dims, rot, torch.eye(4))

    assert len(tracks) == 1
    assert tracks[0].id == 0
    assert tracks[0].age == 2


def test_track_box_replaced_only_if_larger():
    tracker = Tracker(iou_threshold=0.05)
    small_center = torch.tensor([[0.0, 0.0, 5.0]])
    small_dims = torch.tensor([[1.0, 1.0, 1.0]])
    rot = torch.eye(3).unsqueeze(0)

    tracker.step(small_center, small_dims, rot, torch.eye(4))

    # Frame 2: a larger overlapping observation -> should replace.
    large_dims = torch.tensor([[2.0, 2.0, 2.0]])
    tracks = tracker.step(small_center, large_dims, rot, torch.eye(4))
    assert torch.allclose(tracks[0].dims, large_dims)

    # Frame 3: a smaller overlapping observation -> should NOT replace (keep the larger box).
    tracker.step(small_center, small_dims, rot, torch.eye(4))
    assert torch.allclose(tracker.tracks[0].dims, large_dims)


def test_far_away_detection_creates_a_separate_track():
    tracker = Tracker(iou_threshold=0.1)
    center1 = torch.tensor([[0.0, 0.0, 5.0]])
    dims = torch.tensor([[1.0, 1.0, 1.0]])
    rot = torch.eye(3).unsqueeze(0)
    tracker.step(center1, dims, rot, torch.eye(4))

    center2 = torch.tensor([[50.0, 50.0, 50.0]])
    tracks = tracker.step(center2, dims, rot, torch.eye(4))
    assert len(tracks) == 2
