"""Tests CA1MWindowDataset's windowing logic and annotations.py's box
conversion against the REAL `cubifyanything` annotation classes (Instances3D,
GeneralInstance3DBoxes), using small synthetic fixture frames instead of
real downloaded CA-1M tar data -- this validates our own glue code against
the genuine schema without requiring the (large, license-gated) dataset.
"""

import sys
from pathlib import Path

import pytest
import torch

_TOOLKIT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ml-cubifyanything"
if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))

cubifyanything = pytest.importorskip("cubifyanything.instances")

from cubifyanything.boxes import GeneralInstance3DBoxes  # noqa: E402
from cubifyanything.instances import Instances3D  # noqa: E402

from mapdet3d.data.ca1m.annotations import instances_to_target, project_corners_to_2d_box  # noqa: E402
from mapdet3d.data.ca1m.dataset import CA1MWindowDataset  # noqa: E402


class _FakeImageInfo:
    def __init__(self, K):
        self.K = K


class _FakeWideSensor:
    def __init__(self, K):
        self.image = _FakeImageInfo(K)


class _FakeSensorInfo:
    def __init__(self, K):
        self.wide = _FakeWideSensor(K)


def _make_fake_frame(video_id: int, num_boxes: int = 1, img_hw=(64, 48)):
    h, w = img_hw
    img = torch.randint(0, 255, (1, 3, h, w), dtype=torch.uint8)
    K = torch.tensor([[50.0, 0.0, w / 2], [0.0, 50.0, h / 2], [0.0, 0.0, 1.0]]).unsqueeze(0)

    xyzlhw = torch.tensor([[0.0, 0.0, 3.0, 1.0, 1.0, 1.0] for _ in range(num_boxes)])
    R = torch.eye(3).unsqueeze(0).expand(num_boxes, 3, 3).clone()
    boxes = GeneralInstance3DBoxes(xyzlhw, R) if num_boxes > 0 else GeneralInstance3DBoxes.empty()

    instances = Instances3D()
    instances.set("gt_boxes_3d", boxes)

    return {
        "wide": {"image": img, "instances": instances},
        "sensor_info": _FakeSensorInfo(K),
        "meta": {"video_id": video_id, "timestamp": 0.0},
    }


def test_project_corners_to_2d_box_centers_object_in_front_of_camera():
    # A box centered on the optical axis, directly in front, should project
    # to a 2D box roughly centered in the image.
    K = torch.tensor([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
    corners = torch.tensor([[0.5, 0.5, 3.0], [-0.5, 0.5, 3.0], [0.5, -0.5, 3.0], [-0.5, -0.5, 3.0]] * 2)
    box2d = project_corners_to_2d_box(corners, K, img_w=64, img_h=48)
    cx, cy, w, h = box2d.tolist()
    assert abs(cx - 0.5) < 0.05
    assert abs(cy - 0.5) < 0.05
    assert w > 0 and h > 0


def test_instances_to_target_shapes_with_real_annotation_classes():
    frame = _make_fake_frame(video_id=1, num_boxes=2)
    K = frame["sensor_info"].wide.image.K[0]
    target = instances_to_target(frame["wide"]["instances"], K, img_w=48, img_h=64)

    assert target["boxes2d"].shape == (2, 4)
    assert target["center"].shape == (2, 3)
    assert target["dims"].shape == (2, 3)
    assert target["rot"].shape == (2, 3, 3)


def test_instances_to_target_handles_empty_frame():
    frame = _make_fake_frame(video_id=1, num_boxes=0)
    K = frame["sensor_info"].wide.image.K[0]
    target = instances_to_target(frame["wide"]["instances"], K, img_w=48, img_h=64)

    assert target["boxes2d"].shape == (0, 4)
    assert target["center"].shape == (0, 3)


def test_windows_are_chronological_and_scene_isolated(monkeypatch):
    from tests.fixtures import raw_view, target
    frames=[dict(view=raw_view(i,scene),target=target()) for scene in ('a','b') for i in range(20)]
    ds=CA1MWindowDataset('unused',window_size=5)
    monkeypatch.setattr(ds,'iter_frames',lambda:iter(frames))
    windows=list(ds)
    assert len(windows)==40
    assert len(windows[20]['history'])==1
    for sample in windows:
        assert len({f['view']['scene_id'] for f in sample['history']})==1
        stamps=[f['view']['timestamp'] for f in sample['history']]
        assert stamps==sorted(set(stamps))


def test_timestamp_sampling_uses_seconds():
    from tests.fixtures import raw_view, target
    from mapdet3d.data.ca1m.collate import select_history
    history=[dict(view=raw_view(i),target=target()) for i in range(21)]
    chosen=select_history(history,5,2)
    assert [round(f['view']['timestamp'],2) for f in chosen]==[0.,.5,1.,1.5,2.]
