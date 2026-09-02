"""Tests SlidingWindowInference's buffer management and per-view output
slicing using a fake backbone (same lightweight-mock approach as
test_mapanything_wrapper.py) wired into a real MapDet3DCore, avoiding the
real pretrained MapAnything checkpoint.
"""

from dataclasses import dataclass

import torch
from torch import nn

from mapdet3d.inference.sliding_window import SlidingWindowInference
from mapdet3d.models.mapdet3d import MapDet3D, MapDet3DCore


@dataclass
class _FakeEncoderOutput:
    features: torch.Tensor


@dataclass
class _FakeMVOutput:
    features: list
    additional_token_features: torch.Tensor | None = None


class _FakeEncoderInput:
    def __init__(self, image):
        self.image = image


class _FakeEncoder(nn.Module):
    def forward(self, encoder_input):
        n = encoder_input.image.shape[0]
        return _FakeEncoderOutput(features=torch.randn(n, 8, 4, 4))


class _SimpleMVInput:
    def __init__(self, features):
        self.features = features


class _FakeInfoSharing(nn.Module):
    def forward(self, model_input):
        final = _FakeMVOutput(
            features=[torch.randn_like(f) for f in model_input.features],
            additional_token_features=torch.ones(model_input.features[0].shape[0], 8, 1),
        )
        interm7 = _FakeMVOutput(features=[torch.randn_like(f) for f in model_input.features])
        interm11 = _FakeMVOutput(features=[torch.randn_like(f) for f in model_input.features])
        return final, [interm7, interm11]


class _FakeMapAnythingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.info_sharing = _FakeInfoSharing()

    def forward(self, views):
        num_views = len(views)
        imgs = torch.cat([v["img"] for v in views], dim=0)
        batch_size = views[0]["img"].shape[0]
        encoder_output = self.encoder(_FakeEncoderInput(image=imgs))
        per_view_encoder_feats = list(encoder_output.features.chunk(num_views, dim=0))
        _ = self.info_sharing(_SimpleMVInput(features=per_view_encoder_feats))
        scale = torch.ones(batch_size, 1)  # matches real MapAnything's (B, 1) shape
        return [{"metric_scaling_factor": scale} for _ in range(num_views)]


def _make_fake_backbone():
    from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone

    backbone = MapAnythingBackbone.__new__(MapAnythingBackbone)
    nn.Module.__init__(backbone)
    backbone.model = _FakeMapAnythingModel()
    backbone._captured = {}
    backbone.model.encoder.register_forward_hook(backbone._capture("encoder_output"))
    backbone.model.info_sharing.register_forward_hook(backbone._capture("info_sharing_output"))
    return backbone


def _make_model(num_queries=3, num_layers=1):
    backbone = _make_fake_backbone()
    core = MapDet3DCore(
        in_dims={"F_E": 8, "F_7": 8, "F_11": 8, "F_15": 8},
        num_queries=num_queries,
        num_decoder_layers=num_layers,
        base_sizes=[0.3, 0.3, 0.3, 0.3],
        d_model=16,
        n_heads=2,
        n_points=2,
        d_ffn=32,
    )
    return MapDet3D(backbone, core)


def test_window_buffer_never_exceeds_window_size():
    model = _make_model()
    sw = SlidingWindowInference(model, window_size=3)

    for _ in range(5):
        sw.step({"img": torch.randn(1, 3, 4, 4)})
        assert len(sw.buffer) <= 3

    assert len(sw.buffer) == 3


def test_output_batch_size_matches_single_frame_not_the_window():
    batch_size = 2
    model = _make_model()
    sw = SlidingWindowInference(model, window_size=4)

    for _ in range(4):
        out = sw.step({"img": torch.randn(batch_size, 3, 4, 4)})

    assert out["logits"].shape[0] == batch_size
    assert out["center"].shape[0] == batch_size


def test_reset_clears_buffer():
    model = _make_model()
    sw = SlidingWindowInference(model, window_size=3)
    sw.step({"img": torch.randn(1, 3, 4, 4)})
    sw.step({"img": torch.randn(1, 3, 4, 4)})
    assert len(sw.buffer) == 2

    sw.reset()
    assert len(sw.buffer) == 0
