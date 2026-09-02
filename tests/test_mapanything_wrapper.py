"""Tests the MapAnythingBackbone wrapper's hook-capture glue logic using
lightweight fake stand-ins for `model.encoder` / `model.info_sharing`,
matching the real uniception dataclass interfaces confirmed by source
inspection (see mapanything_wrapper.py's module docstring). This avoids
downloading the real multi-GB pretrained MapAnything checkpoint just to
test our own extraction code, which is the part we actually wrote and can
get wrong.
"""

from dataclasses import dataclass

import torch
from torch import nn

from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone


@dataclass
class _FakeEncoderOutput:
    features: torch.Tensor


@dataclass
class _FakeMVOutput:
    features: list
    additional_token_features: torch.Tensor | None = None


class _FakeEncoder(nn.Module):
    def forward(self, encoder_input):
        return _FakeEncoderOutput(features=encoder_input.image.sum(dim=(1, 2, 3), keepdim=True).expand(-1, 4, 2, 2))


class _FakeInfoSharing(nn.Module):
    def forward(self, model_input):
        num_views = len(model_input.features)
        final = _FakeMVOutput(
            features=[torch.full_like(f, fill_value=15.0) for f in model_input.features],
            additional_token_features=torch.ones(model_input.features[0].shape[0], 8, 1),
        )
        interm7 = _FakeMVOutput(features=[torch.full_like(f, fill_value=7.0) for f in model_input.features])
        interm11 = _FakeMVOutput(features=[torch.full_like(f, fill_value=11.0) for f in model_input.features])
        return final, [interm7, interm11]


class _FakeEncoderInput:
    def __init__(self, image):
        self.image = image


class _FakeMapAnythingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.info_sharing = _FakeInfoSharing()

    def forward(self, views):
        num_views = len(views)
        imgs = torch.cat([v["img"] for v in views], dim=0)
        _ = self.encoder(_FakeEncoderInput(image=imgs))  # triggers the hook, mirrors real _encode_n_views
        _ = self.info_sharing(_SimpleMVInput(features=[v["img"] for v in views]))  # triggers the hook
        batch_size = views[0]["img"].shape[0]
        # Real MapAnything returns this as (B, 1), not (B,) -- match that
        # exactly so tests catch shape-handling bugs (see
        # mapanything_wrapper.py's "rho" extraction comment).
        scale = torch.full((batch_size, 1), 2.5)
        return [{"metric_scaling_factor": scale} for _ in range(num_views)]


class _SimpleMVInput:
    def __init__(self, features):
        self.features = features


def _make_backbone_with_fake_model() -> MapAnythingBackbone:
    backbone = MapAnythingBackbone.__new__(MapAnythingBackbone)
    nn.Module.__init__(backbone)
    backbone.model = _FakeMapAnythingModel()
    backbone._captured = {}
    backbone.model.encoder.register_forward_hook(backbone._capture("encoder_output"))
    backbone.model.info_sharing.register_forward_hook(backbone._capture("info_sharing_output"))
    return backbone


def test_forward_extracts_all_expected_keys_with_correct_shapes():
    backbone = _make_backbone_with_fake_model()
    num_views, batch_size = 3, 2
    views = [{"img": torch.randn(batch_size, 3, 8, 8)} for _ in range(num_views)]

    out = backbone(views)

    assert set(out.keys()) == {"F_E", "F_7", "F_11", "F_15", "rho", "scale_token_features"}
    assert len(out["F_E"]) == num_views
    assert len(out["F_7"]) == num_views
    assert len(out["F_11"]) == num_views
    assert len(out["F_15"]) == num_views
    assert out["F_E"][0].shape[0] == batch_size
    assert out["rho"].shape == (batch_size,)
    assert torch.allclose(out["rho"], torch.full((batch_size,), 2.5))


def test_intermediate_layers_are_not_confused_with_each_other():
    # The fake info_sharing fills F_7/F_11/F_15 with distinct constants
    # (7, 11, 15) so a swapped-index bug (e.g. F_7 <-> F_11) is caught.
    backbone = _make_backbone_with_fake_model()
    views = [{"img": torch.randn(1, 3, 4, 4)} for _ in range(2)]

    out = backbone(views)

    assert torch.allclose(out["F_7"][0], torch.full_like(out["F_7"][0], 7.0))
    assert torch.allclose(out["F_11"][0], torch.full_like(out["F_11"][0], 11.0))
    assert torch.allclose(out["F_15"][0], torch.full_like(out["F_15"][0], 15.0))


def test_encoder_output_is_split_per_view_via_chunk():
    # F_E must correspond 1:1 with input views in order, matching
    # MapAnything's own `.chunk(num_views, dim=0)` splitting convention.
    backbone = _make_backbone_with_fake_model()
    views = [
        {"img": torch.full((1, 3, 4, 4), fill_value=float(v))} for v in range(4)
    ]

    out = backbone(views)

    for v in range(4):
        # The fake encoder sums each view's image, so each chunk should
        # reflect only its own view's fill value, not a mix of all views.
        expected_sum = float(v) * 3 * 4 * 4
        assert torch.allclose(out["F_E"][v], torch.full_like(out["F_E"][v], expected_sum))
