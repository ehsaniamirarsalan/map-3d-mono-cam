import pytest
import torch
pytest.importorskip('mapanything.utils.inference')
from tests.fixtures import tiny_model,raw_view
from mapdet3d.data.preprocess import preprocess_view


def test_adapter_uses_fused_features_and_converts_metadata():
    model=tiny_model();backbone=model.backbone
    view=raw_view();view['camera_poses'][0,3]=2
    prepared,_=preprocess_view(view,model.preprocess_config)
    out=backbone([prepared])
    raw,_=backbone.model._encode_n_views([prepared])
    assert not torch.allclose(out['F_E'][0],raw[0])
    assert 'ray_directions_cam' in backbone.model.seen[0]
    assert 'camera_pose_quats' in backbone.model.seen[0]
    assert out['rho'].shape==(1,)
    assert torch.allclose(out['F_11'][0]-out['F_7'][0],torch.full_like(out['F_7'][0],4))


@pytest.mark.parametrize('checkpointing',[False,True])
def test_freeze_policy_and_scale_gradients(checkpointing):
    model=tiny_model(checkpointing);model.train()
    prepared,_=preprocess_view(raw_view(),model.preprocess_config)
    out=model.backbone([prepared])
    (out['F_15'][0].square().mean()+out['rho'].sum()).backward()
    backbone=model.backbone.model
    assert not backbone.encoder.training
    assert all(p.grad is None and not p.requires_grad for p in backbone.encoder.parameters())
    for module in (backbone.info_sharing,backbone.scale_head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert backbone.scale_token.grad is not None


def test_rgb_and_intrinsics_only_are_supported():
    model=tiny_model()
    for keys in [('intrinsics','camera_poses'),('camera_poses',)]:
        view=raw_view()
        for k in keys:view.pop(k)
        prepared,_=preprocess_view(view,model.preprocess_config)
        assert torch.isfinite(model([prepared])['rho']).all()
