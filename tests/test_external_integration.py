"""Explicit opt-in tests. They never download weights unless configured."""
import os
import itertools
import pytest
import torch

@pytest.mark.real_backbone
def test_pretrained_backbone_forward_backward():
    source=os.environ.get('MAPDET3D_BACKBONE')
    if not source:
        pytest.skip('Set MAPDET3D_BACKBONE to a local pretrained model directory or HF ID; requires substantial RAM/VRAM')
    from mapdet3d.configs.real_config import ModelConfig,build_real_model,select_device
    from mapdet3d.data.preprocess import collate_windows
    from mapdet3d.losses.criterion import SetCriterion
    from tests.fixtures import raw_view,target
    model=build_real_model(ModelConfig(pretrained=source,num_queries=4,long_edge=112))
    device=select_device(os.environ.get('MAPDET3D_DEVICE','auto'));model.to(device)
    views,targets=collate_windows([dict(views=[raw_view(0),raw_view(1)],targets=[target(),target()])],model.preprocess_config)
    from mapdet3d.engine.train_loop import views_batch_step
    loss=views_batch_step(model,SetCriterion(),(views,targets),device)['loss_total'];loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.model.scale_token.grad is not None

@pytest.mark.real_data
def test_ca1m_real_archive():
    source=os.environ.get('MAPDET3D_CA1M')
    if not source:pytest.skip('Set MAPDET3D_CA1M to local CA-1M tar archives')
    from mapdet3d.data.ca1m.dataset import CA1MWindowDataset
    frames=list(itertools.islice(CA1MWindowDataset(source).iter_frames(),3))
    assert frames
    assert all(torch.isfinite(f['target']['center']).all() for f in frames)

@pytest.mark.real_data
def test_scannet_real_scene():
    root=os.environ.get('MAPDET3D_SCANNET')
    if not root:pytest.skip('Set MAPDET3D_SCANNET to a prepared ScanNet root')
    from mapdet3d.data.scannet.dataset import ScanNetDataset
    _,target,frames=next(ScanNetDataset(root).iter_scenes())
    first=next(frames)
    assert len(first['target']['center'])==len(target['center'])
