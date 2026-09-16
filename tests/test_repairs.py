import io
import json
import tarfile
import numpy as np
import pytest
import torch
from PIL import Image
from mapdet3d.data.preprocess import PreprocessConfig,preprocess_view,collate_windows
from mapdet3d.data.ca1m.dataset import read_capture,CA1MWindowDataset
from mapdet3d.data.ca1m.collate import CA1MCollator
from mapdet3d.data.ca1m.annotations import project_corners_to_2d_box
from mapdet3d.eval.box3d_iou import box3d_iou
from mapdet3d.eval.ap_ar import compute_ap_ar
from tests.fixtures import raw_view,target,tiny_model


def test_preprocessing_preserves_projection_and_3d():
    view=raw_view();gt=target()
    out,transformed=preprocess_view(view,PreprocessConfig(24,2,'identity'),(24,24),gt)
    point=torch.tensor([.2,.3,3.])
    original=view['intrinsics']@point;original=original[:2]/original[2]
    projected=out['intrinsics'][0]@point;projected=projected[:2]/projected[2]
    tr=out['transform']
    assert torch.allclose(projected,original*torch.tensor(tr['scale_xy'])+torch.tensor(tr['pad_xy']))
    for k in ('center','dims','rot'):assert torch.equal(transformed[k],gt[k])
    assert out['img'].shape==(1,3,24,24)
    assert torch.count_nonzero(out['img'][...,:3,:])==0


def test_multiview_batch_and_proposal_gradients():
    pytest.importorskip('mapanything.utils.inference')
    model=tiny_model()
    batch=[dict(views=[raw_view(i) for i in range(3)],targets=[target(10*b+i) for i in range(3)]) for b in range(2)]
    views,targets=collate_windows(batch,model.preprocess_config)
    assert [float(t['center'][0,0]) for t in targets]==[0.,10.,1.,11.,2.,12.]
    from mapdet3d.losses.criterion import SetCriterion
    output=model(views)
    loss=SetCriterion()(output,targets)['loss_total']
    loss.backward()
    for name,p in model.core.detector.proposal_head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.core.detector.proposal_head.class_head.weight.grad.abs().sum()>0
    assert model.core.detector.proposal_head.bbox_head[-1].weight.grad.abs().sum()>0
    assert model.backbone.model.scale_token.grad.abs().sum()>0
    with pytest.raises(ValueError,match='per-view'):
        SetCriterion()(output,targets[:2])


def test_near_plane_and_behind_camera_projection():
    from mapdet3d.utils.geometry import corners_from_box
    K=torch.tensor([[10.,0,10],[0,10,10],[0,0,1]])
    for z in (-2.,0.,2.):
        corners=corners_from_box(torch.tensor([[0.,0.,z]]),torch.ones(1,3),torch.eye(3)[None])[0]
        box=project_corners_to_2d_box(corners,K,20,20)
        assert torch.isfinite(box).all()
        assert (box>=0).all() and (box<=1).all()
        if z<0:assert torch.equal(box,torch.zeros(4))


def test_cuboid_iou_analytic_and_rotated_cases():
    from scipy.spatial.transform import Rotation
    centers=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    dims=torch.full((3,3),2.)
    rot=torch.eye(3).repeat(3,1,1)
    iou=box3d_iou(centers,dims,rot,centers[:1],dims[:1],rot[:1])
    assert torch.allclose(iou[:,0],torch.tensor([1.,1/3,0.]),atol=1e-6)
    r=torch.tensor(Rotation.from_euler('xyz',[.2,.3,.4]).as_matrix(),dtype=torch.float32)[None]
    center=torch.zeros(1,3)
    small=torch.full((1,3),.2)
    contained=box3d_iou(center,dims[:1],rot[:1],center,small,r)
    assert contained.item()==pytest.approx(.001,abs=1e-7)
    same=box3d_iou(center,dims[:1],r,center,dims[:1],r)
    assert same.item()==pytest.approx(1.,abs=1e-6)
    a=box3d_iou(centers,dims,rot,center,dims[:1],r)
    b=box3d_iou(center,dims[:1],r,centers,dims,rot)
    assert torch.equal(a,b.T)
    assert torch.equal(a,box3d_iou(centers,dims,rot,center,dims[:1],r))


def make_tar(path,scene='1'):
    with tarfile.open(path,'w') as archive:
        for index in reversed(range(6)):
            image=io.BytesIO();Image.fromarray(raw_view(index)['img']).save(image,format='PNG')
            values={'wide/image.png':image.getvalue(),
                    'wide/image/K.json':json.dumps(raw_view()['intrinsics'].tolist()).encode(),
                    'wide/instances.json':json.dumps([] if index==0 else [dict(position=[0,0,3],scale=[1,1,1],R=np.eye(3).tolist())]).encode(),
                    'gt/RT.json':json.dumps(np.eye(4).tolist()).encode()}
            for name,value in values.items():
                info=tarfile.TarInfo(f'{scene}/{index*100000000}.{name}');info.size=len(value)
                archive.addfile(info,io.BytesIO(value))


def test_actual_tar_reader_empty_annotations_sorting_and_rank_sharding(tmp_path):
    a,b=tmp_path/'a.tar',tmp_path/'b.tar'
    make_tar(a,'1');make_tar(b,'2')
    frames=list(read_capture(a))
    assert [f['view']['timestamp'] for f in frames]==[0,.1,.2,.3,.4,.5]
    assert frames[0]['target']['center'].shape==(0,3)
    assert frames[1]['view']['camera_poses'].shape==(4,4)
    one=list(CA1MWindowDataset(tmp_path,rank=0,world_size=2).iter_frames())
    two=list(CA1MWindowDataset(tmp_path,rank=1,world_size=2).iter_frames())
    assert {f['view']['scene_id'] for f in one}=={'1'}
    assert {f['view']['scene_id'] for f in two}=={'2'}
    collator=CA1MCollator(PreprocessConfig(16,2,'identity'))
    views,targets=collator(list(CA1MWindowDataset(a))[-2:])
    assert len(targets)==2*len(views)


def test_frame_metrics_include_duplicates_empty_and_out_of_view_gt():
    gt=target();pred={k:v for k,v in gt.items() if k!='boxes2d'}
    pred['scores']=torch.tensor([.9])
    assert compute_ap_ar([pred],[gt],.5)=={'AP':1.,'AR':1.}
    empty={k:v[:0] for k,v in pred.items()}
    assert compute_ap_ar([empty],[gt],.5)=={'AP':0.,'AR':0.}
