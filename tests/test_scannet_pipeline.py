import json
import numpy as np
import pytest
import torch
from PIL import Image
from mapdet3d.data.scannet.dataset import ScanNetDataset,PAPER_SCENES
from mapdet3d.eval.real import evaluate_scenes
from mapdet3d.tracking.tracker import Tracker
from tests.fixtures import raw_view,target


def test_paper_scene_list():
    scenes=PAPER_SCENES.read_text().split()
    assert len(scenes)==len(set(scenes))==100


def test_aligned_scannet_boxes_transform_and_sampling(tmp_path):
    scene='scene0000_00'
    folder=tmp_path/'data'/scene/'frames'
    for name in ('color','pose','intrinsic'):(folder/name).mkdir(parents=True)
    np.savetxt(folder/'intrinsic'/'intrinsic_color.txt',np.eye(4))
    for index in range(26):
        Image.fromarray(raw_view()['img']).save(folder/'color'/f'{index}.jpg')
        pose=np.eye(4);pose[0,3]=1
        np.savetxt(folder/'pose'/f'{index}.txt',pose)
    alignment=np.eye(4);alignment[0,3]=10
    for name,label in [('scannet200_instance_data',2),('scannet_instance_data',3)]:
        annotations=tmp_path/name;annotations.mkdir()
        # Second annotation lies behind the camera and must remain in GT.
        np.save(annotations/f'{scene}_aligned_bbox.npy',np.array([[11,0,3,1,1,1,label],[11,0,-3,1,1,1,label]],np.float32))
        np.save(annotations/f'{scene}_axis_align_matrix.npy',alignment)
    split=tmp_path/'scenes.txt';split.write_text(scene+'\n')
    dataset=ScanNetDataset(tmp_path,split)
    frames=list(dataset.iter_frames())
    assert [f['view']['frame_id'] for f in frames]==['0','25']
    assert len(frames[0]['target']['center'])==2
    assert torch.equal(frames[0]['target']['center'],torch.tensor([[0.,0.,3.],[0.,0.,-3.]]))
    assert frames[0]['view']['camera_poses'][0,3]==11
    assert len(next(ScanNetDataset(tmp_path,split,False).iter_scenes())[1]['center'])==2


def test_tracker_replaces_larger_box_with_its_score_and_resets():
    tracker=Tracker()
    pose=torch.eye(4);pose[0,3]=2
    tracker.step(torch.tensor([[0.,0.,3.]]),torch.ones(1,3),torch.eye(3)[None],pose,torch.tensor([.9]))
    assert tracker.tracks[0].center[0]==2
    tracker.step(torch.tensor([[0.,0.,3.]]),torch.full((1,3),1.1),torch.eye(3)[None],pose,torch.tensor([.8]))
    assert len(tracker.tracks)==1
    assert tracker.tracks[0].score==pytest.approx(.8)
    tracker.reset();assert not tracker.tracks


def test_scene_metric_and_common_world_transform():
    class Predictor:
        max_detections=100
        score_threshold=0.
        def reset(self):pass
        def step(self,view):
            pred=target();pred['center']=pred['center']-view['camera_poses'][:3,3]
            pred['scores']=torch.tensor([.9]);return pred
    frames=[]
    for i in range(2):
        view=raw_view(i);view['camera_poses'][0,3]=i
        frames.append(dict(view=view,target=target()))
    result=evaluate_scenes(Predictor(),[('one',target(),frames)])
    assert result['metrics']['0.5']=={'AP':1.,'AR':1.}
