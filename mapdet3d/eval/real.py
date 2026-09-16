"""Frame and scene evaluation over the shared image inference path."""
import torch
from mapdet3d.eval.ap_ar import compute_ap_ar
from mapdet3d.tracking.tracker import Tracker

THRESHOLDS=(.15,.25,.5)


def metrics(preds,targets):
    if not targets:
        raise ValueError('No evaluation samples were produced')
    return {str(t):compute_ap_ar(preds,targets,t) for t in THRESHOLDS}


def evaluate_frames(predictor,frames):
    predictions,targets=[],[]
    for frame in frames:
        predictions.append(predictor.step(frame['view']))
        targets.append(frame['target'])
    return dict(mode='frame',num_frames=len(targets),metrics=metrics(predictions,targets),
                interpolation='101-point',max_detections=predictor.max_detections,score_threshold=predictor.score_threshold)


def tracks_as_prediction(tracks):
    if not tracks:
        return dict(center=torch.empty(0,3),dims=torch.empty(0,3),rot=torch.empty(0,3,3),scores=torch.empty(0))
    return dict(center=torch.stack([t.center for t in tracks]).cpu(),
                dims=torch.stack([t.dims for t in tracks]).cpu(),rot=torch.stack([t.rot for t in tracks]).cpu(),
                scores=torch.tensor([t.score for t in tracks]))


def evaluate_scenes(predictor,scenes,iou_threshold=.25):
    predictions,targets=[],[]
    for scene,target,frames in scenes:
        predictor.reset()
        tracker=Tracker(iou_threshold=iou_threshold)
        count=0
        for frame in frames:
            view=frame['view']
            if 'camera_poses' not in view:
                raise ValueError(f'Scene {scene} requires consistent camera-to-world poses')
            pred=predictor.step(view)
            tracker.step(pred['center'],pred['dims'],pred['rot'],view['camera_poses'].cpu(),pred['scores'])
            count+=1
        if not count:
            raise ValueError(f'Scene {scene} contains no evaluation frames')
        predictions.append(tracks_as_prediction(tracker.tracks));targets.append(target)
    return dict(mode='scene',num_scenes=len(targets),metrics=metrics(predictions,targets),
                interpolation='101-point',association_iou=iou_threshold,
                max_detections=predictor.max_detections,score_threshold=predictor.score_threshold)
