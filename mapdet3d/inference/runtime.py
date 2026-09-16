"""Shared real-image loading, streaming predictions, and output conversion."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from mapdet3d.data.preprocess import preprocess_view
from mapdet3d.inference.sliding_window import SlidingWindowInference
from mapdet3d.utils.geometry import corners_from_box


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',str(path))]


def iter_images(path,metadata=None):
    path=Path(path)
    metadata=json.loads(Path(metadata).read_text()) if metadata else {}
    if path.is_dir():
        files=sorted([p for p in path.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')],key=natural_key)
        if not files:
            raise ValueError(f'No RGB images found in {path}')
        for index,file in enumerate(files):
            info=metadata.get(file.name,{})
            yield dict(img=np.array(Image.open(file).convert('RGB')),frame_id=file.name,
                       scene_id=info.get('scene_id',path.name),timestamp=info.get('timestamp',float(index)),
                       **{k:torch.as_tensor(info[k],dtype=torch.float32) for k in ('intrinsics','camera_poses') if k in info})
    else:
        import cv2
        capture=cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f'Cannot open video {path}')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:
            raise ValueError('Video must have a valid frame rate')
        index=0
        try:
            while True:
                ok,image=capture.read()
                if not ok:
                    break
                info=metadata.get(str(index),{})
                yield dict(img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),frame_id=str(index),
                           scene_id=info.get('scene_id',path.stem),timestamp=index/fps,
                           **{k:torch.as_tensor(info[k],dtype=torch.float32) for k in ('intrinsics','camera_poses') if k in info})
                index+=1
        finally:
            capture.release()
        if index==0:
            raise ValueError('Video contains no decodable frames')


class StreamingPredictor:
    def __init__(self,model,device,window=5,max_detections=100,score_threshold=0.):
        if max_detections<1 or not 0<=score_threshold<=1:
            raise ValueError('Invalid detection limit or confidence threshold')
        self.model=model.to(device).eval()
        self.device=device
        self.stream=SlidingWindowInference(model,window)
        self.max_detections=max_detections
        self.score_threshold=score_threshold
        self.output_hw=None
        self.scene=None

    def reset(self):
        self.stream.reset()
        self.output_hw=None
        self.scene=None

    @torch.no_grad()
    def step(self,view):
        if self.scene != view.get('scene_id'):
            self.reset()
            self.scene=view.get('scene_id')
        prepared,_=preprocess_view(view,self.model.preprocess_config,self.output_hw)
        self.output_hw=prepared['img'].shape[-2:]
        prepared={k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in prepared.items()}
        out=self.stream.step(prepared)
        scores=out['logits'][0,:,0].sigmoid()
        order=torch.argsort(scores,descending=True,stable=True)[:self.max_detections]
        order=order[scores[order]>=self.score_threshold]
        result={k:out[k][0,order].detach().cpu() for k in ('boxes2d','center','dims','rot')}
        result['scores']=scores[order].cpu()
        if any(not torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Model produced nonfinite detections')
        return result


def serialize_prediction(pred,view):
    return dict(frame_id=view.get('frame_id'),scene_id=view.get('scene_id'),timestamp=view.get('timestamp'),
                coordinate_frame='opencv_camera',units='meters',
                boxes2d_coordinate_frame='normalized_padded_image_cxcywh',
                **{k:v.tolist() for k,v in pred.items()})


def save_overlay(path,view,pred):
    if 'intrinsics' not in view:
        raise ValueError('Projected overlays require camera intrinsics')
    image=Image.fromarray(np.asarray(view['img']).astype(np.uint8))
    draw=ImageDraw.Draw(image)
    K=torch.as_tensor(view['intrinsics'])
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for corners in corners_from_box(pred['center'],pred['dims'],pred['rot']):
        for i,j in edges:
            a,b=corners[i].clone(),corners[j].clone()
            if a[2]<=1e-4 and b[2]<=1e-4:
                continue
            if a[2]<1e-4:
                a=a+(b-a)*((1e-4-a[2])/(b[2]-a[2]))
            if b[2]<1e-4:
                b=b+(a-b)*((1e-4-b[2])/(a[2]-b[2]))
            uv=torch.stack([K@a,K@b]); uv=uv[:,:2]/uv[:,2:]
            draw.line([tuple(uv[0].tolist()),tuple(uv[1].tolist())],fill=(0,220,130),width=2)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
