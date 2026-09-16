"""Small differentiable backbone implementing the pinned adapter protocol."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone
from mapdet3d.configs.real_config import ModelConfig,build_real_model

class Encoder(nn.Conv2d):
    enc_embed_dim=8
    patch_size=2
    data_norm_type='identity'
    def __init__(self):
        super().__init__(3,8,2,stride=2)

class Fusion(nn.Module):
    dim=8
    def __init__(self):
        super().__init__()
        self.proj=nn.Conv2d(8,8,1)
    def forward(self,inputs):
        mixed=torch.stack(inputs.features).mean(0)
        values=[self.proj(f+mixed) for f in inputs.features]
        token=inputs.additional_input_tokens+torch.stack(values).mean((0,3,4))[...,None]
        return NS(features=values,additional_token_features=token),[NS(features=[f+7 for f in values]),NS(features=[f+11 for f in values])]

class ScaleHead(nn.Module):
    def __init__(self):
        super().__init__();self.fc=nn.Conv1d(8,1,1)
    def forward(self,inputs):
        return NS(decoded_channels=self.fc(inputs.last_feature))

class ScaleAdaptor(nn.Module):
    def forward(self,inputs):
        return NS(value=inputs.adaptor_feature.exp())

class TinyMapAnything(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder();self.info_sharing=Fusion()
        self.scale_token=nn.Parameter(torch.zeros(8))
        self.scale_head=ScaleHead();self.scale_adaptor=ScaleAdaptor()
        self.fusion_norm_layer=nn.Identity()
        self.geometric_input_config={}
        self.seen=None
    def _encode_n_views(self,views):
        encoded=self.encoder(torch.cat([v['img'] for v in views]))
        return list(encoded.chunk(len(views))),None
    def _encode_and_fuse_optional_geometric_inputs(self,views,features):
        self.seen=views
        out=[]
        for view,feature in zip(views,features):
            extra=0.
            if 'ray_directions_cam' in view:
                extra=view['ray_directions_cam'][...,0].mean((1,2))[:,None,None,None]
            if 'camera_pose_trans' in view:
                extra=extra+view['camera_pose_trans'].sum(-1)[:,None,None,None]
            out.append(feature+extra)
        return out


def tiny_model(checkpointing=False):
    backbone=MapAnythingBackbone(model=TinyMapAnything(),activation_checkpointing=checkpointing)
    config=ModelConfig(num_queries=4,num_decoder_layers=1,d_model=16,n_heads=2,n_points=2,d_ffn=32,long_edge=16,activation_checkpointing=checkpointing)
    return build_real_model(config,backbone)


def target(x=0.):
    return dict(boxes2d=torch.tensor([[.5,.5,.3,.3]]),center=torch.tensor([[x,0.,3.]]),
                dims=torch.tensor([[1.,1.,1.]]),rot=torch.eye(3)[None])


def raw_view(index=0,scene='one'):
    return dict(img=np.full((12,16,3),80+index,dtype=np.uint8),intrinsics=torch.tensor([[12.,0,8],[0,12,6],[0,0,1.]]),
                camera_poses=torch.eye(4),timestamp=index/10,frame_id=str(index),scene_id=scene)
