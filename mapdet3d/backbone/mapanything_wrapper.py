"""Differentiable detection adapter for the pinned MapAnything implementation.

Only frozen encoding, multi-view fusion and scale decoding are executed. Dense
reconstruction heads are neither trained nor evaluated by this adapter.
"""
from __future__ import annotations

from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class MapAnythingBackbone(nn.Module):
    def __init__(self, pretrained="facebook/map-anything", freeze_encoder=True,
                 model=None, activation_checkpointing=False):
        super().__init__()
        if not freeze_encoder:
            raise ValueError("The paper preset requires frozen multimodal encoders")
        if model is None:
            try:
                from mapanything.models import MapAnything
            except ImportError as exc:
                raise ImportError("Install the pinned backbone: uv pip install -e third_party/map-anything") from exc
            model = MapAnything.from_pretrained(pretrained)
        self.model = model
        self.pretrained = pretrained
        self.activation_checkpointing = activation_checkpointing
        self.in_dims = dict(F_E=model.encoder.enc_embed_dim,
                            **{k: model.info_sharing.dim for k in ('F_7', 'F_11', 'F_15')})
        self.patch_size = int(model.encoder.patch_size)
        self.norm_type = model.encoder.data_norm_type
        self.model.requires_grad_(False)
        for name in ("info_sharing", "scale_head", "scale_adaptor", "scale_token"):
            getattr(self.model, name).requires_grad_(True)
        # MapAnything samples modality masks even in eval mode. Conditioning is
        # deterministic here; modality presence is controlled by the caller.
        if hasattr(model, "geometric_input_config"):
            model.geometric_input_config = dict(model.geometric_input_config)
            model.geometric_input_config.update(overall_prob=1.0, dropout_prob=0.0,
                                                ray_dirs_prob=1.0, cam_prob=1.0,
                                                depth_prob=0.0, depth_scale_norm_all_prob=0.0)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        for name in ("info_sharing", "scale_head", "scale_adaptor"):
            getattr(self.model, name).train(mode)
        return self

    def forward(self, views):
        from mapanything.utils.inference import preprocess_input_views_for_inference
        if not views:
            raise ValueError("At least one view is required")
        shape = views[0]['img'].shape
        if len(shape) != 4 or any(v['img'].shape != shape for v in views):
            raise ValueError("All views must share (B, C, H, W)")
        if any('camera_poses' in v for v in views) and 'camera_poses' not in views[0]:
            raise ValueError("Pose conditioning requires a pose for the first view")
        views = preprocess_input_views_for_inference(views)
        with torch.no_grad():
            features, registers = self.model._encode_n_views(views)
            with torch.autocast(device_type=views[0]['img'].device.type, enabled=False):
                features = self.model._encode_and_fuse_optional_geometric_inputs(
                    views, [f.float() for f in features])
        token = self.model.scale_token[None, :, None].expand(shape[0], -1, -1)

        def fuse(*inputs):
            out, intermediate = self.model.info_sharing(SimpleNamespace(
                features=list(inputs[:-1]), additional_input_tokens_per_view=registers,
                additional_input_tokens=inputs[-1]))
            if len(intermediate) != 2:
                raise ValueError("Backbone must expose transformer layers 7 and 11")
            return (*intermediate[0].features, *intermediate[1].features,
                    *out.features, out.additional_token_features)

        args = (*features, token)
        flat = checkpoint(fuse, *args, use_reentrant=False) if self.training and self.activation_checkpointing else fuse(*args)
        n = len(views)
        with torch.autocast(device_type=views[0]['img'].device.type, enabled=False):
            decoded = self.model.scale_head(SimpleNamespace(last_feature=flat[-1].float()))
            scale = self.model.scale_adaptor(SimpleNamespace(
                adaptor_feature=decoded.decoded_channels, output_shape_hw=shape[-2:])).value
        return dict(F_E=list(features), F_7=list(flat[:n]), F_11=list(flat[n:2*n]),
                    F_15=list(flat[2*n:3*n]), rho=scale.reshape(shape[0]), scale_token_features=flat[-1])
