"""One construction path for real training, evaluation and inference."""
from dataclasses import dataclass, asdict, field
import torch
from mapdet3d.data.preprocess import PreprocessConfig
from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone
from mapdet3d.models.mapdet3d import MapDet3D, MapDet3DCore

@dataclass
class ModelConfig:
    pretrained: str = 'facebook/map-anything'
    num_queries: int = 900
    num_decoder_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    n_points: int = 4
    d_ffn: int = 1024
    base_sizes: list = field(default_factory=lambda:[.05,.1,.2,.4])
    long_edge: int = 518
    use_intrinsics: bool = True
    use_poses: bool = True
    activation_checkpointing: bool = True


def build_real_model(config: ModelConfig, backbone=None):
    backbone = backbone or MapAnythingBackbone(config.pretrained,activation_checkpointing=config.activation_checkpointing)
    core = MapDet3DCore(in_dims=backbone.in_dims,num_queries=config.num_queries,
                       num_decoder_layers=config.num_decoder_layers,base_sizes=config.base_sizes,
                       d_model=config.d_model,n_heads=config.n_heads,n_points=config.n_points,d_ffn=config.d_ffn)
    # Encoder-stage objectness comes from the proposal classifier.
    core.box3d_heads[0].conf_mlp.requires_grad_(False)
    model = MapDet3D(backbone,core)
    model.config = asdict(config)
    model.preprocess_config = PreprocessConfig(config.long_edge,backbone.patch_size,backbone.norm_type)
    model.config['preprocess'] = asdict(model.preprocess_config)
    return model


def config_from_checkpoint(state):
    return ModelConfig(**{k:v for k,v in state['config'].items() if k!='preprocess'})


def build_optimizer(model,lr=1e-4,weight_decay=1e-4):
    return torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=lr/10),
        dict(params=[p for p in model.core.parameters() if p.requires_grad],lr=lr),
    ],weight_decay=weight_decay)


def select_device(requested='auto'):
    # CUDA is the production target; CPU keeps portable integration tests usable.
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    return device
