import copy
import pytest
import torch
pytest.importorskip('mapanything.utils.inference')
from tests.fixtures import tiny_model,raw_view,target
from mapdet3d.configs.real_config import build_optimizer
from mapdet3d.data.preprocess import collate_windows
from mapdet3d.engine.checkpoint import load_checkpoint,save_checkpoint
from mapdet3d.engine.steps import train_steps
from mapdet3d.losses.criterion import SetCriterion
from mapdet3d.inference.runtime import StreamingPredictor,serialize_prediction,save_overlay
from mapdet3d.eval.real import evaluate_frames,evaluate_scenes


def test_image_training_reload_stream_and_both_evaluators(tmp_path):
    torch.manual_seed(12)
    model=tiny_model()
    frames=[dict(view=raw_view(i),target=target()) for i in range(3)]
    batch=collate_windows([dict(views=[f['view'] for f in frames],targets=[f['target'] for f in frames])],model.preprocess_config)
    optimizer=build_optimizer(model,lr=.005)
    history=train_steps(model,SetCriterion(),[batch],optimizer,torch.device('cpu'),max_steps=60,accumulation=1,
                        checkpoint_dir=tmp_path,checkpoint_every=60,amp=False)
    assert history[-1]['loss_total']<history[0]['loss_total']*.8
    restored=tiny_model();load_checkpoint(tmp_path/'step_000060.pt',restored)
    model.eval();restored.eval()
    assert torch.equal(model(batch[0])['layer_preds'][-1]['center'],restored(batch[0])['layer_preds'][-1]['center'])
    predictor=StreamingPredictor(restored,torch.device('cpu'),window=2)
    result=evaluate_frames(predictor,frames)
    assert result['num_frames']==3
    assert set(result['metrics'])=={'0.15','0.25','0.5'}
    scene=evaluate_scenes(predictor,[('one',target(),iter(frames))])
    assert scene['num_scenes']==1
    predictor.reset();pred=predictor.step(raw_view())
    assert serialize_prediction(pred,raw_view())['units']=='meters'
    save_overlay(tmp_path/'overlay.png',raw_view(),pred)
    assert (tmp_path/'overlay.png').exists()


def test_resume_reproduces_uninterrupted_updates(tmp_path):
    torch.manual_seed(5)
    model=tiny_model()
    batch=collate_windows([dict(views=[raw_view()],targets=[target()])],model.preprocess_config)
    train_steps(model,SetCriterion(),[batch],build_optimizer(model,.001),torch.device('cpu'),
                max_steps=10,accumulation=2,checkpoint_dir=tmp_path,checkpoint_every=5,amp=False)
    expected=copy.deepcopy(model.state_dict())
    resumed=tiny_model()
    train_steps(resumed,SetCriterion(),[batch],build_optimizer(resumed,.001),torch.device('cpu'),
                max_steps=10,accumulation=2,resume=tmp_path/'step_000005.pt',amp=False)
    for k,v in resumed.state_dict().items():
        assert torch.equal(v,expected[k]),k


def test_rejects_legacy_or_mismatched_checkpoint(tmp_path):
    path=tmp_path/'old.pt';torch.save({'model_state':{}},path)
    with pytest.raises(ValueError,match='Incompatible'):
        load_checkpoint(path,tiny_model())
    model=tiny_model();save_checkpoint(path,model)
    model.config['num_queries']=999
    with pytest.raises(ValueError,match='configuration'):
        load_checkpoint(path,model)


def test_real_uniception_transformer_and_scale_interfaces():
    from uniception.models.info_sharing.alternating_attention_transformer import MultiViewAlternatingAttentionTransformerIFR
    from uniception.models.prediction_heads.mlp_head import MLPHead
    from uniception.models.prediction_heads.adaptors import ScaleAdaptor
    from tests.fixtures import TinyMapAnything
    from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone
    from mapdet3d.data.preprocess import preprocess_view,PreprocessConfig
    raw=TinyMapAnything()
    raw.info_sharing=MultiViewAlternatingAttentionTransformerIFR(name='contract_test',input_embed_dim=8,dim=8,depth=16,num_heads=2,indices=[7,11])
    raw.scale_head=MLPHead(input_feature_dim=8,output_dim=1,hidden_dim=8)
    raw.scale_adaptor=ScaleAdaptor(name='scale',mode='exp')
    backbone=MapAnythingBackbone(model=raw,activation_checkpointing=True)
    view,_=preprocess_view(raw_view(),PreprocessConfig(8,2,'identity'))
    output=backbone([view,view])
    (output['F_15'][0].square().mean()+output['rho'].sum()).backward()
    assert raw.scale_token.grad is not None
    assert output['F_7'][0].shape==output['F_15'][0].shape
