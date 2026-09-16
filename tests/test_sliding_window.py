import pytest
import torch
pytest.importorskip('mapanything.utils.inference')
from tests.fixtures import tiny_model,raw_view
from mapdet3d.data.preprocess import preprocess_view
from mapdet3d.inference.sliding_window import SlidingWindowInference


def test_window_output_and_scene_reset():
    model=tiny_model();stream=SlidingWindowInference(model,3)
    for i in range(5):
        view,_=preprocess_view(raw_view(i),model.preprocess_config)
        out=stream.step(view)
        assert len(stream.buffer)==min(i+1,3)
        assert out['center'].shape==(1,4,3)
    view,_=preprocess_view(raw_view(0,'two'),model.preprocess_config)
    stream.step(view)
    assert len(stream.buffer)==1
    stream.reset();assert len(stream.buffer)==0


def test_batched_last_view_slice_and_causality():
    model=tiny_model();stream=SlidingWindowInference(model,3)
    for i in range(3):
        view,_=preprocess_view(raw_view(i),model.preprocess_config)
        view={k:v.repeat(2,*([1]*(v.ndim-1))) if isinstance(v,torch.Tensor) else v for k,v in view.items()}
        out=stream.step(view)
    direct=model(stream.buffer)['layer_preds'][-1]
    assert torch.allclose(out['center'],direct['center'][-2:])
    saved=out['center'].clone()
    view['timestamp']=.4;stream.step(view)
    assert torch.equal(out['center'],saved)
    with pytest.raises(ValueError,match='timestamps'):
        stream.step(view)


def test_invalid_window():
    with pytest.raises(ValueError): SlidingWindowInference(tiny_model(),0)
