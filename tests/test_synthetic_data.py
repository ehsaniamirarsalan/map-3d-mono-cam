import torch
from torch.utils.data import DataLoader

from mapdet3d.data.collate import collate_feature_samples
from mapdet3d.data.synthetic import SyntheticCuboidDataset


def _feature_shapes():
    return {"F_E": (8, 4, 4), "F_7": (8, 4, 4), "F_11": (8, 2, 2), "F_15": (8, 2, 2)}


def test_dataset_length_and_sample_shapes():
    ds = SyntheticCuboidDataset(num_scenes=5, feature_shapes=_feature_shapes(), max_objects=3, seed=0)
    assert len(ds) == 5

    sample = ds[0]
    assert set(sample["features"].keys()) == {"F_E", "F_7", "F_11", "F_15"}
    assert sample["features"]["F_E"].shape == (8, 4, 4)
    n = sample["target"]["boxes2d"].shape[0]
    assert 1 <= n <= 3
    assert sample["target"]["center"].shape == (n, 3)
    assert sample["target"]["dims"].shape == (n, 3)
    assert sample["target"]["rot"].shape == (n, 3, 3)
    assert (sample["target"]["center"][:, 2] > 0).all()  # positive depth


def test_dataset_is_deterministic_given_seed():
    ds_a = SyntheticCuboidDataset(num_scenes=3, feature_shapes=_feature_shapes(), seed=42)
    ds_b = SyntheticCuboidDataset(num_scenes=3, feature_shapes=_feature_shapes(), seed=42)
    for i in range(3):
        assert torch.allclose(ds_a[i]["features"]["F_E"], ds_b[i]["features"]["F_E"])
        assert torch.allclose(ds_a[i]["target"]["center"], ds_b[i]["target"]["center"])


def test_collate_and_dataloader_batching():
    ds = SyntheticCuboidDataset(num_scenes=6, feature_shapes=_feature_shapes(), max_objects=2, seed=1)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_feature_samples, shuffle=False)

    features, rho, targets = next(iter(loader))
    assert features["F_E"].shape == (3, 8, 4, 4)
    assert rho.shape == (3,)
    assert len(targets) == 3
    for t in targets:
        assert t["boxes2d"].shape[0] == t["center"].shape[0] == t["dims"].shape[0] == t["rot"].shape[0]
