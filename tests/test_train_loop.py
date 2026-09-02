"""Tests that train_loop's two batch_step implementations correctly unpack
their respective batch formats and drive one optimizer step, for both the
synthetic (features, rho, targets) path and the real (views, targets) path.
"""

import torch
from torch.utils.data import DataLoader

from mapdet3d.data.collate import collate_feature_samples
from mapdet3d.data.synthetic import SyntheticCuboidDataset
from mapdet3d.engine.train_loop import synthetic_batch_step, train, views_batch_step
from mapdet3d.losses.criterion import SetCriterion
from mapdet3d.models.mapdet3d import MapDet3DCore


class _FakeViewsModel(torch.nn.Module):
    """A tiny stand-in for MapDet3D: takes `views` (list of dicts with
    "img") instead of (features, rho), proving views_batch_step's unpacking
    and model-call convention without needing the real backbone."""

    def __init__(self, num_queries: int = 3):
        super().__init__()
        self.num_queries = num_queries
        self.center_head = torch.nn.Linear(1, 3)
        self.dims_head = torch.nn.Linear(1, 3)
        self.rot_head = torch.nn.Linear(1, 9)
        self.logit_head = torch.nn.Linear(1, 1)

    def forward(self, views: list[dict]) -> dict:
        batch_size = views[0]["img"].shape[0]
        dummy = torch.ones(batch_size, self.num_queries, 1)
        rot = self.rot_head(dummy).view(batch_size, self.num_queries, 3, 3)
        # Orthonormalize loosely via QR so it's a plausible rotation-ish matrix.
        rot = torch.linalg.qr(rot)[0]
        layer_pred = {
            "logits": self.logit_head(dummy),
            "boxes2d": torch.sigmoid(dummy.expand(-1, -1, 4)),
            "center": self.center_head(dummy),
            "dims": torch.exp(self.dims_head(dummy)),
            "rot": rot,
        }
        return {"layer_preds": [layer_pred]}


def _make_views_dataloader(batch_size=2, num_batches=2):
    def gen():
        for _ in range(num_batches):
            views = [{"img": torch.randn(batch_size, 3, 4, 4)}]
            targets = [
                {
                    "boxes2d": torch.rand(1, 4),
                    "center": torch.randn(1, 3),
                    "dims": torch.rand(1, 3) + 0.5,
                    "rot": torch.eye(3).unsqueeze(0),
                }
                for _ in range(batch_size)
            ]
            yield views, targets

    return list(gen())


def test_synthetic_batch_step_computes_finite_loss_and_calls_model_with_features_rho():
    torch.manual_seed(0)
    ds = SyntheticCuboidDataset(
        num_scenes=4,
        feature_shapes={"F_E": (8, 4, 4), "F_7": (8, 4, 4), "F_11": (8, 2, 2), "F_15": (8, 2, 2)},
        max_objects=2,
        seed=0,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_feature_samples)
    model = MapDet3DCore(
        in_dims={"F_E": 8, "F_7": 8, "F_11": 8, "F_15": 8},
        num_queries=3,
        num_decoder_layers=1,
        base_sizes=[0.3, 0.3, 0.3, 0.3],
        d_model=16,
        n_heads=2,
        n_points=2,
        d_ffn=32,
    )
    criterion = SetCriterion()
    batch = next(iter(loader))

    losses = synthetic_batch_step(model, criterion, batch, torch.device("cpu"))
    assert torch.isfinite(losses["loss_total"])


def test_views_batch_step_computes_finite_loss_and_calls_model_with_views_only():
    torch.manual_seed(1)
    model = _FakeViewsModel()
    criterion = SetCriterion()
    views, targets = _make_views_dataloader(batch_size=2, num_batches=1)[0]

    losses = views_batch_step(model, criterion, (views, targets), torch.device("cpu"))
    assert torch.isfinite(losses["loss_total"])


def test_train_runs_end_to_end_with_views_batch_step():
    torch.manual_seed(2)
    model = _FakeViewsModel()
    criterion = SetCriterion()
    batches = _make_views_dataloader(batch_size=2, num_batches=3)

    history = train(
        model, criterion, batches, num_epochs=2, lr=1e-2,
        device=torch.device("cpu"), batch_step=views_batch_step,
    )
    assert len(history) == 2
    for epoch_losses in history:
        assert torch.isfinite(torch.tensor(epoch_losses["loss_total"]))
