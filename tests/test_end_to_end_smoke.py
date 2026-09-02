"""End-to-end synthetic smoke test (plan §4 Phase 6 / §8 verification):
overfit MapDet3DCore + SetCriterion on a single fixed synthetic sample and
confirm the loss drops substantially. This is the cheapest full-pipeline
correctness signal before any real data or the real (large, slow) backbone
are involved.
"""

import torch

from mapdet3d.losses.criterion import SetCriterion
from mapdet3d.models.mapdet3d import MapDet3DCore


def test_overfits_a_single_synthetic_sample():
    torch.manual_seed(0)

    in_dims = {"F_E": 8, "F_7": 8, "F_11": 8, "F_15": 8}
    core = MapDet3DCore(
        in_dims=in_dims,
        num_queries=4,
        num_decoder_layers=2,
        base_sizes=[0.4, 0.3, 0.2, 0.1],
        d_model=32,
        n_heads=4,
        n_points=2,
        d_ffn=64,
    )
    criterion = SetCriterion()
    optimizer = torch.optim.Adam(core.parameters(), lr=1e-3)

    features = {
        "F_E": torch.randn(1, 8, 4, 4),
        "F_7": torch.randn(1, 8, 4, 4),
        "F_11": torch.randn(1, 8, 2, 2),
        "F_15": torch.randn(1, 8, 2, 2),
    }
    rho = torch.ones(1)

    target = {
        "boxes2d": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "center": torch.tensor([[0.1, -0.1, 5.0]]),
        "dims": torch.tensor([[1.0, 2.0, 1.5]]),
        "rot": torch.eye(3).unsqueeze(0),
    }

    losses_over_time = []
    for _ in range(300):
        optimizer.zero_grad()
        out = core(features, rho)
        losses = criterion(out["layer_preds"], [target])
        loss = losses["loss_total"]
        loss.backward()
        optimizer.step()
        losses_over_time.append(loss.item())

    initial_loss = sum(losses_over_time[:5]) / 5
    final_loss = sum(losses_over_time[-5:]) / 5

    assert torch.isfinite(torch.tensor(losses_over_time)).all()
    assert final_loss < initial_loss * 0.5, (
        f"loss did not decrease substantially: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
