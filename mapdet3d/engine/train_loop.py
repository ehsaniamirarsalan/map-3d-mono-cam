"""Training loop: runs a model + SetCriterion over a DataLoader, with
optional checkpointing. Optimizer parameter groups (frozen backbone /
LR-10 backbone submodules / full-LR detector+heads) are the caller's
responsibility when a real backbone is attached (see plan §4 Phase 6) --
this loop itself is backbone-agnostic and just optimizes whatever
parameters the given `model` exposes.

The loop is also *batch-format-agnostic*: a `batch_step` callback tells it
how to unpack one DataLoader batch, move it to `device`, call the model,
and return the criterion's losses. Two are provided:
  - `synthetic_batch_step` (default): the (features, rho, targets) 3-tuple
    produced by `mapdet3d.data.collate.collate_feature_samples`, calling
    `MapDet3DCore(features, rho)` -- used by tools/train.py.
  - `views_batch_step`: the (views, targets) 2-tuple produced by
    `mapdet3d.data.ca1m.collate.collate_ca1m_batch`, calling
    `MapDet3D(views)` (real backbone + core) -- used by tools/train_real.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from mapdet3d.losses.criterion import SetCriterion

BatchStepFn = Callable[[torch.nn.Module, SetCriterion, Any, torch.device], dict[str, torch.Tensor]]


def synthetic_batch_step(
    model: torch.nn.Module, criterion: SetCriterion, batch: Any, device: torch.device
) -> dict[str, torch.Tensor]:
    features, rho, targets = batch
    features = {k: v.to(device) for k, v in features.items()}
    rho = rho.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    out = model(features, rho)
    return criterion(out, targets)


def views_batch_step(
    model: torch.nn.Module, criterion: SetCriterion, batch: Any, device: torch.device
) -> dict[str, torch.Tensor]:
    views, targets = batch
    views = [
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in view.items()}
        for view in views
    ]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    out = model(views)
    return criterion(out, targets)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: SetCriterion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_step: BatchStepFn = synthetic_batch_step,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    num_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()
        losses = batch_step(model, criterion, batch, device)
        losses["loss_total"].backward()
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in totals.items()}


def train(
    model: torch.nn.Module,
    criterion: SetCriterion,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    checkpoint_dir: str | Path | None = None,
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
    batch_step: BatchStepFn = synthetic_batch_step,
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint_state_dict_fn: Callable[[torch.nn.Module], dict] | None = None,
) -> list[dict[str, float]]:
    """
    Args:
        optimizer: pre-built optimizer (e.g. with per-parameter-group LRs
            for a frozen/reduced-LR backbone, plan §4 Phase 6's freeze
            scheme). If omitted, builds `Adam(model.parameters(), lr=lr)`.
        checkpoint_state_dict_fn: what to save per checkpoint (defaults to
            `model.state_dict()`). Pass e.g. `lambda m: m.core.state_dict()`
            when wrapping a large frozen backbone you don't want to
            re-serialize into every checkpoint file.
    """
    if checkpoint_state_dict_fn is None:
        checkpoint_state_dict_fn = lambda m: m.state_dict()  # noqa: E731
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))

    history = []
    for epoch in range(num_epochs):
        epoch_losses = train_one_epoch(model, criterion, dataloader, optimizer, device, batch_step=batch_step)
        scheduler.step()
        history.append(epoch_losses)
        if log_fn is not None:
            log_fn(epoch, epoch_losses)

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict_fn(model),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )

    return history
