"""Train Map-Det3D on real CA-1M data with the real MapAnything backbone.

Requires CA-1M tar shards downloaded locally (see README's "Getting real
CA-1M data" section) and, on first run, downloads the pretrained MapAnything
checkpoint (multi-GB) from Hugging Face.

Usage:
    python tools/train_real.py --data data/ca1m --epochs 5 --checkpoint-dir runs/ca1m
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

try:
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

import torch
from torch.utils.data import DataLoader

from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone
from mapdet3d.data.ca1m.collate import collate_ca1m_batch
from mapdet3d.data.ca1m.dataset import CA1MWindowDataset
from mapdet3d.engine.train_loop import train, views_batch_step
from mapdet3d.losses.criterion import SetCriterion
from mapdet3d.models.mapdet3d import MapDet3D, MapDet3DCore


def resolve_data_source(data_path: str) -> str | list[str]:
    """Accepts a single tar file, a directory of tars, or a .txt link-list."""
    p = Path(data_path)
    if p.is_dir():
        tars = sorted(glob.glob(str(p / "*.tar")))
        if not tars:
            raise FileNotFoundError(f"No .tar files found in {data_path}")
        return tars
    return str(p)


def build_real_model(num_queries: int, num_decoder_layers: int, pretrained: str) -> MapDet3D:
    backbone = MapAnythingBackbone(pretrained=pretrained, freeze_encoder=True)
    # Confirmed via the real checkpoint: encoder dim 1536, transformer dim
    # 1536, indices [7, 11] (see tools/infer_real.py's comment / README).
    in_dims = {"F_E": 1536, "F_7": 1536, "F_11": 1536, "F_15": 1536}
    core = MapDet3DCore(
        in_dims=in_dims,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        base_sizes=[0.4, 0.3, 0.2, 0.1],
        d_model=256,
        n_heads=8,
        n_points=4,
        d_ffn=1024,
    )
    return MapDet3D(backbone, core)


def build_optimizer(model: MapDet3D, base_lr: float) -> torch.optim.Optimizer:
    """Freeze the MapAnything encoder; fine-tune its multi-view transformer
    and scale head at LR/10; train the detector+heads at full LR (paper
    Sec. 4.1's optimizer configuration)."""
    backbone = model.backbone.model
    encoder_params = list(backbone.encoder.parameters())
    reduced_lr_params = list(backbone.info_sharing.parameters())
    if hasattr(backbone, "scale_head"):
        reduced_lr_params += list(backbone.scale_head.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    reduced_lr_param_ids = {id(p) for p in reduced_lr_params}

    for p in encoder_params:
        p.requires_grad_(False)

    full_lr_params = [
        p
        for p in model.parameters()
        if id(p) not in encoder_param_ids and id(p) not in reduced_lr_param_ids
    ]

    param_groups = [
        {"params": [p for p in reduced_lr_params if p.requires_grad], "lr": base_lr / 10.0},
        {"params": full_lr_params, "lr": base_lr},
    ]
    return torch.optim.Adam(param_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Map-Det3D on real CA-1M data")
    parser.add_argument("--data", type=str, required=True, help="CA-1M tar file, directory, or .txt link-list")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained", type=str, default="facebook/map-anything-apache",
        help="HF checkpoint id; use 'facebook/map-anything' for CC-BY-NC research weights",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    source = resolve_data_source(args.data)
    dataset = CA1MWindowDataset(source, window_size=args.window_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_ca1m_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU detected. Training the real backbone on CPU will be extremely slow.")

    model = build_real_model(args.num_queries, args.num_decoder_layers, args.pretrained)
    optimizer = build_optimizer(model, args.lr)
    criterion = SetCriterion()

    def log_fn(epoch: int, losses: dict[str, float]) -> None:
        print(f"epoch {epoch:3d} | loss_total={losses['loss_total']:.4f} | "
              f"loss_2d={losses['loss_2d']:.4f} | loss_3d={losses['loss_3d']:.4f}")

    train(
        model,
        criterion,
        dataloader,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_fn=log_fn,
        batch_step=views_batch_step,
        optimizer=optimizer,
        # Only the trainable core (detector + heads) needs saving -- the
        # backbone is either frozen or reloadable from `--pretrained`.
        checkpoint_state_dict_fn=lambda m: m.core.state_dict(),
    )


if __name__ == "__main__":
    main()
