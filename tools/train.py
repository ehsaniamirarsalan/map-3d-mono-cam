"""End-to-end training entry point on the synthetic dataset (see plan §4
Phase 5 fallback + README). Usage:

    python tools/train.py --epochs 5 --num-scenes 40 --checkpoint-dir runs/smoke

This exercises the full trainable pipeline we control (data loading,
MapDet3DCore forward/backward, SetCriterion, checkpointing) without
requiring the real CA-1M/ScanNet datasets or the real pretrained
MapAnything backbone (see mapdet3d/data/synthetic.py's docstring).
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mapdet3d.configs.smoke_config import FEATURE_SHAPES, build_model
from mapdet3d.data.collate import collate_feature_samples
from mapdet3d.data.synthetic import SyntheticCuboidDataset
from mapdet3d.engine.train_loop import train
from mapdet3d.losses.criterion import SetCriterion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Map-Det3D on the synthetic dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-scenes", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    dataset = SyntheticCuboidDataset(
        num_scenes=args.num_scenes,
        feature_shapes=FEATURE_SHAPES,
        max_objects=args.max_objects,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_feature_samples
    )

    model = build_model(args.num_queries, args.num_decoder_layers)
    criterion = SetCriterion()

    def log_fn(epoch: int, losses: dict[str, float]) -> None:
        print(f"epoch {epoch:3d} | loss_total={losses['loss_total']:.4f} | "
              f"loss_2d={losses['loss_2d']:.4f} | loss_3d={losses['loss_3d']:.4f}")

    train(
        model,
        criterion,
        dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=torch.device("cpu"),
        checkpoint_dir=args.checkpoint_dir,
        log_fn=log_fn,
    )


if __name__ == "__main__":
    main()
