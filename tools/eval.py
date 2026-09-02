"""End-to-end evaluation entry point on the synthetic dataset. Usage:

    python tools/eval.py --checkpoint runs/smoke/epoch_0004.pt --num-scenes 20

Prints class-agnostic 3D AP/AR at IoU thresholds {0.15, 0.25, 0.50} (paper
Sec. 4.2's convention), computed with mapdet3d.eval.ap_ar / box3d_iou.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from mapdet3d.configs.smoke_config import FEATURE_SHAPES, build_model
from mapdet3d.data.collate import collate_feature_samples
from mapdet3d.data.synthetic import SyntheticCuboidDataset
from mapdet3d.eval.eval_per_frame import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Map-Det3D on the synthetic dataset")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)  # different seed than training data
    parser.add_argument("--num-samples-iou", type=int, default=5000)
    args = parser.parse_args()

    dataset = SyntheticCuboidDataset(
        num_scenes=args.num_scenes,
        feature_shapes=FEATURE_SHAPES,
        max_objects=args.max_objects,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_feature_samples
    )

    model = build_model(args.num_queries, args.num_decoder_layers)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    results = evaluate(
        model, dataloader, iou_thresholds=[0.15, 0.25, 0.5],
        device=torch.device("cpu"), num_samples_iou=args.num_samples_iou,
    )

    print("IoU  |    AP    |    AR")
    for thr, metrics in results.items():
        print(f"{thr:.2f} | {metrics['AP']:.4f}  | {metrics['AR']:.4f}")


if __name__ == "__main__":
    main()
