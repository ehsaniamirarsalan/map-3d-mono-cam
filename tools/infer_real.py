"""Run Map-Det3D on real images/video using the real pretrained MapAnything
backbone. Detector weights are randomly initialized until you train on real
data (see README's "Next steps toward the real paper") -- this script
proves the real backbone <-> our detector wiring works end-to-end; it will
NOT produce meaningful boxes until trained.

Usage:
    python tools/infer_real.py --image-folder path/to/frames --window 5
"""

from __future__ import annotations

import argparse

try:
    # Workaround for environments where Python's bundled certifi CA bundle
    # doesn't match the local network's TLS setup (e.g. corporate proxies):
    # patches ssl to trust the OS certificate store instead. Harmless no-op
    # if the package isn't installed or isn't needed.
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

import torch

from mapdet3d.backbone.mapanything_wrapper import MapAnythingBackbone
from mapdet3d.inference.sliding_window import SlidingWindowInference
from mapdet3d.models.mapdet3d import MapDet3D, MapDet3DCore


def build_real_model(num_queries: int, num_decoder_layers: int, pretrained: str) -> MapDet3D:
    backbone = MapAnythingBackbone(pretrained=pretrained, freeze_encoder=True)

    # Real MapAnything channel dims: 1536 (ViT-giant encoder) for F_E,
    # and the alternating-attention transformer's own dim (also 1536 for
    # the giant config) for F_7/F_11/F_15. Confirm against your loaded
    # model if using a different config: backbone.model.encoder.enc_embed_dim
    # and backbone.model.info_sharing.dim.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument(
        "--pretrained", type=str, default="facebook/map-anything-apache",
        help="HF checkpoint id; use 'facebook/map-anything' for the CC-BY-NC research weights",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="optional trained mapdet3d checkpoint")
    args = parser.parse_args()

    from mapanything.utils.image import load_images

    model = build_real_model(args.num_queries, args.num_decoder_layers, args.pretrained)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.core.load_state_dict(state["model_state"])

    views = load_images(args.image_folder)
    print(f"Loaded {len(views)} frames from {args.image_folder}")

    sw = SlidingWindowInference(model, window_size=args.window)
    for t, view in enumerate(views):
        pred = sw.step(view)
        scores = torch.sigmoid(pred["logits"]).squeeze(-1)
        keep = scores[0] > 0.5
        print(f"frame {t}: {keep.sum().item()} boxes above 0.5 confidence "
              f"(out of {pred['logits'].shape[1]} queries)")


if __name__ == "__main__":
    main()
