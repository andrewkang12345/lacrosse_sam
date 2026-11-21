"""Run VGGT-Omega on a frame directory and save cameras, depth, confidence and colors.

The saved npz is the input for `fit_field_plane_landmarks.py` and
`render_vggt_scene_3d.py`. Cameras follow the VGGT-Omega convention:
`extrinsic` is camera-from-world (OpenCV axes) and `intrinsic` is expressed in
model-image pixels (`model_hw`). Use `original_hw` to map back to frame pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output .npz path.")
    parser.add_argument("--omega-repo", type=Path, default=Path("third_party/vggt_omega"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt_omega/vggt_omega_1b_512.pt"))
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=0, help="0 keeps every frame.")
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.omega_repo.resolve()))
    from PIL import Image
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    frame_paths = sorted(args.frames_dir.glob("frame_*.jpg"))[:: max(1, args.frame_stride)]
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.jpg files in {args.frames_dir}")
    with Image.open(frame_paths[0]) as first:
        original_w, original_h = first.size

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    images = load_and_preprocess_images([str(p) for p in frame_paths], image_resolution=args.image_resolution).to("cuda")
    start = time.time()
    with torch.inference_mode():
        predictions = model(images)
        extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    torch.cuda.synchronize()
    elapsed = time.time() - start

    model_h, model_w = (int(v) for v in predictions["images"].shape[-2:])
    depth = predictions["depth"][0].float().cpu().numpy()
    if depth.ndim == 4:
        depth = depth[..., 0]
    colors = (predictions["images"][0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frame_names=np.asarray([p.name for p in frame_paths]),
        extrinsic=extrinsic[0].float().cpu().numpy(),
        intrinsic=intrinsic[0].float().cpu().numpy(),
        depth=depth.astype(np.float32),
        depth_conf=predictions["depth_conf"][0].float().cpu().numpy().astype(np.float32),
        colors=colors,
        model_hw=np.asarray([model_h, model_w], dtype=np.int32),
        original_hw=np.asarray([original_h, original_w], dtype=np.int32),
    )
    summary = {
        "output": str(args.output),
        "frames": len(frame_paths),
        "model_hw": [model_h, model_w],
        "original_hw": [original_h, original_w],
        "inference_seconds": round(elapsed, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
