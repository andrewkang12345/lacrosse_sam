from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_vggt_birds_eye import run_vggt_predictions, sorted_frame_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT on a frame directory and save a compact reconstruction file.")
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--model", default="facebook/VGGT-1B")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths_all = sorted_frame_paths(Path(args.frames_dir))
    selected = list(range(0, len(frame_paths_all), max(1, args.frame_stride)))
    if args.max_frames:
        selected = selected[: args.max_frames]
    frame_paths = [frame_paths_all[idx] for idx in selected]
    if not frame_paths:
        raise SystemExit(f"No frames found in {args.frames_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extrinsic, intrinsic, depth_map, depth_conf, _ = run_vggt_predictions(args, frame_paths)
    np.savez_compressed(
        output_dir / "vggt_predictions_compact.npz",
        frame_indices=np.asarray(selected, dtype=np.int32),
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_conf=depth_conf,
    )
    metadata = {
        "schema": "vggt_reconstruction_clip_v1",
        "frames_dir": args.frames_dir,
        "frame_count": len(frame_paths),
        "frame_indices": selected,
        "model": args.model,
        "resolution": args.resolution,
        "output_npz": str(output_dir / "vggt_predictions_compact.npz"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
