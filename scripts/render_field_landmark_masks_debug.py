from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_birds_eye_locations import FEATURE_COLORS_BGR
from video_utils import write_h264_video


FEATURE_OBJECTS = {
    "left_restraining_line": 1,
    "right_restraining_line": 2,
    "midfield_line": 3,
    "goal_crease": 4,
    "field_outline": 5,
}


def load_masks(mask_dir: Path, frame_idx: int, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    height, width = shape
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    object_ids = [int(v) for v in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    output = {}
    for feature, object_id in FEATURE_OBJECTS.items():
        if object_id not in object_ids:
            continue
        mask = masks[object_ids.index(object_id)]
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        output[feature] = mask
    return output


def render_frame(frame_bgr: np.ndarray, masks: dict[str, np.ndarray], mask_only: bool, alpha: float) -> np.ndarray:
    if mask_only:
        rendered = np.zeros_like(frame_bgr)
    else:
        rendered = frame_bgr.copy()
    overlay = rendered.astype(np.float32)
    for feature, mask in masks.items():
        if not mask.any():
            continue
        color = np.asarray(FEATURE_COLORS_BGR.get(feature, (255, 255, 255)), dtype=np.float32)
        if mask_only:
            overlay[mask] = color
        else:
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
        ys, xs = np.where(mask)
        if len(xs):
            x, y = int(np.median(xs)), int(np.median(ys))
            cv2.putText(overlay, feature, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color.tolist(), 1, cv2.LINE_AA)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--mask-dir", default="outputs/sam2/floor_features/sam2_floor_feature_instance_masks_with_outline")
    parser.add_argument("--output-video", default="outputs/vggt/debug_full/field_landmark_masks_overlay_h264.mp4")
    parser.add_argument("--output-mask-only-video", default="outputs/vggt/debug_full/field_landmark_masks_only_h264.mp4")
    parser.add_argument("--output-json", default="outputs/vggt/debug_full/field_landmark_masks_summary.json")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    overlay_frames = []
    mask_only_frames = []
    summary = []
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        masks = load_masks(Path(args.mask_dir), frame_idx, frame.shape[:2])
        overlay_frames.append(cv2.cvtColor(render_frame(frame, masks, mask_only=False, alpha=args.alpha), cv2.COLOR_BGR2RGB))
        mask_only_frames.append(cv2.cvtColor(render_frame(frame, masks, mask_only=True, alpha=args.alpha), cv2.COLOR_BGR2RGB))
        summary.append({"frame": frame_idx, "features": {feature: int(mask.sum()) for feature, mask in masks.items()}})
    write_h264_video(overlay_frames, Path(args.output_video), args.fps)
    write_h264_video(mask_only_frames, Path(args.output_mask_only_video), args.fps)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"frames": summary}, indent=2) + "\n")
    print(json.dumps({"output_video": args.output_video, "output_mask_only_video": args.output_mask_only_video, "frames": len(overlay_frames)}, indent=2))


if __name__ == "__main__":
    main()
