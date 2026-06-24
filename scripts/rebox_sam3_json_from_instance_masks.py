from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def mask_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute SAM3 boxes from remapped per-instance masks.")
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--instance-mask-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=16.0)
    args = parser.parse_args()

    source = json.loads(Path(args.source_json).read_text())
    by_frame = {int(frame["frame"]): frame for frame in source.get("frames", [])}
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    output_frames = []
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        record = by_frame.get(frame_idx, {"frame": frame_idx, "object_ids": [], "scores": []})
        score_by_id = {
            int(obj_id): float(record.get("scores", [1.0] * len(record.get("object_ids", [])))[idx])
            for idx, obj_id in enumerate(record.get("object_ids", []))
        }
        path = Path(args.instance_mask_dir) / f"{frame_idx:08d}.npz"
        object_ids: list[int] = []
        boxes: list[list[float]] = []
        scores: list[float] = []
        if path.exists():
            data = np.load(path)
            ids = data["object_ids"].astype(int).tolist()
            masks = data["masks"].astype(bool)
            for obj_id, mask in zip(ids, masks):
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                box = mask_box(mask)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if x2 - x1 < args.min_width or y2 - y1 < args.min_height:
                    continue
                object_ids.append(int(obj_id))
                boxes.append(box)
                scores.append(score_by_id.get(int(obj_id), 1.0))
        out = dict(record)
        out["frame"] = int(frame_idx)
        out["object_ids"] = object_ids
        out["boxes"] = boxes
        out["scores"] = scores
        output_frames.append(out)

    source["frames"] = output_frames
    source["source_json_before_rebox"] = args.source_json
    source["instance_mask_dir_for_rebox"] = args.instance_mask_dir
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps({"output_json": str(output_path), "frames": len(output_frames), "detections": sum(len(f["object_ids"]) for f in output_frames)}, indent=2))


if __name__ == "__main__":
    main()
