from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

from video_utils import load_pil_frames, overlay_masks, write_h264_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--output", default="outputs/sam2_click_tracks.mp4")
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--instance-mask-dir", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def load_prompts(path: Path) -> dict[tuple[int, int], dict[str, list]]:
    raw = json.loads(path.read_text())
    records = raw["prompts"] if isinstance(raw, dict) and "prompts" in raw else raw
    grouped: dict[tuple[int, int], dict[str, list]] = defaultdict(lambda: {"points": [], "labels": []})
    for record in records:
        frame_idx = int(record.get("frame", 0))
        obj_id = int(record["object_id"])
        points = record.get("points")
        if points is None:
            points = [[record["x"], record["y"]]]
        labels = record.get("labels")
        if labels is None:
            labels = [int(record.get("label", 1))] * len(points)
        for point, label in zip(points, labels):
            grouped[(frame_idx, obj_id)]["points"].append([float(point[0]), float(point[1])])
            grouped[(frame_idx, obj_id)]["labels"].append(int(label))
    return grouped


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_pil_frames(Path(args.frames_dir))
    if not frames:
        raise RuntimeError(f"No frames found in {args.frames_dir}")

    prompt_groups = load_prompts(Path(args.prompts))
    if not prompt_groups:
        raise RuntimeError("No click prompts were provided")

    model = Sam2VideoModel.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(args.model)
    inference_session = processor.init_video_session(video=frames, inference_device=device)

    prompts_by_frame: dict[int, list[tuple[int, list, list]]] = defaultdict(list)
    for (frame_idx, obj_id), payload in sorted(prompt_groups.items()):
        prompts_by_frame[frame_idx].append((obj_id, payload["points"], payload["labels"]))

    for frame_idx, entries in sorted(prompts_by_frame.items()):
        obj_ids = [obj_id for obj_id, _, _ in entries]
        input_points = [[points for _, points, _ in entries]]
        input_labels = [[labels for _, _, labels in entries]]
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            input_points=input_points,
            input_labels=input_labels,
        )

        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            model(inference_session=inference_session, frame_idx=frame_idx)

    rendered_frames = []
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for path in mask_dir.glob("*.png"):
            path.unlink()
    instance_mask_dir = Path(args.instance_mask_dir) if args.instance_mask_dir else None
    if instance_mask_dir:
        instance_mask_dir.mkdir(parents=True, exist_ok=True)
        for path in instance_mask_dir.glob("*.npz"):
            path.unlink()
    metadata = {
        "model": args.model,
        "fps": args.fps,
        "frames": [],
        "object_ids": [],
    }

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for output in model.propagate_in_video_iterator(inference_session):
            video_res_masks = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=False,
            )[0]
            masks_by_id = {
                int(obj_id): video_res_masks[i]
                for i, obj_id in enumerate(inference_session.obj_ids)
                if i < video_res_masks.shape[0]
            }
            obj_ids = [int(obj_id) for obj_id in masks_by_id]
            mask_arrays = [
                np.squeeze(masks_by_id[obj_id].detach().float().cpu().numpy())
                for obj_id in obj_ids
            ]
            if mask_dir:
                from PIL import Image

                if mask_arrays:
                    logits = np.stack(mask_arrays, axis=0)
                    best = logits.argmax(axis=0)
                    score = logits.max(axis=0)
                    label_mask = np.zeros(score.shape, dtype=np.uint8)
                    for i, obj_id in enumerate(obj_ids):
                        label_mask[(best == i) & (score > 0)] = min(int(obj_id), 255)
                else:
                    label_mask = np.zeros((inference_session.video_height, inference_session.video_width), dtype=np.uint8)
                Image.fromarray(label_mask, mode="P").save(mask_dir / f"{output.frame_idx:08d}.png")
            if instance_mask_dir:
                if mask_arrays:
                    stack = np.stack(mask_arrays, axis=0) > 0
                else:
                    stack = np.zeros((0, inference_session.video_height, inference_session.video_width), dtype=bool)
                np.savez_compressed(
                    instance_mask_dir / f"{output.frame_idx:08d}.npz",
                    object_ids=np.asarray(obj_ids, dtype=np.int32),
                    masks=stack.astype(np.uint8),
                )
            rendered_frames.append(overlay_masks(frames[output.frame_idx], masks_by_id))
            metadata["frames"].append(
                {
                    "frame": int(output.frame_idx),
                    "object_ids": [int(obj_id) for obj_id in masks_by_id],
                }
            )

    metadata["object_ids"] = [int(obj_id) for obj_id in inference_session.obj_ids]
    write_h264_video(rendered_frames, Path(args.output), fps=args.fps)
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": args.output, "metadata": str(meta_path), "objects": metadata["object_ids"]}, indent=2))


if __name__ == "__main__":
    main()
