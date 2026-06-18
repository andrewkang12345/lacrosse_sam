from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from video_utils import load_pil_frames, overlay_masks, write_h264_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--text", default="lacrosse player")
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--output", default="outputs/sam3_text_tracks.mp4")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_pil_frames(Path(args.frames_dir))
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError(f"No frames found in {args.frames_dir}")

    model = Sam3VideoModel.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16,
        token=token,
    )
    processor = Sam3VideoProcessor.from_pretrained(args.model, token=token)

    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=args.text,
    )

    rendered_frames: list = []
    metadata = {
        "model": args.model,
        "text": args.text,
        "fps": args.fps,
        "frames": [],
    }

    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for output in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(frames) - 1,
            show_progress_bar=True,
        ):
            processed = processor.postprocess_outputs(inference_session, output)
            object_ids = processed["object_ids"].detach().cpu().tolist()
            masks = processed["masks"]
            scores = processed.get("scores")
            boxes = processed.get("boxes")
            masks_by_id = {int(obj_id): masks[i] for i, obj_id in enumerate(object_ids)}
            rendered_frames.append(overlay_masks(frames[output.frame_idx], masks_by_id))
            metadata["frames"].append(
                {
                    "frame": int(output.frame_idx),
                    "object_ids": [int(obj_id) for obj_id in object_ids],
                    "scores": scores.detach().float().cpu().tolist() if scores is not None else [],
                    "boxes": boxes.detach().float().cpu().tolist() if boxes is not None else [],
                }
            )

    write_h264_video(rendered_frames, Path(args.output), fps=args.fps)
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    all_ids = sorted({obj_id for frame in metadata["frames"] for obj_id in frame["object_ids"]})
    print(json.dumps({"output": args.output, "metadata": str(meta_path), "objects": all_ids}, indent=2))


if __name__ == "__main__":
    main()
