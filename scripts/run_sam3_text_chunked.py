from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from video_utils import load_pil_frames, overlay_masks, write_h264_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--instance-mask-dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=45)
    parser.add_argument("--object-id-stride", type=int, default=10000)
    parser.add_argument("--work-dir", default=None)
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_chunk_frames(frame_paths: list[Path], chunk_dir: Path) -> None:
    reset_dir(chunk_dir)
    for idx, source in enumerate(frame_paths):
        target = chunk_dir / f"frame_{idx:05d}.jpg"
        target.symlink_to(source.resolve())


def offset_instance_npz(source: Path, target: Path, object_offset: int) -> None:
    data = np.load(source)
    object_ids = data["object_ids"].astype(np.int32) + object_offset
    masks = data["masks"].astype(np.uint8)
    np.savez_compressed(target, object_ids=object_ids, masks=masks)


def write_label_mask_from_npz(npz_path: Path, png_path: Path) -> None:
    data = np.load(npz_path)
    object_ids = data["object_ids"].astype(np.int32)
    masks = data["masks"].astype(bool)
    if masks.size == 0:
        if masks.ndim == 3:
            label = np.zeros(masks.shape[1:], dtype=np.uint8)
        else:
            label = np.zeros((1, 1), dtype=np.uint8)
    else:
        stack = masks.astype(np.uint8)
        best = stack.argmax(axis=0)
        score = stack.max(axis=0)
        label = np.zeros(score.shape, dtype=np.uint8)
        for idx, obj_id in enumerate(object_ids):
            label[(best == idx) & (score > 0)] = min(int(obj_id), 255)
    Image.fromarray(label, mode="P").save(png_path)


def run_chunk(args: argparse.Namespace, chunk_dir: Path, chunk_output: Path, chunk_mask_dir: Path, chunk_instance_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("track_sam3_text.py")),
        "--frames-dir",
        str(chunk_dir),
        "--text",
        args.text,
        "--model",
        args.model,
        "--output",
        str(chunk_output),
        "--mask-dir",
        str(chunk_mask_dir),
        "--instance-mask-dir",
        str(chunk_instance_dir),
        "--fps",
        str(args.fps),
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return json.loads(chunk_output.with_suffix(".json").read_text())


def merged_overlay_video(frames_dir: Path, instance_mask_dir: Path, output: Path, fps: float) -> None:
    frames = load_pil_frames(frames_dir)
    rendered = []
    for frame_idx, frame in enumerate(frames):
        path = instance_mask_dir / f"{frame_idx:08d}.npz"
        masks_by_id = {}
        if path.exists():
            data = np.load(path)
            for obj_id, mask in zip(data["object_ids"].astype(int).tolist(), data["masks"].astype(bool)):
                masks_by_id[int(obj_id)] = mask
        rendered.append(overlay_masks(frame, masks_by_id))
    write_h264_video(rendered, output, fps=fps)


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frame_*.jpg files found in {frames_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir)
    instance_mask_dir = Path(args.instance_mask_dir)
    reset_dir(mask_dir)
    reset_dir(instance_mask_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output.parent / f"{output.stem}_chunks"
    reset_dir(work_dir)

    merged = {
        "model": args.model,
        "text": args.text,
        "fps": args.fps,
        "chunk_size": args.chunk_size,
        "frames": [],
    }

    for chunk_idx, start in enumerate(range(0, len(frame_paths), args.chunk_size)):
        end = min(start + args.chunk_size, len(frame_paths))
        object_offset = chunk_idx * args.object_id_stride
        chunk_dir = work_dir / f"frames_{chunk_idx:03d}"
        chunk_output = work_dir / f"track_{chunk_idx:03d}.mp4"
        chunk_mask_dir = work_dir / f"label_masks_{chunk_idx:03d}"
        chunk_instance_dir = work_dir / f"instance_masks_{chunk_idx:03d}"
        link_chunk_frames(frame_paths[start:end], chunk_dir)
        chunk_meta = run_chunk(args, chunk_dir, chunk_output, chunk_mask_dir, chunk_instance_dir)

        for record in chunk_meta.get("frames", []):
            local_frame = int(record["frame"])
            global_frame = start + local_frame
            record["frame"] = global_frame
            record["object_ids"] = [int(obj_id) + object_offset for obj_id in record.get("object_ids", [])]
            merged["frames"].append(record)
            source_npz = chunk_instance_dir / f"{local_frame:08d}.npz"
            target_npz = instance_mask_dir / f"{global_frame:08d}.npz"
            if source_npz.exists():
                offset_instance_npz(source_npz, target_npz, object_offset)
                write_label_mask_from_npz(target_npz, mask_dir / f"{global_frame:08d}.png")

    merged["frames"].sort(key=lambda item: int(item["frame"]))
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(merged, indent=2) + "\n")
    merged_overlay_video(frames_dir, instance_mask_dir, output, args.fps)
    all_ids = sorted({int(obj_id) for frame in merged["frames"] for obj_id in frame.get("object_ids", [])})
    print(json.dumps({"output": str(output), "metadata": str(meta_path), "frames": len(merged["frames"]), "objects": all_ids}, indent=2))


if __name__ == "__main__":
    main()
