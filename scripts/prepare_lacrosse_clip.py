from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image

from video_utils import draw_coordinate_grid, write_h264_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/lacrosse_long_source_iniyaa.mp4")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--prompt-dir", default="outputs/prompt_frames")
    parser.add_argument("--preview-video", default="outputs/lacrosse_10s_10fps_preview.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    frames_dir = Path(args.frames_dir)
    prompt_dir = Path(args.prompt_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for path in frames_dir.glob("frame_*.jpg"):
        path.unlink()
    for path in prompt_dir.glob("prompt_frame_*.jpg"):
        path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(round(args.seconds * args.fps))
    rgb_frames = []
    sample_records = []

    for out_idx in range(total_frames):
        time_sec = out_idx / args.fps
        src_frame = int(round(time_sec * native_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        frame_path = frames_dir / f"frame_{out_idx:05d}.jpg"
        image.save(frame_path, quality=95)
        rgb_frames.append(rgb)
        sample_records.append(
            {"frame": out_idx, "time_sec": round(time_sec, 3), "source_frame": src_frame}
        )

    cap.release()
    if not rgb_frames:
        raise RuntimeError("No frames were extracted")

    write_h264_video(rgb_frames, Path(args.preview_video), fps=args.fps)

    prompt_indices = sorted({0, len(rgb_frames) // 4, len(rgb_frames) // 2, 3 * len(rgb_frames) // 4, len(rgb_frames) - 1})
    for idx in prompt_indices:
        frame = Image.fromarray(rgb_frames[idx])
        title = f"frame {idx} | t={idx / args.fps:.1f}s | click coordinates are x,y"
        draw_coordinate_grid(frame, title).save(prompt_dir / f"prompt_frame_{idx:05d}.jpg", quality=95)

    manifest = {
        "source_video": str(video_path),
        "frames_dir": str(frames_dir),
        "fps": args.fps,
        "seconds": args.seconds,
        "width": int(rgb_frames[0].shape[1]),
        "height": int(rgb_frames[0].shape[0]),
        "frames": sample_records,
        "prompt_frames": [str(prompt_dir / f"prompt_frame_{idx:05d}.jpg") for idx in prompt_indices],
        "preview_video": args.preview_video,
    }
    manifest_path = prompt_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
