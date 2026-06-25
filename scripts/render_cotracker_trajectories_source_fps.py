from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from render_birds_eye_locations import draw_floor, rgb_to_bgr, world_to_canvas


def parse_fps(value: str) -> float:
    if "/" in value:
        return float(Fraction(value))
    return float(value)


def probe_video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0 or not np.isfinite(fps):
        raise RuntimeError(f"Could not read FPS from source video: {path}")
    return fps


def open_h264_writer(path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "21"],
    )


def append_bgr(writer, frame_bgr: np.ndarray) -> None:
    writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def load_tracks(path: Path, input_fps: float) -> dict[int, list[dict]]:
    data = json.loads(path.read_text())
    tracks: dict[int, list[dict]] = defaultdict(list)
    for frame in data.get("frames", []):
        frame_idx = int(frame["frame"])
        t = frame_idx / input_fps
        for player in frame.get("players", []):
            object_id = int(player["object_id"])
            tracks[object_id].append(
                {
                    "t": float(t),
                    "frame": frame_idx,
                    "floor_xy_ft": [float(v) for v in player["floor_xy_ft"]],
                    "camera_xy": [float(v) for v in player["camera_xy"]],
                    "color": [int(v) for v in player.get("color", [230, 230, 230])],
                }
            )
    return {object_id: sorted(items, key=lambda item: item["t"]) for object_id, items in tracks.items()}


def interpolate_track(items: list[dict], t: float, max_gap_s: float) -> dict | None:
    if not items:
        return None
    if t < items[0]["t"] or t > items[-1]["t"]:
        return None
    lo = 0
    hi = len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid]["t"] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    right_idx = lo
    left_idx = lo - 1
    if right_idx < len(items) and abs(items[right_idx]["t"] - t) < 1e-9:
        return items[right_idx]
    if left_idx < 0 or right_idx >= len(items):
        return None
    left = items[left_idx]
    right = items[right_idx]
    gap = right["t"] - left["t"]
    if gap <= 0 or gap > max_gap_s:
        return None
    alpha = (t - left["t"]) / gap
    floor_xy = (1.0 - alpha) * np.asarray(left["floor_xy_ft"], dtype=np.float64) + alpha * np.asarray(right["floor_xy_ft"], dtype=np.float64)
    camera_xy = (1.0 - alpha) * np.asarray(left["camera_xy"], dtype=np.float64) + alpha * np.asarray(right["camera_xy"], dtype=np.float64)
    return {
        "t": float(t),
        "floor_xy_ft": floor_xy.tolist(),
        "camera_xy": camera_xy.tolist(),
        "color": left["color"],
    }


def active_players(tracks: dict[int, list[dict]], t: float, max_gap_s: float) -> list[tuple[int, dict]]:
    active = []
    for object_id, items in tracks.items():
        item = interpolate_track(items, t, max_gap_s)
        if item is not None:
            active.append((object_id, item))
    return active


def render_birds_eye(
    tracks: dict[int, list[dict]],
    output_video: Path,
    fps: float,
    duration_s: float,
    max_gap_s: float,
    width: int,
    height: int,
    margin: int,
    trail_s: float,
) -> int:
    writer = open_h264_writer(output_video, fps)
    trail_len = max(1, int(round(trail_s * fps)))
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
    frame_count = int(round(duration_s * fps))
    try:
        for frame_idx in range(frame_count):
            t = frame_idx / fps
            canvas = draw_floor(width, height, margin)
            cv2.putText(canvas, f"VGGT + CoTracker trajectories {t:06.2f}s", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            for object_id, item in active_players(tracks, t, max_gap_s):
                color = rgb_to_bgr(item["color"])
                x, y = item["floor_xy_ft"]
                px, py = world_to_canvas(float(x), float(y), width, height, margin)
                history[object_id].append((px, py))
                pts = list(history[object_id])
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, str(object_id), (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            append_bgr(writer, canvas)
    finally:
        writer.close()
    return frame_count


def render_camera_overlay(
    tracks: dict[int, list[dict]],
    source_video: Path,
    output_video: Path,
    fps: float,
    start_s: float,
    duration_s: float,
    max_gap_s: float,
    trail_s: float,
) -> int:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    sequential_read = abs(source_fps - fps) / max(fps, 1e-9) < 0.002
    writer = open_h264_writer(output_video, fps)
    trail_len = max(1, int(round(trail_s * fps)))
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
    frame_count = int(round(duration_s * fps))
    written = 0
    try:
        for frame_idx in range(frame_count):
            if not sequential_read:
                expected_source_frame = int(round((start_s + frame_idx / fps) * source_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, expected_source_frame)
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            cv2.putText(frame, f"CoTracker SAM3 tracks {t:06.2f}s", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            for object_id, item in active_players(tracks, t, max_gap_s):
                color = rgb_to_bgr(item["color"])
                x, y = item["camera_xy"]
                px, py = int(round(x)), int(round(y))
                if not (0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]):
                    continue
                history[object_id].append((px, py))
                pts = list(history[object_id])
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, p0, p1, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(object_id), (px + 7, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            append_bgr(writer, frame)
            written += 1
    finally:
        writer.close()
        cap.release()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render CoTracker/VGGT player trajectories at the source video's frame rate by interpolating trajectory JSON samples.")
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-fps", type=float, default=1.0)
    parser.add_argument("--source-video")
    parser.add_argument("--source-start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--output-fps", default="")
    parser.add_argument("--max-interp-gap-sec", type=float, default=2.25)
    parser.add_argument("--trail-sec", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=54)
    parser.add_argument("--skip-camera", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(Path(args.trajectory_json), args.input_fps)
    if not tracks:
        raise RuntimeError("No player tracks found in trajectory JSON.")
    inferred_duration = max(item["t"] for items in tracks.values() for item in items) + (1.0 / args.input_fps)
    duration_s = args.duration_sec if args.duration_sec > 0 else inferred_duration
    if args.output_fps:
        output_fps = parse_fps(args.output_fps)
    elif args.source_video:
        output_fps = probe_video_fps(Path(args.source_video))
    else:
        output_fps = args.input_fps

    birds_eye_video = output_dir / "cotracker_vggt_player_trajectories_birds_eye_source_fps_h264.mp4"
    birds_frames = render_birds_eye(
        tracks,
        birds_eye_video,
        output_fps,
        duration_s,
        args.max_interp_gap_sec,
        args.width,
        args.height,
        args.margin,
        args.trail_sec,
    )
    result = {
        "schema": "cotracker_vggt_source_fps_render_v1",
        "trajectory_json": args.trajectory_json,
        "input_fps": args.input_fps,
        "output_fps": output_fps,
        "duration_sec": duration_s,
        "max_interp_gap_sec": args.max_interp_gap_sec,
        "birds_eye_video": str(birds_eye_video),
        "birds_eye_frames": birds_frames,
        "objects": len(tracks),
    }
    if args.source_video and not args.skip_camera:
        camera_video = output_dir / "cotracker_sam3_player_tracks_camera_source_fps_h264.mp4"
        camera_frames = render_camera_overlay(
            tracks,
            Path(args.source_video),
            camera_video,
            output_fps,
            args.source_start_sec,
            duration_s,
            args.max_interp_gap_sec,
            args.trail_sec,
        )
        result["camera_video"] = str(camera_video)
        result["camera_frames"] = camera_frames
    metadata = output_dir / "source_fps_render_metadata.json"
    metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
