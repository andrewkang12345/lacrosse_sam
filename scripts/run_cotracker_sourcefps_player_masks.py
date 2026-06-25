from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from render_birds_eye_locations import rgb_to_bgr, write_h264
from run_cotracker_vggt_player_trajectories import (
    build_queries,
    color_for_object,
    load_frames_bgr,
    load_sam3_metadata,
    mask_for_object,
    object_presence,
    run_cotracker,
)


def aggregate_camera_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    query_metadata: list[dict],
    frame_count: int,
    frame_shape: tuple[int, int],
    mask_dir: Path,
    require_sam3_presence: bool,
    ignore_cotracker_visibility: bool,
    mask_gate: bool,
    mask_dilation: int,
    sam3_presence: dict[int, set[int]],
) -> list[dict]:
    height, width = frame_shape
    query_indices_by_object: dict[int, list[int]] = defaultdict(list)
    query_frame_by_index = {}
    for idx, item in enumerate(query_metadata):
        query_indices_by_object[int(item["object_id"])].append(idx)
        query_frame_by_index[idx] = int(item["query_frame"])

    kernel = np.ones((max(1, mask_dilation), max(1, mask_dilation)), np.uint8)
    frames = []
    for frame_idx in range(frame_count):
        players = []
        for object_id, point_indices in sorted(query_indices_by_object.items()):
            if require_sam3_presence and frame_idx not in sam3_presence.get(object_id, set()):
                continue
            points = []
            for point_idx in point_indices:
                if frame_idx < query_frame_by_index[point_idx]:
                    continue
                if not ignore_cotracker_visibility and not visibility[frame_idx, point_idx]:
                    continue
                x, y = tracks[frame_idx, point_idx]
                if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
                    points.append([float(x), float(y)])
            if not points:
                continue
            points_xy = np.asarray(points, dtype=np.float32)
            gated_xy = points_xy
            mask_available = False
            if mask_gate:
                mask = mask_for_object(mask_dir, frame_idx, object_id, frame_shape)
                if mask.any():
                    mask_available = True
                    if mask_dilation > 1:
                        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                    xs = np.clip(np.round(points_xy[:, 0]).astype(int), 0, width - 1)
                    ys = np.clip(np.round(points_xy[:, 1]).astype(int), 0, height - 1)
                    gated_xy = points_xy[mask[ys, xs]]
            if len(gated_xy) == 0:
                continue
            camera_xy = np.median(gated_xy, axis=0)
            players.append(
                {
                    "object_id": int(object_id),
                    "camera_xy": [float(camera_xy[0]), float(camera_xy[1])],
                    "visible_track_points": int(len(points_xy)),
                    "mask_gated_track_points": int(len(gated_xy)),
                    "mask_available": bool(mask_available),
                    "cotracker_visibility_used": not bool(ignore_cotracker_visibility),
                    "color": color_for_object(object_id),
                }
            )
        frames.append({"frame": int(frame_idx), "players": players})
    return frames


def render_camera_overlay(frames_bgr: list[np.ndarray], frames_out: list[dict], output_video: Path, fps: float, trail_frames: int) -> None:
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_frames))
    rendered = []
    for frame, item in zip(frames_bgr, frames_out):
        canvas = frame.copy()
        cv2.putText(canvas, f"Source-fps CoTracker SAM3 tracks frame {item['frame']}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        for player in item["players"]:
            object_id = int(player["object_id"])
            color = rgb_to_bgr(player["color"])
            x, y = player["camera_xy"]
            px, py = int(round(x)), int(round(y))
            history[object_id].append((px, py))
            pts = list(history[object_id])
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(object_id), (px + 7, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        rendered.append(canvas)
    write_h264(rendered, output_video, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoTracker on SAM3 player masks at the same frame rate as the input frame directory.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sam3-json", required=True)
    parser.add_argument("--player-mask-dir", required=True)
    parser.add_argument("--cotracker-repo", default="third_party/co-tracker")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=59.94005994)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--points-per-object", type=int, default=3)
    parser.add_argument("--min-object-frames", type=int, default=8)
    parser.add_argument("--min-mask-area", type=int, default=24)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--vertical-min-frac", type=float, default=0.15)
    parser.add_argument("--vertical-max-frac", type=float, default=0.95)
    parser.add_argument("--require-sam3-presence", action="store_true")
    parser.add_argument("--ignore-cotracker-visibility", action="store_true")
    parser.add_argument("--disable-mask-gate", action="store_true")
    parser.add_argument("--mask-dilation", type=int, default=13)
    parser.add_argument("--trail-frames", type=int, default=180)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_bgr = load_frames_bgr(Path(args.frames_dir), args.max_frames)
    sam3 = load_sam3_metadata(Path(args.sam3_json), len(frames_bgr))
    queries, query_metadata = build_queries(
        sam3,
        Path(args.player_mask_dir),
        frames_bgr[0].shape[:2],
        args.min_object_frames,
        args.points_per_object,
        args.min_mask_area,
        args.max_objects,
        args.vertical_min_frac,
        args.vertical_max_frac,
    )
    import torch

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(json.dumps({"frames": len(frames_bgr), "query_points": int(len(queries)), "objects": len({m["object_id"] for m in query_metadata}), "device": device}, indent=2))
    tracks, visibility = run_cotracker(frames_bgr, queries, Path(args.cotracker_repo), device)
    frames_out = aggregate_camera_tracks(
        tracks,
        visibility,
        query_metadata,
        len(frames_bgr),
        frames_bgr[0].shape[:2],
        Path(args.player_mask_dir),
        args.require_sam3_presence,
        args.ignore_cotracker_visibility,
        not args.disable_mask_gate,
        args.mask_dilation,
        object_presence(sam3),
    )
    output_json = output_dir / "cotracker_sourcefps_player_tracks.json"
    tracks_npz = output_dir / "cotracker_sourcefps_tracks_raw.npz"
    output_video = output_dir / "cotracker_sourcefps_player_tracks_camera_h264.mp4"
    render_camera_overlay(frames_bgr, frames_out, output_video, args.fps, args.trail_frames)
    output_json.write_text(
        json.dumps(
            {
                "schema": "cotracker_sourcefps_player_tracks_v1",
                "frames_dir": args.frames_dir,
                "sam3_json": args.sam3_json,
                "player_mask_dir": args.player_mask_dir,
                "fps": args.fps,
                "query_metadata": query_metadata,
                "frames": frames_out,
                "trajectory_lengths": {
                    str(object_id): sum(1 for frame in frames_out for player in frame["players"] if int(player["object_id"]) == object_id)
                    for object_id in sorted({int(item["object_id"]) for item in query_metadata})
                },
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        tracks_npz,
        queries=queries,
        tracks=tracks.astype(np.float32),
        visibility=visibility.astype(np.uint8),
        object_ids=np.asarray([item["object_id"] for item in query_metadata], dtype=np.int32),
        query_frames=np.asarray([item["query_frame"] for item in query_metadata], dtype=np.int32),
    )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "tracks_npz": str(tracks_npz),
                "camera_video": str(output_video),
                "frames": len(frames_out),
                "objects": len({m["object_id"] for m in query_metadata}),
                "trajectory_points": sum(len(frame["players"]) for frame in frames_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
