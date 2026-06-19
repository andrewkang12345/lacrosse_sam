from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64)


def write_h264_rgb(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="error",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def load_team_metadata(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    return {int(frame["frame"]): frame for frame in data.get("frames", [])}


def frame_color(frame_meta: dict, obj_id: int) -> list[int]:
    ids = [int(v) for v in frame_meta.get("object_ids", [])]
    if obj_id in ids:
        idx = ids.index(obj_id)
        colors = frame_meta.get("team_colors", [])
        if idx < len(colors):
            return [int(v) for v in colors[idx]]
    return [220, 220, 220]


def collect_bounds(mesh_root: Path, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for obj_dir in sorted(mesh_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not obj_dir.is_dir():
            continue
        for path in sorted(obj_dir.glob("*.obj")):
            frame_idx = int(path.stem)
            if max_frames and frame_idx >= max_frames:
                continue
            vertices = read_obj_vertices(path)
            if len(vertices) == 0:
                continue
            xz = vertices[:, [0, 2]]
            mins.append(np.percentile(xz, 1, axis=0))
            maxs.append(np.percentile(xz, 99, axis=0))
    if not mins:
        raise RuntimeError(f"No OBJ vertices found under {mesh_root}")
    mn = np.min(np.stack(mins), axis=0)
    mx = np.max(np.stack(maxs), axis=0)
    pad = np.maximum((mx - mn) * 0.08, 1e-3)
    return mn - pad, mx + pad


def to_canvas(points_xz: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], width: int, height: int, margin: int) -> np.ndarray:
    mn, mx = bounds
    span = np.maximum(mx - mn, 1e-6)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    ox = (width - span[0] * scale) * 0.5
    oy = (height - span[1] * scale) * 0.5
    out = np.zeros_like(points_xz, dtype=np.int32)
    out[:, 0] = np.round(ox + (points_xz[:, 0] - mn[0]) * scale).astype(np.int32)
    out[:, 1] = np.round(height - (oy + (points_xz[:, 1] - mn[1]) * scale)).astype(np.int32)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", default="outputs/meshes/4d_humans/4d_humans_transreid_3clusters_overlay/meshes_obj")
    parser.add_argument("--sam3-json", default="outputs/sam3/team_classification/sam3_team_transreid_3clusters_detections.json")
    parser.add_argument("--output-video", default="outputs/vggt/debug_full/4d_humans_own_birds_eye_no_field_h264.mp4")
    parser.add_argument("--output-json", default="outputs/vggt/debug_full/4d_humans_own_birds_eye_no_field.json")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_root = Path(args.mesh_root)
    frame_meta = load_team_metadata(Path(args.sam3_json))
    bounds = collect_bounds(mesh_root, args.max_frames)
    rendered = []
    output_frames = []
    for frame_idx in range(args.max_frames):
        canvas = np.full((args.height, args.width, 3), 18, dtype=np.uint8)
        cv2.putText(canvas, f"4D-Humans own mesh top-down frame {frame_idx}", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, "raw HMR mesh x/z coordinates; no rink fit", (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        players = []
        for obj_dir in sorted(mesh_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if not obj_dir.is_dir():
                continue
            obj_id = int(obj_dir.name)
            path = obj_dir / f"{frame_idx:08d}.obj"
            if not path.exists():
                continue
            vertices = read_obj_vertices(path)
            if len(vertices) == 0:
                continue
            xz = vertices[:, [0, 2]]
            sample = xz[:: max(1, len(xz) // 220)]
            pts = to_canvas(sample, bounds, args.width, args.height, args.margin)
            color_rgb = frame_color(frame_meta.get(frame_idx, {}), obj_id)
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            for point in pts:
                cv2.circle(canvas, tuple(point), 1, color_bgr, -1, cv2.LINE_AA)
            center_xz = np.median(xz, axis=0)
            center = to_canvas(center_xz.reshape(1, 2), bounds, args.width, args.height, args.margin)[0]
            cv2.circle(canvas, tuple(center), 7, color_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(center), 7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(obj_id), (int(center[0]) + 8, int(center[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            players.append({"object_id": obj_id, "raw_xz": [float(center_xz[0]), float(center_xz[1])]})
        rendered.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        output_frames.append({"frame": frame_idx, "players": players})
    write_h264_rgb(rendered, Path(args.output_video), args.fps)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"schema": "4d_humans_own_birds_eye_v1", "bounds_xz": [bounds[0].tolist(), bounds[1].tolist()], "frames": output_frames}, indent=2) + "\n")
    print(json.dumps({"output_video": args.output_video, "output_json": args.output_json, "frames": len(rendered)}, indent=2))


if __name__ == "__main__":
    main()
