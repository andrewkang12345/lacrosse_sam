from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import viser

from render_birds_eye_locations import rgb_to_bgr


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_frame_rgb(path: Path, max_width: int) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if max_width > 0 and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (max_width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return frame


def load_tracks(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["frames"]


def color_rgb(color: list[int]) -> np.ndarray:
    return np.asarray(color, dtype=np.uint8)


def image_xy_to_scene(points_xy: np.ndarray, image_shape: tuple[int, int], render_width: float, render_height: float, z: float) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    height, width = image_shape
    x = (points_xy[:, 0] / max(1.0, width - 1.0) - 0.5) * render_width
    y = (0.5 - points_xy[:, 1] / max(1.0, height - 1.0)) * render_height
    return np.column_stack([x, y, np.full(len(points_xy), z)]).astype(np.float32)


def frame_players_by_id(frames: list[dict]) -> dict[int, dict[int, dict]]:
    out: dict[int, dict[int, dict]] = {}
    for frame in frames:
        out[int(frame["frame"])] = {int(player["object_id"]): player for player in frame.get("players", [])}
    return out


def trail_segments(
    frames_by_idx: dict[int, dict[int, dict]],
    frame_idx: int,
    image_shape: tuple[int, int],
    render_width: float,
    render_height: float,
    trail_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    by_object: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    start = max(0, frame_idx - trail_frames)
    for idx in range(start, frame_idx + 1):
        for object_id, player in frames_by_idx.get(idx, {}).items():
            by_object[object_id].append((idx, player))
    segments = []
    colors = []
    for object_id, items in by_object.items():
        items = sorted(items, key=lambda item: item[0])
        for (_, p0), (_, p1) in zip(items[:-1], items[1:]):
            xy = np.asarray([p0["camera_xy"], p1["camera_xy"]], dtype=np.float32)
            scene = image_xy_to_scene(xy, image_shape, render_width, render_height, z=0.04)
            segments.append(scene)
            colors.append(np.stack([color_rgb(p1["color"]), color_rgb(p1["color"])], axis=0))
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8)
    return np.stack(segments, axis=0).astype(np.float32), np.stack(colors, axis=0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Viser viewer for source-fps CoTracker player tracks.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--tracks-json", required=True)
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--fps", type=float, default=59.94005994)
    parser.add_argument("--trail-frames", type=int, default=120)
    parser.add_argument("--max-image-width", type=int, default=1280)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = sorted_frame_paths(Path(args.frames_dir))
    if not frame_paths:
        raise SystemExit(f"No frame_*.jpg files found in {args.frames_dir}")
    frames = load_tracks(Path(args.tracks_json))
    frames_by_idx = frame_players_by_id(frames)
    first_image = load_frame_rgb(frame_paths[0], args.max_image_width)
    image_shape = first_image.shape[:2]
    aspect = image_shape[1] / image_shape[0]
    render_width = 16.0
    render_height = render_width / aspect

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", visible=False)
    image_handle = server.scene.add_image("/video_frame", first_image, render_width=render_width, render_height=render_height, jpeg_quality=88)
    point_handle = server.scene.add_point_cloud("/players", np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), point_size=0.085, point_shape="circle")
    line_handle = server.scene.add_line_segments("/trails", np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.uint8), line_width=3)
    label_handles: dict[int, viser.LabelHandle] = {}

    gui_frame = server.gui.add_slider("Frame", min=0, max=len(frame_paths) - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("Play", initial_value=False)
    gui_trail = server.gui.add_slider("Trail Frames", min=0, max=max(1, len(frame_paths) - 1), step=1, initial_value=min(args.trail_frames, len(frame_paths) - 1))
    gui_time = server.gui.add_text("Time", initial_value="0.000s", disabled=True)

    state = {"last_frame": -1}

    def update(frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, len(frame_paths) - 1))
        if state["last_frame"] == frame_idx:
            return
        state["last_frame"] = frame_idx
        img = load_frame_rgb(frame_paths[frame_idx], args.max_image_width)
        image_handle.image = img
        players = list(frames_by_idx.get(frame_idx, {}).values())
        if players:
            xy = np.asarray([player["camera_xy"] for player in players], dtype=np.float32)
            point_handle.points = image_xy_to_scene(xy, image_shape, render_width, render_height, z=0.06)
            point_handle.colors = np.asarray([player["color"] for player in players], dtype=np.uint8)
        else:
            point_handle.points = np.zeros((0, 3), dtype=np.float32)
            point_handle.colors = np.zeros((0, 3), dtype=np.uint8)

        segments, segment_colors = trail_segments(frames_by_idx, frame_idx, image_shape, render_width, render_height, int(gui_trail.value))
        line_handle.points = segments
        line_handle.colors = segment_colors

        active_ids = {int(player["object_id"]) for player in players}
        for object_id in list(label_handles):
            if object_id not in active_ids:
                label_handles[object_id].visible = False
        for player in players:
            object_id = int(player["object_id"])
            pos = image_xy_to_scene(np.asarray([player["camera_xy"]], dtype=np.float32), image_shape, render_width, render_height, z=0.09)[0]
            text = str(object_id)
            if object_id not in label_handles:
                label_handles[object_id] = server.scene.add_label(f"/labels/{object_id}", text=text, position=pos)
            label_handles[object_id].position = pos + np.asarray([0.22, 0.08, 0.0], dtype=np.float32)
            label_handles[object_id].visible = True
        gui_time.value = f"{frame_idx / args.fps:.3f}s"

    @gui_frame.on_update
    def _(_) -> None:
        update(int(gui_frame.value))

    @gui_trail.on_update
    def _(_) -> None:
        state["last_frame"] = -1
        update(int(gui_frame.value))

    def play_loop() -> None:
        while True:
            if gui_play.value:
                next_frame = int(gui_frame.value) + 1
                if next_frame >= len(frame_paths):
                    next_frame = 0
                gui_frame.value = next_frame
                update(next_frame)
                time.sleep(max(1.0 / args.fps, 0.005))
            else:
                time.sleep(0.05)

    update(0)
    thread = threading.Thread(target=play_loop, daemon=True)
    thread.start()
    print(f"Open http://127.0.0.1:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
