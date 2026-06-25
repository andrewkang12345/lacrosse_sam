from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from render_birds_eye_locations import (
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    draw_floor,
    rgb_to_bgr,
    world_to_canvas,
    write_h264,
)
from run_vggt_birds_eye import original_pixels_to_vggt, project_to_plane_coords, transform_plane_to_floor


DEFAULT_COLORS = [
    [255, 80, 40],
    [70, 170, 255],
    [245, 210, 55],
    [100, 230, 150],
    [210, 120, 255],
    [255, 160, 80],
    [80, 220, 240],
    [230, 230, 230],
]


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_frames_bgr(frames_dir: Path, max_frames: int = 0) -> list[np.ndarray]:
    paths = sorted_frame_paths(frames_dir)
    if max_frames:
        paths = paths[:max_frames]
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read frame: {path}")
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No frame_*.jpg files found in {frames_dir}")
    return frames


def load_sam3_metadata(path: Path, max_frames: int = 0) -> dict:
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    if max_frames:
        frames = [item for item in frames if int(item["frame"]) < max_frames]
    data["frames"] = frames
    return data


def load_instance_masks(mask_dir: Path, frame_idx: int, shape: tuple[int, int]) -> tuple[list[int], np.ndarray]:
    height, width = shape
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return [], np.zeros((0, height, width), dtype=bool)
    data = np.load(path)
    object_ids = [int(value) for value in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    if masks.ndim != 3:
        return [], np.zeros((0, height, width), dtype=bool)
    if masks.shape[1:] != (height, width):
        resized = []
        for mask in masks:
            resized.append(cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool))
        masks = np.stack(resized, axis=0) if resized else np.zeros((0, height, width), dtype=bool)
    return object_ids, masks


def mask_for_object(mask_dir: Path, frame_idx: int, object_id: int, shape: tuple[int, int]) -> np.ndarray:
    object_ids, masks = load_instance_masks(mask_dir, frame_idx, shape)
    if object_id not in object_ids:
        return np.zeros(shape, dtype=bool)
    return masks[object_ids.index(object_id)]


def object_counts(sam3: dict) -> Counter:
    counts: Counter = Counter()
    for frame in sam3.get("frames", []):
        for object_id in frame.get("object_ids", []):
            counts[int(object_id)] += 1
    return counts


def first_object_frames(sam3: dict, min_frames: int) -> dict[int, int]:
    counts = object_counts(sam3)
    first: dict[int, int] = {}
    for frame in sam3.get("frames", []):
        frame_idx = int(frame["frame"])
        for object_id in frame.get("object_ids", []):
            object_id = int(object_id)
            if counts[object_id] >= min_frames and object_id not in first:
                first[object_id] = frame_idx
    return first


def object_presence(sam3: dict) -> dict[int, set[int]]:
    presence: dict[int, set[int]] = defaultdict(set)
    for frame in sam3.get("frames", []):
        frame_idx = int(frame["frame"])
        for object_id in frame.get("object_ids", []):
            presence[int(object_id)].add(frame_idx)
    return presence


def sample_query_points(
    mask: np.ndarray,
    points_per_object: int,
    seed: int,
    min_area: int,
    vertical_min_frac: float,
    vertical_max_frac: float,
) -> np.ndarray:
    if int(mask.sum()) < min_area:
        return np.zeros((0, 2), dtype=np.float32)
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if eroded.sum() >= min_area:
        mask = eroded
    ys_all, xs_all = np.where(mask)
    if len(xs_all) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    y_min, y_max = float(ys_all.min()), float(ys_all.max())
    y0 = y_min + vertical_min_frac * max(1.0, y_max - y_min)
    y1 = y_min + vertical_max_frac * max(1.0, y_max - y_min)
    keep = (ys_all >= y0) & (ys_all <= y1)
    if keep.sum() >= min_area // 2:
        xs_all = xs_all[keep]
        ys_all = ys_all[keep]
    points = np.column_stack([xs_all, ys_all]).astype(np.float32)
    centroid = np.median(points, axis=0, keepdims=True)
    if len(points) <= points_per_object:
        return points.astype(np.float32)
    rng = np.random.default_rng(seed)
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    weights = distances[ys_all, xs_all].astype(np.float64)
    if np.isfinite(weights).all() and weights.sum() > 0:
        weights = weights / weights.sum()
        choice = rng.choice(len(points), size=max(0, points_per_object - 1), replace=False, p=weights)
    else:
        choice = rng.choice(len(points), size=max(0, points_per_object - 1), replace=False)
    sampled = np.concatenate([centroid, points[choice]], axis=0)
    return sampled.astype(np.float32)


def build_queries(
    sam3: dict,
    mask_dir: Path,
    frame_shape: tuple[int, int],
    min_object_frames: int,
    points_per_object: int,
    min_mask_area: int,
    max_objects: int,
    vertical_min_frac: float,
    vertical_max_frac: float,
) -> tuple[np.ndarray, list[dict]]:
    first = first_object_frames(sam3, min_object_frames)
    if max_objects > 0:
        counts = object_counts(sam3)
        keep_ids = {object_id for object_id, _ in counts.most_common(max_objects)}
        first = {object_id: frame_idx for object_id, frame_idx in first.items() if object_id in keep_ids}
    query_rows = []
    metadata = []
    for object_id, frame_idx in sorted(first.items(), key=lambda item: (item[1], item[0])):
        mask = mask_for_object(mask_dir, frame_idx, object_id, frame_shape)
        points = sample_query_points(
            mask,
            points_per_object=points_per_object,
            seed=object_id + 1009 * frame_idx,
            min_area=min_mask_area,
            vertical_min_frac=vertical_min_frac,
            vertical_max_frac=vertical_max_frac,
        )
        for point_idx, (x, y) in enumerate(points.tolist()):
            query_rows.append([float(frame_idx), float(x), float(y)])
            metadata.append({"object_id": int(object_id), "query_frame": int(frame_idx), "query_index": int(point_idx)})
    if not query_rows:
        raise RuntimeError("No query points could be sampled from the SAM3 player masks.")
    return np.asarray(query_rows, dtype=np.float32), metadata


def load_cotracker_model(repo: Path, device: str):
    sys.path.insert(0, str(repo.resolve()))
    import hubconf  # type: ignore

    model = hubconf.cotracker3_offline(pretrained=True)
    model = model.to(device)
    model.eval()
    return model


def run_cotracker(frames_bgr: list[np.ndarray], queries: np.ndarray, repo: Path, device: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    video_np = np.stack(rgb, axis=0)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(device)
    query_tensor = torch.from_numpy(queries)[None].float().to(device)
    model = load_cotracker_model(repo, device)
    with torch.no_grad():
        tracks, visibility = model(video, queries=query_tensor, backward_tracking=False)
    return tracks[0].detach().cpu().numpy(), visibility[0].detach().cpu().numpy().astype(bool)


def load_vggt_world_points(vggt_npz: Path, vggt_repo: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys.path.insert(0, str(vggt_repo.resolve()))
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    data = np.load(vggt_npz)
    frame_indices = data["frame_indices"].astype(int)
    depth_map = data["depth_map"]
    depth_conf = data["depth_conf"]
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]
    world_points = unproject_depth_map_to_point_map(depth_map, data["extrinsic"], data["intrinsic"])
    return frame_indices, depth_conf, world_points


def confidence_thresholds(depth_conf: np.ndarray, percentile: float, absolute: float | None) -> np.ndarray:
    if absolute is not None:
        return np.full((depth_conf.shape[0],), float(absolute), dtype=np.float64)
    thresholds = []
    for conf in depth_conf:
        valid = conf[np.isfinite(conf)]
        thresholds.append(float(np.percentile(valid, percentile)) if valid.size else float("inf"))
    return np.asarray(thresholds, dtype=np.float64)


def sample_world_points_at_pixels(
    world_points: np.ndarray,
    confidence: np.ndarray,
    pixels_xy: np.ndarray,
    original_shape: tuple[int, int],
    resolution: int,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels_xy) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    mapped = original_pixels_to_vggt(pixels_xy.astype(np.float64), original_shape, resolution)
    xs = np.clip(np.round(mapped[:, 0]).astype(np.int32), 0, resolution - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(np.int32), 0, resolution - 1)
    pts = world_points[ys, xs].astype(np.float64)
    conf = confidence[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    return pts[valid], conf[valid]


def color_for_object(object_id: int) -> list[int]:
    base = DEFAULT_COLORS[object_id % len(DEFAULT_COLORS)]
    jitter = np.asarray([(object_id * 37) % 45, (object_id * 17) % 35, (object_id * 23) % 40], dtype=int)
    return np.clip(np.asarray(base, dtype=int) - 20 + jitter, 45, 255).astype(int).tolist()


def aggregate_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    query_metadata: list[dict],
    mask_dir: Path,
    frames_bgr: list[np.ndarray],
    vggt_npz: Path,
    vggt_repo: Path,
    field_fit_json: Path,
    min_depth_conf: float | None,
    min_depth_conf_percentile: float,
    mask_gate: bool,
    mask_dilation: int,
    min_visible_points: int,
    ignore_cotracker_visibility: bool,
    presence_by_object: dict[int, set[int]] | None,
) -> tuple[list[dict], dict[int, list[dict]]]:
    field_fit = json.loads(field_fit_json.read_text())
    plane = field_fit["floor_plane"]
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    H_plane_to_floor = np.asarray(field_fit["plane_to_floor_homography"], dtype=np.float64)
    resolution = int(field_fit["vggt_resolution"])

    frame_indices, depth_conf, world_points = load_vggt_world_points(vggt_npz, vggt_repo)
    vggt_by_frame = {int(frame_idx): local_idx for local_idx, frame_idx in enumerate(frame_indices.tolist())}
    thresholds = confidence_thresholds(depth_conf, min_depth_conf_percentile, min_depth_conf)
    shape = frames_bgr[0].shape[:2]
    kernel = np.ones((max(1, mask_dilation), max(1, mask_dilation)), np.uint8)

    query_indices_by_object: dict[int, list[int]] = defaultdict(list)
    query_frame_by_index = {}
    for idx, item in enumerate(query_metadata):
        query_indices_by_object[int(item["object_id"])].append(idx)
        query_frame_by_index[idx] = int(item["query_frame"])

    frames_out = []
    trajectories: dict[int, list[dict]] = defaultdict(list)
    for frame_idx in range(len(frames_bgr)):
        players = []
        if frame_idx not in vggt_by_frame:
            frames_out.append({"frame": frame_idx, "players": players})
            continue
        local_idx = vggt_by_frame[frame_idx]
        for object_id, point_indices in sorted(query_indices_by_object.items()):
            if presence_by_object is not None and frame_idx not in presence_by_object.get(object_id, set()):
                continue
            usable = []
            for point_idx in point_indices:
                if frame_idx < query_frame_by_index[point_idx]:
                    continue
                if not ignore_cotracker_visibility and not visibility[frame_idx, point_idx]:
                    continue
                x, y = tracks[frame_idx, point_idx]
                if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < shape[1] and 0 <= y < shape[0]):
                    continue
                usable.append([float(x), float(y)])
            if not usable:
                continue
            usable_xy = np.asarray(usable, dtype=np.float32)
            mask_points = usable_xy
            mask_available = False
            if mask_gate:
                mask = mask_for_object(mask_dir, frame_idx, object_id, shape)
                if mask.any():
                    mask_available = True
                    if mask_dilation > 1:
                        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                    xs = np.clip(np.round(usable_xy[:, 0]).astype(int), 0, shape[1] - 1)
                    ys = np.clip(np.round(usable_xy[:, 1]).astype(int), 0, shape[0] - 1)
                    inside = mask[ys, xs]
                    mask_points = usable_xy[inside]
            if len(mask_points) < min_visible_points:
                continue
            pts3d, conf = sample_world_points_at_pixels(
                world_points[local_idx],
                depth_conf[local_idx],
                mask_points,
                shape,
                resolution,
                thresholds[local_idx],
            )
            if len(pts3d) == 0:
                continue
            plane_uv = project_to_plane_coords(pts3d, center, basis_u, basis_v)
            floor_xy = transform_plane_to_floor(H_plane_to_floor, plane_uv)
            valid = np.isfinite(floor_xy).all(axis=1)
            if not valid.any():
                continue
            floor_xy = floor_xy[valid]
            median_xy = np.median(floor_xy, axis=0)
            median_xy[0] = np.clip(median_xy[0], 0.0, FLOOR_LENGTH_FT)
            median_xy[1] = np.clip(median_xy[1], 0.0, FLOOR_WIDTH_FT)
            camera_xy = np.median(mask_points, axis=0)
            color = color_for_object(object_id)
            item = {
                "object_id": int(object_id),
                "floor_xy_ft": [float(median_xy[0]), float(median_xy[1])],
                "camera_xy": [float(camera_xy[0]), float(camera_xy[1])],
                "visible_track_points": int(len(usable_xy)),
                "cotracker_visibility_used": not bool(ignore_cotracker_visibility),
                "mask_gated_track_points": int(len(mask_points)),
                "mask_available": bool(mask_available),
                "vggt_points": int(len(pts3d)),
                "median_depth_conf": float(np.median(conf)) if len(conf) else 0.0,
                "color": color,
            }
            players.append(item)
            trajectories[object_id].append({"frame": frame_idx, **item})
        frames_out.append({"frame": frame_idx, "players": players})
    return frames_out, trajectories


def render_camera_overlay(
    frames_bgr: list[np.ndarray],
    frames_out: list[dict],
    output_video: Path,
    fps: float,
    trail_frames: int,
) -> None:
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_frames))
    rendered = []
    for frame, item in zip(frames_bgr, frames_out):
        canvas = frame.copy()
        cv2.putText(canvas, f"CoTracker SAM3 player tracks frame {item['frame']}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        for player in item["players"]:
            object_id = int(player["object_id"])
            color = rgb_to_bgr(player["color"])
            x, y = player["camera_xy"]
            history[object_id].append((int(round(x)), int(round(y))))
            pts = list(history[object_id])
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(object_id), (int(x) + 7, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        rendered.append(canvas)
    write_h264(rendered, output_video, fps=fps)


def render_birds_eye(
    frames_out: list[dict],
    output_video: Path,
    fps: float,
    width: int,
    height: int,
    margin: int,
    trail_frames: int,
) -> None:
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_frames))
    rendered = []
    for item in frames_out:
        canvas = draw_floor(width, height, margin)
        cv2.putText(canvas, f"VGGT + CoTracker player trajectories frame {item['frame']}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        for player in item["players"]:
            object_id = int(player["object_id"])
            color = rgb_to_bgr(player["color"])
            x, y = player["floor_xy_ft"]
            px, py = world_to_canvas(float(x), float(y), width, height, margin)
            history[object_id].append((px, py))
            pts = list(history[object_id])
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(object_id), (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        rendered.append(canvas)
    write_h264(rendered, output_video, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track SAM3 player-mask points with CoTracker and project trajectories through VGGT to floor coordinates.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sam3-json", required=True)
    parser.add_argument("--player-mask-dir", required=True)
    parser.add_argument("--vggt-npz", required=True)
    parser.add_argument("--field-fit-json", required=True)
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--cotracker-repo", default="third_party/co-tracker")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--points-per-object", type=int, default=3)
    parser.add_argument("--min-object-frames", type=int, default=10)
    parser.add_argument("--min-mask-area", type=int, default=24)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--vertical-min-frac", type=float, default=0.15)
    parser.add_argument("--vertical-max-frac", type=float, default=0.95)
    parser.add_argument("--min-depth-conf", type=float, default=None)
    parser.add_argument("--min-depth-conf-percentile", type=float, default=20.0)
    parser.add_argument("--disable-mask-gate", action="store_true")
    parser.add_argument("--ignore-cotracker-visibility", action="store_true", help="Use CoTracker positions after each query frame even when the model's strict visibility boolean is false.")
    parser.add_argument("--require-sam3-presence", action="store_true", help="Only emit an object trajectory point on frames where that SAM3 object ID is present.")
    parser.add_argument("--mask-dilation", type=int, default=13)
    parser.add_argument("--min-visible-points", type=int, default=1)
    parser.add_argument("--trail-frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=54)
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
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(json.dumps({"query_points": int(len(queries)), "objects": len({m["object_id"] for m in query_metadata}), "frames": len(frames_bgr), "device": device}, indent=2))
    tracks, visibility = run_cotracker(frames_bgr, queries, Path(args.cotracker_repo), device)
    frames_out, trajectories = aggregate_tracks(
        tracks,
        visibility,
        query_metadata,
        Path(args.player_mask_dir),
        frames_bgr,
        Path(args.vggt_npz),
        Path(args.vggt_repo),
        Path(args.field_fit_json),
        args.min_depth_conf,
        args.min_depth_conf_percentile,
        not args.disable_mask_gate,
        args.mask_dilation,
        args.min_visible_points,
        args.ignore_cotracker_visibility,
        object_presence(sam3) if args.require_sam3_presence else None,
    )

    camera_video = output_dir / "cotracker_sam3_player_tracks_camera_h264.mp4"
    birds_eye_video = output_dir / "cotracker_vggt_player_trajectories_birds_eye_h264.mp4"
    output_json = output_dir / "cotracker_vggt_player_trajectories.json"
    tracks_npz = output_dir / "cotracker_tracks_raw.npz"
    render_camera_overlay(frames_bgr, frames_out, camera_video, args.fps, args.trail_frames)
    render_birds_eye(frames_out, birds_eye_video, args.fps, args.width, args.height, args.margin, args.trail_frames)
    output_json.write_text(
        json.dumps(
            {
                "schema": "cotracker_vggt_player_trajectories_v1",
                "frames_dir": args.frames_dir,
                "sam3_json": args.sam3_json,
                "player_mask_dir": args.player_mask_dir,
                "vggt_npz": args.vggt_npz,
                "field_fit_json": args.field_fit_json,
                "query_metadata": query_metadata,
                "frames": frames_out,
                "trajectory_lengths": {str(object_id): len(items) for object_id, items in sorted(trajectories.items())},
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
                "camera_video": str(camera_video),
                "birds_eye_video": str(birds_eye_video),
                "frames": len(frames_out),
                "trajectory_objects": len(trajectories),
                "trajectory_points": sum(len(items) for items in trajectories.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
