from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

from run_vggt_birds_eye import (
    fit_plane_to_floor_homography,
    original_pixels_to_vggt,
    project_to_plane_coords,
    render_birds_eye,
    sample_mask_pixels,
    sample_world_points,
    transform_plane_to_floor,
)
from render_birds_eye_locations import FEATURE_COLORS_BGR, FLOOR_LENGTH_FT, FLOOR_WIDTH_FT, draw_floor, rgb_to_bgr, world_to_canvas, write_h264


FLOOR_FEATURE_OBJECTS = {
    "left_restraining_line": 1,
    "right_restraining_line": 2,
    "midfield_line": 3,
    "goal_crease": 4,
    "field_outline": 5,
}


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


def frame_team(frame_meta: dict, obj_id: int) -> str:
    ids = [int(v) for v in frame_meta.get("object_ids", [])]
    if obj_id in ids:
        idx = ids.index(obj_id)
        teams = frame_meta.get("teams", [])
        if idx < len(teams):
            return str(teams[idx])
    return "unknown"


def project_sam_body4d_mesh_to_image(vertices_ply: np.ndarray, focal_length: float, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    # SAM-Body4D saved meshes after the renderer's 180-degree x-axis flip.
    # Undo that flip to get the camera coordinates used by the perspective projection.
    cam = np.column_stack([vertices_ply[:, 0], -vertices_ply[:, 1], -vertices_ply[:, 2]])
    z = np.maximum(cam[:, 2], 1e-6)
    return np.column_stack(
        [
            focal_length * cam[:, 0] / z + width / 2.0,
            focal_length * cam[:, 1] / z + height / 2.0,
        ]
    ).astype(np.float64)


def sam_body4d_ply_to_camera(vertices_ply: np.ndarray) -> np.ndarray:
    return np.column_stack([vertices_ply[:, 0], -vertices_ply[:, 1], -vertices_ply[:, 2]]).astype(np.float64)


def sample_world_points_with_indices(
    world_points: np.ndarray,
    confidence: np.ndarray,
    original_points_xy: np.ndarray,
    original_shape: tuple[int, int],
    resolution: int,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(original_points_xy) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    mapped = original_pixels_to_vggt(original_points_xy, original_shape, resolution)
    xs = np.clip(np.round(mapped[:, 0]).astype(np.int32), 0, resolution - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(np.int32), 0, resolution - 1)
    pts = world_points[ys, xs].astype(np.float64)
    conf = confidence[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    return pts[valid], conf[valid], np.nonzero(valid)[0].astype(np.int64)


def fit_similarity_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    if len(source) < 6 or len(target) < 6 or len(source) != len(target):
        return None
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var < 1e-10:
        return None
    covariance = (tgt_centered.T @ src_centered) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    scale = float(np.sum(singular * d) / src_var)
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def transform_similarity(points: np.ndarray, similarity: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = similarity
    return scale * (points @ rotation.T) + translation[None, :]


def mesh_foot_pixels(mesh_path: Path, focal_path: Path, image_shape: tuple[int, int], bottom_quantile: float) -> np.ndarray:
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    focal_data = json.loads(focal_path.read_text())
    uv = project_sam_body4d_mesh_to_image(vertices, float(focal_data["focal_length"]), image_shape)
    valid = np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < image_shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image_shape[0])
    if not valid.any():
        return np.zeros((0, 2), dtype=np.float32)
    uv = uv[valid]
    cutoff = np.quantile(uv[:, 1], bottom_quantile)
    foot = uv[uv[:, 1] >= cutoff]
    return foot.astype(np.float32)


def mesh_floor_points_from_vggt(
    mesh_path: Path,
    focal_path: Path,
    image_shape: tuple[int, int],
    world_points_frame: np.ndarray,
    confidence_frame: np.ndarray,
    resolution: int,
    min_conf: float,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    H_plane_to_floor: np.ndarray,
    max_fit_vertices: int,
    max_render_vertices: int,
) -> tuple[np.ndarray, dict]:
    mesh = trimesh.load(mesh_path, process=False)
    vertices_ply = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices_ply) == 0:
        return np.zeros((0, 2), dtype=np.float64), {"fit_points": 0, "render_points": 0}
    focal_data = json.loads(focal_path.read_text())
    cam_vertices = sam_body4d_ply_to_camera(vertices_ply)
    uv_all = project_sam_body4d_mesh_to_image(vertices_ply, float(focal_data["focal_length"]), image_shape)
    valid = np.isfinite(uv_all).all(axis=1)
    valid &= (uv_all[:, 0] >= 0) & (uv_all[:, 0] < image_shape[1]) & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < image_shape[0])
    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) < 8:
        return np.zeros((0, 2), dtype=np.float64), {"fit_points": 0, "render_points": 0}
    if len(valid_indices) > max_fit_vertices:
        sample_idx = np.linspace(0, len(valid_indices) - 1, max_fit_vertices).round().astype(int)
        valid_indices = valid_indices[sample_idx]
    sampled_uv = uv_all[valid_indices].astype(np.float32)
    target_world, _, kept_local = sample_world_points_with_indices(
        world_points_frame,
        confidence_frame,
        sampled_uv,
        image_shape,
        resolution,
        min_conf,
    )
    source_cam = cam_vertices[valid_indices[kept_local]]
    similarity = fit_similarity_umeyama(source_cam, target_world)
    if similarity is None:
        return np.zeros((0, 2), dtype=np.float64), {"fit_points": int(len(source_cam)), "render_points": 0}
    if len(cam_vertices) > max_render_vertices:
        render_indices = np.linspace(0, len(cam_vertices) - 1, max_render_vertices).round().astype(int)
        render_cam = cam_vertices[render_indices]
    else:
        render_cam = cam_vertices
    mesh_world = transform_similarity(render_cam, similarity)
    plane_uv = project_to_plane_coords(mesh_world, center, basis_u, basis_v)
    floor_xy = transform_plane_to_floor(H_plane_to_floor, plane_uv)
    valid_floor = np.isfinite(floor_xy).all(axis=1)
    valid_floor &= (floor_xy[:, 0] >= -5.0) & (floor_xy[:, 0] <= FLOOR_LENGTH_FT + 5.0)
    valid_floor &= (floor_xy[:, 1] >= -5.0) & (floor_xy[:, 1] <= FLOOR_WIDTH_FT + 5.0)
    return floor_xy[valid_floor], {"fit_points": int(len(source_cam)), "render_points": int(valid_floor.sum()), "similarity_scale": float(similarity[0])}


def load_floor_feature_masks(mask_dir: Path, frame_idx: int, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    height, width = shape
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    object_ids = [int(v) for v in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    output = {}
    for feature, object_id in FLOOR_FEATURE_OBJECTS.items():
        if object_id not in object_ids:
            continue
        mask = masks[object_ids.index(object_id)]
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        output[feature] = mask
    return output


def project_floor_masks_to_floor(
    masks: dict[str, np.ndarray],
    frame_bgr: np.ndarray,
    world_points_frame: np.ndarray,
    confidence_frame: np.ndarray,
    resolution: int,
    min_conf: float,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    H_plane_to_floor: np.ndarray,
    max_points_per_feature: int,
    frame_idx: int,
) -> dict[str, np.ndarray]:
    output = {}
    for feature, mask in masks.items():
        mask_use = mask
        if feature == "field_outline":
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 90)
            mask_use = mask & yellow
        pixels = sample_mask_pixels(mask_use, max_points_per_feature, seed=frame_idx * 1009 + FLOOR_FEATURE_OBJECTS.get(feature, 0))
        pts3d, _ = sample_world_points(world_points_frame, confidence_frame, pixels, frame_bgr.shape[:2], resolution, min_conf)
        if len(pts3d) == 0:
            continue
        plane_uv = project_to_plane_coords(pts3d, center, basis_u, basis_v)
        floor_xy = transform_plane_to_floor(H_plane_to_floor, plane_uv)
        valid = np.isfinite(floor_xy).all(axis=1)
        valid &= (floor_xy[:, 0] >= -8.0) & (floor_xy[:, 0] <= FLOOR_LENGTH_FT + 8.0)
        valid &= (floor_xy[:, 1] >= -8.0) & (floor_xy[:, 1] <= FLOOR_WIDTH_FT + 8.0)
        output[feature] = floor_xy[valid]
    return output


def project_floor_masks_to_plane(
    masks: dict[str, np.ndarray],
    frame_bgr: np.ndarray,
    world_points_frame: np.ndarray,
    confidence_frame: np.ndarray,
    resolution: int,
    min_conf: float,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    max_points_per_feature: int,
    frame_idx: int,
) -> dict[str, np.ndarray]:
    output = {}
    for feature, mask in masks.items():
        mask_use = mask
        if feature == "field_outline":
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 90)
            mask_use = mask & yellow
        pixels = sample_mask_pixels(mask_use, max_points_per_feature, seed=frame_idx * 1009 + FLOOR_FEATURE_OBJECTS.get(feature, 0))
        pts3d, _ = sample_world_points(world_points_frame, confidence_frame, pixels, frame_bgr.shape[:2], resolution, min_conf)
        if len(pts3d) == 0:
            continue
        output[feature] = project_to_plane_coords(pts3d, center, basis_u, basis_v)
    return output


def transform_feature_uv_to_floor(feature_uv: dict[str, np.ndarray], H_plane_to_floor: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for feature, uv in feature_uv.items():
        floor_xy = transform_plane_to_floor(H_plane_to_floor, uv)
        valid = np.isfinite(floor_xy).all(axis=1)
        valid &= (floor_xy[:, 0] >= -8.0) & (floor_xy[:, 0] <= FLOOR_LENGTH_FT + 8.0)
        valid &= (floor_xy[:, 1] >= -8.0) & (floor_xy[:, 1] <= FLOOR_WIDTH_FT + 8.0)
        output[feature] = floor_xy[valid]
    return output


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-json", default="outputs/vggt/birds_eye_full/birds_eye_player_locations_vggt.json")
    parser.add_argument("--vggt-npz", default="outputs/vggt/birds_eye_full/vggt_predictions_compact.npz")
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--sam3-json", default="outputs/sam3/team_classification/sam3_team_transreid_3clusters_detections.json")
    parser.add_argument("--mesh-root", default="outputs/meshes/sam_body4d/sam_body4d_transreid_3clusters_overlay/mesh_4d_individual")
    parser.add_argument("--focal-root", default="outputs/meshes/sam_body4d/sam_body4d_transreid_3clusters_overlay/focal_4d_individual")
    parser.add_argument("--floor-mask-dir", default="outputs/sam2/floor_features/sam2_floor_feature_instance_masks_with_outline")
    parser.add_argument("--output-dir", default="outputs/vggt/sam_body4d_fused")
    parser.add_argument("--min-depth-conf", type=float, default=1.0)
    parser.add_argument("--bottom-quantile", type=float, default=0.90)
    parser.add_argument("--max-mesh-fit-vertices", type=int, default=900)
    parser.add_argument("--max-mesh-render-vertices", type=int, default=900)
    parser.add_argument("--max-mask-points-per-feature", type=int, default=700)
    parser.add_argument("--fit-pitch-per-frame", action="store_true", help="Refit plane-to-rink transform from each frame's top-down landmark projections before moving meshes.")
    parser.add_argument("--pitch-fit-regularization", type=float, default=0.04)
    parser.add_argument("--min-pitch-fit-points", type=int, default=80)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=54)
    parser.add_argument("--trail-frames", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vggt = json.loads(Path(args.vggt_json).read_text())
    plane = vggt["floor_plane"]
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    H_plane_to_floor = np.asarray(vggt["plane_to_floor_homography"], dtype=np.float64)
    resolution = int(vggt["vggt_resolution"])

    frame_indices, depth_conf, world_points = load_vggt_world_points(Path(args.vggt_npz), Path(args.vggt_repo))
    frame_meta = load_team_metadata(Path(args.sam3_json))
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    mesh_root = Path(args.mesh_root)
    focal_root = Path(args.focal_root)
    floor_mask_dir = Path(args.floor_mask_dir)

    output_frames = []
    mesh_frames = []
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        masks = load_floor_feature_masks(floor_mask_dir, frame_idx, frame.shape[:2])
        feature_plane_uv = project_floor_masks_to_plane(
            masks,
            frame,
            world_points[local_idx],
            depth_conf[local_idx],
            resolution,
            args.min_depth_conf,
            center,
            basis_u,
            basis_v,
            args.max_mask_points_per_feature,
            frame_idx,
        )
        frame_H_plane_to_floor = H_plane_to_floor
        pitch_fit_metrics = {"fallback": True, "reason": "global_vggt_alignment"}
        if args.fit_pitch_per_frame and sum(len(points) for points in feature_plane_uv.values()) >= args.min_pitch_fit_points:
            try:
                frame_H_plane_to_floor, fit_metrics = fit_plane_to_floor_homography(feature_plane_uv, args.pitch_fit_regularization)
                pitch_fit_metrics = {"fallback": False, **fit_metrics}
            except Exception as exc:
                pitch_fit_metrics = {"fallback": True, "reason": str(exc)}
        players = []
        mesh_players = []
        for obj_dir in sorted(mesh_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if not obj_dir.is_dir():
                continue
            obj_id = int(obj_dir.name)
            mesh_path = obj_dir / f"{frame_idx:08d}.ply"
            focal_path = focal_root / str(obj_id) / f"{frame_idx:08d}.json"
            if not mesh_path.exists() or not focal_path.exists():
                continue
            mesh_floor_xy, mesh_metrics = mesh_floor_points_from_vggt(
                mesh_path,
                focal_path,
                frame.shape[:2],
                world_points[local_idx],
                depth_conf[local_idx],
                resolution,
                args.min_depth_conf,
                center,
                basis_u,
                basis_v,
                frame_H_plane_to_floor,
                args.max_mesh_fit_vertices,
                args.max_mesh_render_vertices,
            )
            foot_pixels = mesh_foot_pixels(mesh_path, focal_path, frame.shape[:2], args.bottom_quantile)
            pts3d, conf = sample_world_points(world_points[local_idx], depth_conf[local_idx], foot_pixels, frame.shape[:2], resolution, args.min_depth_conf)
            if len(pts3d) == 0:
                continue
            plane_uv = project_to_plane_coords(pts3d, center, basis_u, basis_v)
            floor_xy = transform_plane_to_floor(frame_H_plane_to_floor, np.median(plane_uv, axis=0, keepdims=True))[0]
            floor_xy[0] = np.clip(floor_xy[0], 0.0, FLOOR_LENGTH_FT)
            floor_xy[1] = np.clip(floor_xy[1], 0.0, FLOOR_WIDTH_FT)
            meta = frame_meta.get(frame_idx, {})
            players.append(
                {
                    "object_id": obj_id,
                    "team": frame_team(meta, obj_id),
                    "team_color": frame_color(meta, obj_id),
                    "floor_xy_ft": [float(floor_xy[0]), float(floor_xy[1])],
                    "source": "sam_body4d_projected_foot_vertices_vggt_depth",
                    "mesh_foot_pixels": int(len(foot_pixels)),
                    "vggt_samples": int(len(pts3d)),
                    "median_depth_conf": float(np.median(conf)) if len(conf) else 0.0,
                    "mesh_fit_points": int(mesh_metrics.get("fit_points", 0)),
                    "mesh_render_points": int(mesh_metrics.get("render_points", 0)),
                }
            )
            if len(mesh_floor_xy):
                mesh_players.append(
                    {
                        "object_id": obj_id,
                        "team_color": frame_color(meta, obj_id),
                        "floor_points": mesh_floor_xy,
                        "metrics": mesh_metrics,
                    }
                )
        output_frames.append({"frame": int(frame_idx), "pitch_fit_metrics": pitch_fit_metrics, "players": players})

        mask_floor = transform_feature_uv_to_floor(feature_plane_uv, frame_H_plane_to_floor)
        mesh_frames.append({"frame": int(frame_idx), "players": mesh_players, "field_masks": mask_floor, "pitch_fit_metrics": pitch_fit_metrics})

    suffix = "_pitchfit" if args.fit_pitch_per_frame else ""
    output_json = output_dir / f"birds_eye_player_locations_sam_body4d_vggt{suffix}.json"
    output_video = output_dir / f"birds_eye_player_locations_sam_body4d_vggt{suffix}_h264.mp4"
    mesh_output_json = output_dir / f"birds_eye_sam_body4d_meshes_and_field_masks_vggt{suffix}.json"
    mesh_output_video = output_dir / f"birds_eye_sam_body4d_meshes_and_field_masks_vggt{suffix}_h264.mp4"
    output_json.write_text(
        json.dumps(
            {
                "schema": "sam_body4d_vggt_birds_eye_v1",
                "vggt_json": args.vggt_json,
                "vggt_npz": args.vggt_npz,
                "mesh_root": args.mesh_root,
                "focal_root": args.focal_root,
                "floor_mask_dir": args.floor_mask_dir,
                "fit_pitch_per_frame": bool(args.fit_pitch_per_frame),
                "frames": output_frames,
            },
            indent=2,
        )
        + "\n"
    )
    render_birds_eye(output_frames, output_video, args.fps, args.width, args.height, args.margin, args.trail_frames)

    mesh_json_frames = []
    rendered_mesh_frames = []
    for item in mesh_frames:
        canvas = draw_floor(args.width, args.height, args.margin)
        cv2.putText(canvas, f"SAM-Body4D meshes + SAM2 field masks frame {item['frame']}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        mask_counts = {}
        for feature, points in item["field_masks"].items():
            color = FEATURE_COLORS_BGR.get(feature, (255, 255, 255))
            mask_counts[feature] = int(len(points))
            for x, y in points:
                px, py = world_to_canvas(float(np.clip(x, 0.0, FLOOR_LENGTH_FT)), float(np.clip(y, 0.0, FLOOR_WIDTH_FT)), args.width, args.height, args.margin)
                cv2.circle(canvas, (px, py), 2, color, -1, cv2.LINE_AA)
        players_json = []
        for player in item["players"]:
            color = rgb_to_bgr(player["team_color"])
            points = player["floor_points"]
            if len(points) == 0:
                continue
            canvas_points = []
            for x, y in points:
                px, py = world_to_canvas(float(np.clip(x, 0.0, FLOOR_LENGTH_FT)), float(np.clip(y, 0.0, FLOOR_WIDTH_FT)), args.width, args.height, args.margin)
                canvas_points.append((px, py))
            for px, py in canvas_points:
                cv2.circle(canvas, (px, py), 1, color, -1, cv2.LINE_AA)
            median = np.median(points, axis=0)
            cx, cy = world_to_canvas(float(np.clip(median[0], 0.0, FLOOR_LENGTH_FT)), float(np.clip(median[1], 0.0, FLOOR_WIDTH_FT)), args.width, args.height, args.margin)
            cv2.circle(canvas, (cx, cy), 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(player["object_id"]), (cx + 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1, cv2.LINE_AA)
            players_json.append(
                {
                    "object_id": int(player["object_id"]),
                    "team_color": [int(v) for v in player["team_color"]],
                    "mesh_floor_point_count": int(len(points)),
                    "median_floor_xy_ft": [float(median[0]), float(median[1])],
                    "metrics": player["metrics"],
                }
            )
        rendered_mesh_frames.append(canvas)
        mesh_json_frames.append(
            {
                "frame": int(item["frame"]),
                "pitch_fit_metrics": item.get("pitch_fit_metrics", {}),
                "field_mask_points": mask_counts,
                "players": players_json,
            }
        )
    write_h264(rendered_mesh_frames, mesh_output_video, args.fps)
    mesh_output_json.write_text(
        json.dumps(
            {
                "schema": "sam_body4d_vggt_meshes_and_field_masks_v1",
                "vggt_json": args.vggt_json,
                "vggt_npz": args.vggt_npz,
                "mesh_root": args.mesh_root,
                "focal_root": args.focal_root,
                "floor_mask_dir": args.floor_mask_dir,
                "fit_pitch_per_frame": bool(args.fit_pitch_per_frame),
                "frames": mesh_json_frames,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_video": str(output_video),
                "mesh_output_json": str(mesh_output_json),
                "mesh_output_video": str(mesh_output_video),
                "frames": len(output_frames),
                "players": sum(len(f["players"]) for f in output_frames),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
