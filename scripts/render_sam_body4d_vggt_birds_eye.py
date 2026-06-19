from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

from run_vggt_birds_eye import (
    project_to_plane_coords,
    render_birds_eye,
    sample_world_points,
    transform_plane_to_floor,
)
from render_birds_eye_locations import FLOOR_LENGTH_FT, FLOOR_WIDTH_FT


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
    parser.add_argument("--output-dir", default="outputs/vggt/sam_body4d_fused")
    parser.add_argument("--min-depth-conf", type=float, default=1.0)
    parser.add_argument("--bottom-quantile", type=float, default=0.90)
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

    output_frames = []
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        players = []
        for obj_dir in sorted(mesh_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if not obj_dir.is_dir():
                continue
            obj_id = int(obj_dir.name)
            mesh_path = obj_dir / f"{frame_idx:08d}.ply"
            focal_path = focal_root / str(obj_id) / f"{frame_idx:08d}.json"
            if not mesh_path.exists() or not focal_path.exists():
                continue
            foot_pixels = mesh_foot_pixels(mesh_path, focal_path, frame.shape[:2], args.bottom_quantile)
            pts3d, conf = sample_world_points(world_points[local_idx], depth_conf[local_idx], foot_pixels, frame.shape[:2], resolution, args.min_depth_conf)
            if len(pts3d) == 0:
                continue
            plane_uv = project_to_plane_coords(pts3d, center, basis_u, basis_v)
            floor_xy = transform_plane_to_floor(H_plane_to_floor, np.median(plane_uv, axis=0, keepdims=True))[0]
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
                }
            )
        output_frames.append({"frame": int(frame_idx), "players": players})

    output_json = output_dir / "birds_eye_player_locations_sam_body4d_vggt.json"
    output_video = output_dir / "birds_eye_player_locations_sam_body4d_vggt_h264.mp4"
    output_json.write_text(
        json.dumps(
            {
                "schema": "sam_body4d_vggt_birds_eye_v1",
                "vggt_json": args.vggt_json,
                "vggt_npz": args.vggt_npz,
                "mesh_root": args.mesh_root,
                "focal_root": args.focal_root,
                "frames": output_frames,
            },
            indent=2,
        )
        + "\n"
    )
    render_birds_eye(output_frames, output_video, args.fps, args.width, args.height, args.margin, args.trail_frames)
    print(json.dumps({"output_json": str(output_json), "output_video": str(output_video), "frames": len(output_frames), "players": sum(len(f["players"]) for f in output_frames)}, indent=2))


if __name__ == "__main__":
    main()
