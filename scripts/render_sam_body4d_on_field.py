"""SAM-Body4D player meshes placed in the VGGT-Omega scene and drawn from above on the field.

Inputs are the 518x518 export of export_vggt_omega_vggt_layout.py (depth, cameras and the
rigid plane->field fit) and the per-player meshes of run_sam_body4d_from_sam3_boxes.py.
Each mesh keeps its orientation in SAM-Body4D's camera frame. Its scale and offset are
fitted to the VGGT-Omega depth under its projected vertices (four rounds, each dropping
the worst 30% of points), and the scales of one frame are kept within 0.72-1.28x of
their median. With --fixed-field-player-size, all field players are first resized to
their common median body size; tracks more than --goalie-extent-threshold times larger,
and --goalie-object-ids, keep their own. The placed vertices are mapped onto the field
through the floor plane and the rigid plane->field fit.

Writes meshes_on_field_h264.mp4 (each mesh's vertices from above, in its team colour, on
the fitted field) and meshes_on_field.json (per frame and player: the
median field point and the fit). view_vggt_reconstruction.py shows the same placement
in 3D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

import field_model as fm
from fit_field_plane_landmarks import field_canvas
from render_field_birds_eye import FOOTER_PX, HEADER_PX, TEAM_STYLE_BGR, legend
from video_utils import write_h264_video

SCALE_CLAMP = (0.72, 1.28)  # per-frame mesh scales, relative to their median


# ---------------------------------------------------------------------------
# SAM-Body4D meshes
# ---------------------------------------------------------------------------


def ply_to_camera(vertices_ply: np.ndarray) -> np.ndarray:
    """SAM-Body4D saves meshes after its renderer's 180-degree x-axis flip; undo it."""
    return np.column_stack([vertices_ply[:, 0], -vertices_ply[:, 1], -vertices_ply[:, 2]]).astype(np.float64)


def project_to_image(vertices_camera: np.ndarray, focal_length: float, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    z = np.maximum(vertices_camera[:, 2], 1e-6)
    return np.column_stack([focal_length * vertices_camera[:, 0] / z + width / 2.0, focal_length * vertices_camera[:, 1] / z + height / 2.0])


def robust_extent(vertices: np.ndarray) -> float:
    if len(vertices) < 20:
        return 0.0
    extent = float(np.max(np.percentile(vertices, 95.0, axis=0) - np.percentile(vertices, 5.0, axis=0)))
    return extent if np.isfinite(extent) and extent > 1e-8 else 0.0


def mesh_object_dirs(mesh_root: Path) -> list[Path]:
    return sorted((p for p in mesh_root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))


def mesh_frames(mesh_root: Path) -> list[int]:
    return sorted({int(path.stem) for obj_dir in mesh_object_dirs(mesh_root) for path in obj_dir.glob("*.ply")})


def body_extent_targets(mesh_root: Path, fixed_field_player_size: bool, goalie_ids: set[int], goalie_extent_threshold: float) -> dict[int, float]:
    """Body size each track is resized to: its own median, or the shared field-player median."""
    track_extents = {}
    for obj_dir in mesh_object_dirs(mesh_root):
        extents = [robust_extent(ply_to_camera(np.asarray(trimesh.load(path, process=False).vertices, dtype=np.float64))) for path in sorted(obj_dir.glob("*.ply"))]
        extents = [e for e in extents if e > 0]
        if extents:
            track_extents[int(obj_dir.name)] = float(np.median(extents))
    if not fixed_field_player_size or not track_extents:
        return track_extents
    field_extent = float(np.median(list(track_extents.values())))
    return {obj_id: extent if obj_id in goalie_ids or extent >= field_extent * goalie_extent_threshold else field_extent
            for obj_id, extent in track_extents.items()}


def resize_to_extent(vertices: np.ndarray, target_extent: float | None) -> np.ndarray:
    current = robust_extent(vertices)
    if not target_extent or current <= 0:
        return vertices
    ratio = float(np.clip(target_extent / current, 0.72, 1.38))
    center = np.median(vertices, axis=0)
    return center + ratio * (vertices - center)


# ---------------------------------------------------------------------------
# VGGT-Omega scene in the 518x518 layout
# ---------------------------------------------------------------------------


def layout_pixels(points_xy: np.ndarray, image_shape: tuple[int, int], resolution: int) -> np.ndarray:
    """Original image pixels -> pixels of the letterboxed square layout."""
    height, width = image_shape
    side = max(width, height)
    scale = resolution / side
    return np.column_stack([(points_xy[:, 0] + (side - width) / 2.0) * scale, (points_xy[:, 1] + (side - height) / 2.0) * scale])


def load_layout(npz_path: Path, vggt_repo: Path) -> dict:
    sys.path.insert(0, str(vggt_repo.resolve()))
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    with np.load(npz_path) as data:
        layout = {key: data[key] for key in ("frame_indices", "extrinsic", "intrinsic", "depth_map", "depth_conf")}
    layout["frame_indices"] = layout["frame_indices"].astype(int)
    layout["world_points"] = unproject_depth_map_to_point_map(layout["depth_map"], layout["extrinsic"], layout["intrinsic"])
    return layout


def world_to_camera(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    return points @ extrinsic[:, :3].T + extrinsic[:, 3]


def camera_to_world(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    return (points - extrinsic[:, 3]) @ extrinsic[:, :3]


def fit_scale_offset(source: np.ndarray, target: np.ndarray, rounds: int = 4) -> tuple[float, np.ndarray, np.ndarray] | None:
    """target ~ scale * source + offset, refitted after dropping the worst 30% of points each round."""
    keep = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    src, tgt = source[keep], target[keep]
    for round_idx in range(rounds + 1):
        if len(src) < 12:
            return None
        src_c, tgt_c = src - src.mean(axis=0), tgt - tgt.mean(axis=0)
        denom = float(np.sum(src_c * src_c))
        if denom < 1e-10:
            return None
        scale = float(np.sum(src_c * tgt_c) / denom)
        if not np.isfinite(scale) or scale <= 0:
            return None
        if round_idx == rounds:
            break
        err = np.linalg.norm(scale * src_c - tgt_c, axis=1)
        inliers = err <= max(0.01, float(np.percentile(err, 70)))
        if inliers.sum() < 12 or inliers.all():
            break
        src, tgt = src[inliers], tgt[inliers]
    return scale, src.mean(axis=0), tgt.mean(axis=0)


def fit_mesh(mesh_path: Path, focal_path: Path, image_shape: tuple[int, int], world_points: np.ndarray, confidence: np.ndarray,
             extrinsic: np.ndarray, min_conf: float, max_fit_vertices: int, target_extent: float | None) -> dict | None:
    """A mesh's camera-frame vertices and the scale/offset that put it on the VGGT-Omega depth."""
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) < 20:
        return None
    focal_length = float(json.loads(focal_path.read_text())["focal_length"])
    camera = ply_to_camera(vertices)
    uv = project_to_image(camera, focal_length, image_shape)  # projection of the mesh as SAM-Body4D fitted it
    camera = resize_to_extent(camera, target_extent)
    inside = np.flatnonzero(np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < image_shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image_shape[0]))
    if len(inside) < 20:
        return None
    if len(inside) > max_fit_vertices:
        inside = inside[np.linspace(0, len(inside) - 1, max_fit_vertices).round().astype(int)]
    resolution = world_points.shape[0]
    px = np.clip(np.round(layout_pixels(uv[inside], image_shape, resolution)).astype(int), 0, resolution - 1)
    target, conf = world_points[px[:, 1], px[:, 0]].astype(np.float64), confidence[px[:, 1], px[:, 0]]
    valid = np.isfinite(target).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    fit = fit_scale_offset(camera[inside[valid]], world_to_camera(target[valid], extrinsic))
    if fit is None:
        return None
    scale, source_mean, target_mean = fit
    return {"camera_vertices": camera, "faces": np.asarray(mesh.faces, dtype=np.int32), "scale": scale, "source_mean": source_mean,
            "target_mean": target_mean, "fit_points": int(valid.sum())}


def place_meshes(fits: list[dict], extrinsic: np.ndarray) -> list[np.ndarray]:
    """World vertices of one frame's meshes, with each scale clamped around the frame's median."""
    if not fits:
        return []
    median = float(np.median([f["scale"] for f in fits]))
    placed = []
    for f in fits:
        f["placed_scale"] = float(np.clip(f["scale"], SCALE_CLAMP[0] * median, SCALE_CLAMP[1] * median))
        camera = f["placed_scale"] * (f["camera_vertices"] - f["source_mean"]) + f["target_mean"]
        placed.append(camera_to_world(camera, extrinsic))
    return placed


def frame_meshes(layout: dict, local_idx: int, frame_idx: int, image_shape: tuple[int, int], mesh_root: Path, focal_root: Path,
                 extents: dict[int, float], min_conf_percentile: float, max_fit_vertices: int) -> list[tuple[int, dict, np.ndarray]]:
    """(object id, fit, world vertices) for every mesh of one frame."""
    conf = layout["depth_conf"][local_idx]
    min_conf = float(np.percentile(conf.reshape(-1), min_conf_percentile))
    fits = []
    for obj_dir in mesh_object_dirs(mesh_root):
        mesh_path, focal_path = obj_dir / f"{frame_idx:08d}.ply", focal_root / obj_dir.name / f"{frame_idx:08d}.json"
        if mesh_path.exists() and focal_path.exists():
            fit = fit_mesh(mesh_path, focal_path, image_shape, layout["world_points"][local_idx], conf, layout["extrinsic"][local_idx],
                           min_conf, max_fit_vertices, extents.get(int(obj_dir.name)))
            if fit is not None:
                fits.append((int(obj_dir.name), fit))
    placed = place_meshes([f for _, f in fits], layout["extrinsic"][local_idx])
    return [(obj_id, fit, world) for (obj_id, fit), world in zip(fits, placed)]


def world_to_field(points: np.ndarray, layout_fit: dict) -> np.ndarray:
    plane = layout_fit["floor_plane"]
    rel = points - np.asarray(plane["center"])
    uv = np.column_stack([rel @ np.asarray(plane["basis_u"]), rel @ np.asarray(plane["basis_v"]), np.ones(len(rel))])
    xyw = uv @ np.asarray(layout_fit["plane_to_floor_homography"]).T
    return xyw[:, :2] / xyw[:, 2:]


# ---------------------------------------------------------------------------
# Top-down video
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout-dir", required=True, help="Output folder of export_vggt_omega_vggt_layout.py.")
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--teams-json", required=True, help="teams.json from classify_sam3_teams_by_torso_color.py (the SAM3 IDs the meshes were made from).")
    parser.add_argument("--mesh-root", required=True, help="mesh_4d_individual from run_sam_body4d_from_sam3_boxes.py")
    parser.add_argument("--focal-root", required=True, help="focal_4d_individual from run_sam_body4d_from_sam3_boxes.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-conf-percentile", type=float, default=20.0, help="Depth points below this confidence percentile of the frame are not fitted.")
    parser.add_argument("--max-fit-vertices", type=int, default=1200)
    parser.add_argument("--max-draw-vertices", type=int, default=900)
    parser.add_argument("--fixed-field-player-size", action="store_true")
    parser.add_argument("--goalie-object-ids", default="", help="Comma-separated SAM3 IDs that keep their own body size.")
    parser.add_argument("--goalie-extent-threshold", type=float, default=1.10)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--team-names", default="", help='Legend names, e.g. "black=Oshawa,white=Toronto".')
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout_dir, mesh_root, focal_root = Path(args.layout_dir), Path(args.mesh_root), Path(args.focal_root)
    layout_fit = json.loads((layout_dir / "field_fit_vggt_layout.json").read_text())
    dims = fm.FieldDims(**json.loads(Path(layout_fit["field_plane_fit"]).read_text())["field_dims"])
    layout = load_layout(layout_dir / "vggt_predictions_compact.npz", Path(args.vggt_repo))
    teams = {r["frame"]: dict(zip(r["object_ids"], r["teams"])) for r in json.loads(Path(args.teams_json).read_text())["frames"]}
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    image_shape = cv2.imread(str(frame_paths[0])).shape[:2]
    goalie_ids = {int(v) for v in args.goalie_object_ids.split(",") if v.strip()}
    extents = body_extent_targets(mesh_root, args.fixed_field_player_size, goalie_ids, args.goalie_extent_threshold)
    available = set(mesh_frames(mesh_root))

    field_h = args.height - HEADER_PX - FOOTER_PX
    base, to_px = field_canvas(dims, args.width, field_h, margin=36)
    footer = legend(args.width, dict(item.split("=", 1) for item in args.team_names.split(",") if "=" in item), floor_in_view=False)
    video, records = [], []
    for local_idx, frame_idx in enumerate(layout["frame_indices"].tolist()):
        canvas = base.copy()
        players = []
        if frame_idx in available:
            for obj_id, fit, world in frame_meshes(layout, local_idx, frame_idx, image_shape, mesh_root, focal_root, extents,
                                                   args.min_conf_percentile, args.max_fit_vertices):
                team = teams.get(frame_idx, {}).get(obj_id, "unknown")
                if team == "ignore":
                    continue
                draw = world[np.linspace(0, len(world) - 1, min(len(world), args.max_draw_vertices)).round().astype(int)]
                xy = world_to_field(draw, layout_fit)
                fill, outline = TEAM_STYLE_BGR.get(team, TEAM_STYLE_BGR["unknown"])
                px = np.round(to_px(xy)).astype(int)
                for p in px:
                    cv2.circle(canvas, tuple(p), 3, outline, -1, cv2.LINE_AA)
                for p in px:
                    cv2.circle(canvas, tuple(p), 2, fill, -1, cv2.LINE_AA)
                median = np.median(xy, axis=0)
                players.append({"object_id": obj_id, "team": team, "median_field_point_ft": median.tolist(), "fit_points": fit["fit_points"],
                                "scale": fit["scale"], "placed_scale": fit["placed_scale"]})
        top = np.full((HEADER_PX, args.width, 3), 18, np.uint8)
        header = f"{frame_paths[frame_idx].name}   t = {frame_idx / args.fps:.1f} s   {len(players)} SAM-Body4D meshes"
        cv2.putText(top, header, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
        video.append(cv2.cvtColor(np.vstack([top, canvas, footer]), cv2.COLOR_BGR2RGB))
        records.append({"frame": frame_idx, "frame_name": frame_paths[frame_idx].name, "players": players})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_h264_video(video, out / "meshes_on_field_h264.mp4", args.fps)
    (out / "meshes_on_field.json").write_text(json.dumps({
        "layout_dir": args.layout_dir, "mesh_root": args.mesh_root, "teams_json": args.teams_json, "field_dims": dims.to_dict(),
        "fixed_field_player_size": args.fixed_field_player_size, "frames": records,
    }) + "\n")
    print(json.dumps({"video": str(out / "meshes_on_field_h264.mp4"), "frames": len(records), "meshes": sum(len(r["players"]) for r in records)}))


if __name__ == "__main__":
    main()
