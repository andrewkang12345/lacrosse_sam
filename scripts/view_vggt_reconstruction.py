from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import trimesh
import viser
import viser.transforms as viser_tf
from PIL import Image
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = REPO_ROOT / "third_party" / "VGGT"
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map  # noqa: E402
from run_vggt_birds_eye import original_pixels_to_vggt  # noqa: E402
from nll_field_geometry import circle_points, goal_crease_segments, line_points  # noqa: E402
from render_birds_eye_locations import FLOOR_LENGTH_FT, FLOOR_WIDTH_FT, rounded_floor_points  # noqa: E402
from render_sam_body4d_vggt_birds_eye import (  # noqa: E402
    project_sam_body4d_mesh_to_image,
    sam_body4d_ply_to_camera,
)


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def scale_intrinsics_for_stride(intrinsic: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return intrinsic
    scaled = intrinsic.copy()
    scaled[:, 0, :] /= float(stride)
    scaled[:, 1, :] /= float(stride)
    return scaled


def fit_scale_translation(
    source: np.ndarray,
    target: np.ndarray,
    rounds: int = 4,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    keep = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    if keep.sum() < 12:
        return None
    src = source[keep]
    tgt = target[keep]
    for _ in range(rounds):
        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean
        denom = float(np.sum(src_centered * src_centered))
        if denom < 1e-10:
            return None
        scale = float(np.sum(src_centered * tgt_centered) / denom)
        if not np.isfinite(scale) or scale <= 0:
            return None
        translation = tgt_mean - scale * src_mean
        pred = scale * src + translation[None, :]
        err = np.linalg.norm(pred - tgt, axis=1)
        cutoff = max(0.01, float(np.percentile(err, 70)))
        next_keep = err <= cutoff
        if next_keep.sum() < 12 or next_keep.sum() == len(src):
            break
        src = src[next_keep]
        tgt = tgt[next_keep]
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    denom = float(np.sum(src_centered * src_centered))
    if denom < 1e-10:
        return None
    scale = float(np.sum(src_centered * tgt_centered) / denom)
    if not np.isfinite(scale) or scale <= 0:
        return None
    translation = tgt_mean - scale * src_mean
    return scale, translation, src, tgt


def world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return points_world @ rotation.T + translation[None, :]


def camera_to_world(points_camera: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return (points_camera - translation[None, :]) @ rotation


def sample_vggt_points_at_pixels(
    world_points: np.ndarray,
    confidence: np.ndarray,
    pixels_xy: np.ndarray,
    image_shape: tuple[int, int],
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = world_points.shape[0]
    mapped = original_pixels_to_vggt(pixels_xy, image_shape, resolution)
    xs = np.clip(np.round(mapped[:, 0]).astype(np.int32), 0, resolution - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(np.int32), 0, resolution - 1)
    target = world_points[ys, xs].astype(np.float64)
    conf = confidence[ys, xs].astype(np.float64)
    valid = np.isfinite(target).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    return target[valid], conf[valid], np.nonzero(valid)[0]


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= max_faces:
        return vertices, faces
    face_idx = np.linspace(0, len(faces) - 1, max_faces).round().astype(np.int64)
    selected_faces = faces[face_idx]
    used_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    return vertices[used_vertices], inverse.reshape(-1, 3).astype(np.int32)


def robust_camera_extent(vertices_camera: np.ndarray) -> float:
    if len(vertices_camera) < 20:
        return 0.0
    lo = np.percentile(vertices_camera, 5.0, axis=0)
    hi = np.percentile(vertices_camera, 95.0, axis=0)
    extent = float(np.max(hi - lo))
    return extent if np.isfinite(extent) and extent > 1e-8 else 0.0


def compute_track_body_extents(mesh_root: Path) -> dict[str, float]:
    extents: dict[str, list[float]] = {}
    if not mesh_root.exists():
        return {}
    for obj_dir in sorted(mesh_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not obj_dir.is_dir():
            continue
        values = []
        for mesh_path in sorted(obj_dir.glob("*.ply")):
            try:
                mesh = trimesh.load(mesh_path, process=False)
            except Exception:
                continue
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if len(vertices) == 0:
                continue
            values.append(robust_camera_extent(sam_body4d_ply_to_camera(vertices)))
        values = [value for value in values if value > 0]
        if values:
            extents[obj_dir.name] = values
    return {obj_id: float(np.median(values)) for obj_id, values in extents.items()}


def compute_body_extent_targets(
    track_extents: dict[str, float],
    fixed_field_player_size: bool,
    goalie_ids: set[str],
    goalie_extent_threshold: float,
) -> dict[str, float]:
    if not fixed_field_player_size or not track_extents:
        return track_extents
    values = np.asarray([value for value in track_extents.values() if np.isfinite(value) and value > 0], dtype=np.float64)
    if len(values) == 0:
        return track_extents
    field_extent = float(np.median(values))
    threshold = field_extent * goalie_extent_threshold
    targets = {}
    exempt = set(goalie_ids)
    for obj_id, extent in track_extents.items():
        if obj_id in goalie_ids or extent >= threshold:
            targets[obj_id] = extent
            exempt.add(obj_id)
        else:
            targets[obj_id] = field_extent
    print(
        f"SAM-Body4D fixed field-player extent={field_extent:.4f}; "
        f"goalie/outlier exemptions={len(exempt)}; threshold={threshold:.4f}"
    )
    if exempt:
        print(f"Goalie/outlier extent IDs: {', '.join(sorted(exempt, key=lambda v: int(v) if v.isdigit() else v)[:30])}")
    return targets


def normalize_camera_vertices_to_track_extent(
    vertices_camera: np.ndarray,
    target_extent: float | None,
) -> tuple[np.ndarray, float]:
    if target_extent is None or target_extent <= 0:
        return vertices_camera, 1.0
    current_extent = robust_camera_extent(vertices_camera)
    if current_extent <= 0:
        return vertices_camera, 1.0
    ratio = float(np.clip(target_extent / current_extent, 0.72, 1.38))
    center = np.median(vertices_camera, axis=0)
    return (center[None, :] + ratio * (vertices_camera - center[None, :])).astype(np.float64), ratio


def vivid_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.asarray(color, dtype=np.float64)
    mean = float(arr.mean())
    arr = mean + 1.45 * (arr - mean)
    arr = np.clip(arr + 35.0, 0, 255)
    if arr.max() < 180:
        arr *= 180.0 / max(1.0, float(arr.max()))
    return tuple(int(v) for v in np.clip(arr, 0, 255))


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homog @ H.T
    denom = np.where(np.abs(out[:, 2]) < 1e-9, np.nan, out[:, 2])
    return out[:, :2] / denom[:, None]


def floor_to_vggt_world(
    floor_xy: np.ndarray,
    floor_fit: dict,
    scene_center: np.ndarray,
    offset: float = 0.012,
) -> np.ndarray:
    plane = floor_fit["floor_plane"]
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    normal = np.cross(basis_u, basis_v)
    normal = normal / max(1e-9, float(np.linalg.norm(normal)))
    H_plane_to_floor = np.asarray(floor_fit["plane_to_floor_homography"], dtype=np.float64)
    H_floor_to_plane = np.linalg.inv(H_plane_to_floor)
    plane_uv = apply_homography(H_floor_to_plane, floor_xy)
    world = center[None, :] + plane_uv[:, 0:1] * basis_u[None, :] + plane_uv[:, 1:2] * basis_v[None, :]
    return (world + offset * normal[None, :] - scene_center[None, :]).astype(np.float32)


def field_grounding_plane(floor_fit_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not floor_fit_path.exists():
        return None
    floor_fit = json.loads(floor_fit_path.read_text())
    plane = floor_fit.get("floor_plane")
    if not plane:
        return None
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    normal = np.cross(basis_u, basis_v)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    return center, normal / norm


def segments_from_polyline(points: np.ndarray, closed: bool = False) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 2), dtype=np.float64)
    pairs = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    if closed:
        pairs.append([points[-1], points[0]])
    return np.asarray(pairs, dtype=np.float64)


def synthetic_field_segments() -> list[tuple[str, np.ndarray, tuple[int, int, int], float]]:
    white = (245, 245, 235)
    yellow = (255, 216, 48)
    red = (235, 55, 45)
    blue = (80, 190, 255)
    segments: list[tuple[str, np.ndarray, tuple[int, int, int], float]] = []
    segments.append(("outline", segments_from_polyline(rounded_floor_points(samples_per_corner=44), closed=True), yellow, 4.0))
    for x in [57.5, 100.0, 142.5]:
        width = 5.0 if x == 100.0 else 3.5
        segments.append((f"line_{x:g}", segments_from_polyline(line_points(x, 0.0, x, FLOOR_WIDTH_FT, 90)), white, width))
    for x in [12.0, 188.0]:
        segments.append((f"goal_line_{x:g}", segments_from_polyline(line_points(x, 0.0, x, FLOOR_WIDTH_FT, 90)), (255, 180, 180), 2.0))
        segments.append((f"goal_posts_{x:g}", segments_from_polyline(line_points(x, 40.125, x, 44.875, 12)), red, 5.0))
    center_circle = circle_points(100.0, 42.5, 11.0, 160, endpoint=True)
    segments.append(("center_circle", segments_from_polyline(center_circle, closed=True), white, 3.0))
    for x in [12.0, 188.0]:
        for idx, crease in enumerate(goal_crease_segments(x, arc_samples=120, chord_samples=48)):
            segments.append((f"crease_{x:g}_{idx}", segments_from_polyline(crease), blue, 3.5))
    for x, y in [(42.5, 15.0), (42.5, 70.0), (157.5, 15.0), (157.5, 70.0), (100.0, 42.5)]:
        spot = circle_points(x, y, 1.0, 32, endpoint=True)
        segments.append((f"spot_{x:g}_{y:g}", segments_from_polyline(spot, closed=True), white, 3.0))
    return segments


def synthetic_field_surface() -> tuple[np.ndarray, np.ndarray]:
    boundary = rounded_floor_points(samples_per_corner=56).astype(np.float64)
    vertices = np.vstack([np.asarray([[FLOOR_LENGTH_FT / 2.0, FLOOR_WIDTH_FT / 2.0]], dtype=np.float64), boundary])
    faces = []
    for i in range(1, len(vertices)):
        j = 1 if i == len(vertices) - 1 else i + 1
        faces.append([0, i, j])
    return vertices, np.asarray(faces, dtype=np.int32)


def add_synthetic_field_to_scene(
    server: viser.ViserServer,
    floor_fit_path: Path,
    scene_center: np.ndarray,
) -> list:
    if not floor_fit_path.exists():
        print(f"Synthetic field disabled; missing floor fit: {floor_fit_path}")
        return []
    floor_fit = json.loads(floor_fit_path.read_text())
    handles = []
    surface_xy, surface_faces = synthetic_field_surface()
    surface_world = floor_to_vggt_world(surface_xy, floor_fit, scene_center, offset=0.006)
    handles.append(
        server.scene.add_mesh_simple(
            "synthetic_field/surface",
            vertices=surface_world,
            faces=surface_faces,
            color=(24, 110, 72),
            opacity=0.22,
            side="double",
            flat_shading=True,
        )
    )
    for name, floor_segments, color, width in synthetic_field_segments():
        flat = floor_segments.reshape(-1, 2)
        world = floor_to_vggt_world(flat, floor_fit, scene_center, offset=0.018).reshape(-1, 2, 3)
        valid = np.isfinite(world).all(axis=(1, 2))
        if valid.any():
            handles.append(
                server.scene.add_line_segments(
                    f"synthetic_field/{name}",
                    points=world[valid],
                    colors=color,
                    line_width=width,
                )
            )
    metrics = floor_fit.get("alignment_metrics", {})
    print(
        "Synthetic field loaded from "
        f"{floor_fit_path}; residual={metrics.get('mean_abs_residual_ft', 'unknown')} ft"
    )
    return handles


def prepare_sam_body4d_mesh_fit(
    mesh_path: Path,
    focal_path: Path,
    image_shape: tuple[int, int],
    world_points_frame: np.ndarray,
    confidence_frame: np.ndarray,
    extrinsic_frame: np.ndarray,
    min_conf: float,
    max_fit_vertices: int,
    target_body_extent: float | None,
) -> dict | None:
    mesh = trimesh.load(mesh_path, process=False)
    vertices_ply = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices_ply) < 20 or len(faces) == 0:
        return None
    focal_data = json.loads(focal_path.read_text())
    focal_length = float(focal_data["focal_length"])
    color = vivid_color(tuple(int(v) for v in focal_data.get("color_rgb", [90, 200, 255])[:3]))

    camera_vertices = sam_body4d_ply_to_camera(vertices_ply)
    camera_vertices, body_extent_ratio = normalize_camera_vertices_to_track_extent(camera_vertices, target_body_extent)
    uv = project_sam_body4d_mesh_to_image(vertices_ply, focal_length, image_shape)
    valid = np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < image_shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image_shape[0])
    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) < 20:
        return None
    foot_cutoff = float(np.percentile(uv[valid_indices, 1], 90.0))
    foot_indices = valid_indices[uv[valid_indices, 1] >= foot_cutoff]
    if len(foot_indices) < 8:
        foot_indices = valid_indices
    if len(valid_indices) > max_fit_vertices:
        sample = np.linspace(0, len(valid_indices) - 1, max_fit_vertices).round().astype(np.int64)
        valid_indices = valid_indices[sample]

    target_world, _, kept_local = sample_vggt_points_at_pixels(
        world_points_frame,
        confidence_frame,
        uv[valid_indices].astype(np.float32),
        image_shape,
        min_conf,
    )
    target_camera = world_to_camera(target_world, extrinsic_frame)
    source_camera = camera_vertices[valid_indices[kept_local]]
    fit = fit_scale_translation(source_camera, target_camera)
    if fit is None:
        return None
    scale, translation, source_fit, target_fit = fit
    return {
        "camera_vertices": camera_vertices,
        "faces": faces,
        "color": color,
        "foot_camera_vertices": camera_vertices[foot_indices],
        "source_fit": source_fit,
        "target_fit": target_fit,
        "raw_scale": float(scale),
        "raw_translation": translation,
        "body_extent_ratio": float(body_extent_ratio),
        "target_body_extent": float(target_body_extent) if target_body_extent is not None else 0.0,
        "fit_points": int(len(source_fit)),
        "mesh_file": str(mesh_path),
    }


def finalize_sam_body4d_mesh_fit(
    candidate: dict,
    extrinsic_frame: np.ndarray,
    scene_center: np.ndarray,
    scale: float,
    max_faces: int,
    grounding_plane: tuple[np.ndarray, np.ndarray] | None,
    ground_offset: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], dict]:
    source_mean = candidate["source_fit"].mean(axis=0)
    target_mean = candidate["target_fit"].mean(axis=0)
    translation = target_mean - scale * source_mean
    fitted_camera = scale * candidate["camera_vertices"] + translation[None, :]
    fitted_world = camera_to_world(fitted_camera, extrinsic_frame)
    ground_shift = 0.0
    normal_flipped = False
    if grounding_plane is not None:
        plane_center, plane_normal = grounding_plane
        foot_camera = scale * candidate["foot_camera_vertices"] + translation[None, :]
        foot_world = camera_to_world(foot_camera, extrinsic_frame)
        signed_all = (fitted_world - plane_center[None, :]) @ plane_normal
        signed_foot = (foot_world - plane_center[None, :]) @ plane_normal
        if np.nanmedian(signed_foot) > np.nanmedian(signed_all):
            plane_normal = -plane_normal
            signed_foot = -signed_foot
            normal_flipped = True
        foot_valid = signed_foot[np.isfinite(signed_foot)]
        if len(foot_valid):
            foot_distance = float(np.percentile(foot_valid, 5.0))
            ground_shift = float(ground_offset - foot_distance)
            fitted_world = fitted_world + ground_shift * plane_normal[None, :]
    fitted_vertices = fitted_world - scene_center[None, :]
    faces = candidate["faces"]
    fitted_vertices, faces = decimate_mesh(fitted_vertices.astype(np.float32), faces, max_faces=max_faces)
    metrics = {
        "fit_points": int(candidate["fit_points"]),
        "scale": float(scale),
        "raw_scale": float(candidate["raw_scale"]),
        "body_extent_ratio": float(candidate.get("body_extent_ratio", 1.0)),
        "target_body_extent": float(candidate.get("target_body_extent", 0.0)),
        "mesh_file": candidate["mesh_file"],
        "ground_shift": float(ground_shift),
        "ground_normal_flipped": bool(normal_flipped),
    }
    return fitted_vertices, faces, candidate["color"], metrics


def available_mesh_frames(mesh_root: Path) -> list[int]:
    frames: set[int] = set()
    if not mesh_root.exists():
        return []
    for obj_dir in mesh_root.iterdir():
        if not obj_dir.is_dir():
            continue
        for path in obj_dir.glob("*.ply"):
            try:
                frames.add(int(path.stem))
            except ValueError:
                continue
    return sorted(frames)


def viser_vggt_mesh_wrapper(
    pred_dict: dict,
    frame_indices: np.ndarray,
    original_image_shape: tuple[int, int],
    mesh_root: Path,
    focal_root: Path,
    port: int,
    init_conf_threshold: float,
    show_vggt_points: bool,
    mesh_min_conf_percentile: float,
    max_mesh_fit_vertices: int,
    max_mesh_faces: int,
    floor_fit_path: Path,
    ground_meshes_to_field: bool,
    mesh_ground_offset: float,
    defer_initial_meshes: bool,
    fixed_field_player_size: bool,
    goalie_ids: set[str],
    goalie_extent_threshold: float,
) -> viser.ViserServer:
    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    images = pred_dict["images"]
    depth_map = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]
    extrinsics_cam = pred_dict["extrinsic"]
    intrinsics_cam = pred_dict["intrinsic"]

    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
    conf = depth_conf
    colors = images.transpose(0, 2, 3, 1)
    s, h, w, _ = world_points.shape
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    frame_lookup = {int(frame_idx): local_idx for local_idx, frame_idx in enumerate(frame_indices.tolist())}

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center
    frame_point_indices = np.repeat(np.arange(s), h * w)

    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_show_points = server.gui.add_checkbox("Show VGGT Points", initial_value=show_vggt_points)
    gui_show_field = server.gui.add_checkbox("Show Synthetic Field", initial_value=True)
    gui_show_meshes = server.gui.add_checkbox("Show SAM-Body4D Meshes", initial_value=not defer_initial_meshes)
    gui_points_conf = server.gui.add_slider(
        "VGGT Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_selector = server.gui.add_dropdown(
        "Show VGGT Points from Frames", options=["All"] + [str(i) for i in frame_indices.tolist()], initial_value="All"
    )

    mesh_frames = [idx for idx in available_mesh_frames(mesh_root) if idx in frame_lookup]
    initial_mesh_frame = "50" if 50 in mesh_frames else (str(mesh_frames[0]) if mesh_frames else "None")
    gui_mesh_frame = server.gui.add_dropdown(
        "SAM-Body4D Mesh Frame",
        options=[str(i) for i in mesh_frames] if mesh_frames else ["None"],
        initial_value=initial_mesh_frame,
    )

    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 1e-5) if show_vggt_points else np.zeros_like(conf_flat, dtype=bool)
    point_cloud = server.scene.add_point_cloud(
        name="vggt_points",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )
    point_cloud.visible = show_vggt_points

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    field_handles = add_synthetic_field_to_scene(server, floor_fit_path, scene_center)
    grounding_plane = field_grounding_plane(floor_fit_path) if ground_meshes_to_field else None
    if grounding_plane is not None:
        print(f"SAM-Body4D grounding enabled on synthetic field plane; offset={mesh_ground_offset:.3f}")
    track_body_extents = compute_track_body_extents(mesh_root)
    if track_body_extents:
        print(f"SAM-Body4D stable per-track body extents enabled for {len(track_body_extents)} tracks")
    body_extent_targets = compute_body_extent_targets(
        track_body_extents,
        fixed_field_player_size=fixed_field_player_size,
        goalie_ids=goalie_ids,
        goalie_extent_threshold=goalie_extent_threshold,
    )
    mesh_handles: list[viser.MeshHandle] = []

    def visualize_frames() -> None:
        for img_id in tqdm(range(s)):
            cam2world_3x4 = cam_to_world[img_id]
            transform = viser_tf.SE3.from_matrix(cam2world_3x4)
            frame_axis = server.scene.add_frame(
                f"camera_{int(frame_indices[img_id])}",
                wxyz=transform.rotation().wxyz,
                position=transform.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)
            img = (images[img_id].transpose(1, 2, 0) * 255).astype(np.uint8)
            ih, iw = img.shape[:2]
            fy = 1.1 * ih
            fov = 2 * np.arctan2(ih / 2, fy)
            frustum = server.scene.add_camera_frustum(
                f"camera_{int(frame_indices[img_id])}/frustum",
                fov=fov,
                aspect=iw / ih,
                scale=0.05,
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum)

            @frustum.on_click
            def _(_, frame=frame_axis) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

    def update_point_cloud() -> None:
        point_cloud.visible = gui_show_points.value
        if not gui_show_points.value:
            print("VGGT point cloud hidden")
            return
        threshold_val = np.percentile(conf_flat, gui_points_conf.value)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_frame = int(gui_frame_selector.value)
            frame_mask = frame_point_indices == frame_lookup[selected_frame]
        combined = conf_mask & frame_mask
        point_cloud.points = points_centered[combined]
        point_cloud.colors = colors_flat[combined]
        print(f"VGGT threshold={threshold_val:.4f}, percent={gui_points_conf.value:.1f}, points={int(combined.sum())}")

    def clear_meshes() -> None:
        for handle in mesh_handles:
            handle.remove()
        mesh_handles.clear()

    def update_meshes() -> None:
        clear_meshes()
        if not gui_show_meshes.value or gui_mesh_frame.value == "None":
            return
        frame_idx = int(gui_mesh_frame.value)
        if frame_idx not in frame_lookup:
            return
        local_idx = frame_lookup[frame_idx]
        min_conf = float(np.percentile(conf[local_idx].reshape(-1), mesh_min_conf_percentile))
        candidates = []
        for obj_dir in sorted(mesh_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if not obj_dir.is_dir():
                continue
            mesh_path = obj_dir / f"{frame_idx:08d}.ply"
            focal_path = focal_root / obj_dir.name / f"{frame_idx:08d}.json"
            if not mesh_path.exists() or not focal_path.exists():
                continue
            candidate = prepare_sam_body4d_mesh_fit(
                mesh_path,
                focal_path,
                original_image_shape,
                world_points[local_idx],
                conf[local_idx],
                extrinsics_cam[local_idx],
                min_conf,
                max_mesh_fit_vertices,
                body_extent_targets.get(obj_dir.name),
            )
            if candidate is None:
                continue
            candidate["player_id"] = obj_dir.name
            candidates.append(candidate)

        if not candidates:
            print(f"SAM-Body4D meshes shown for frame {frame_idx}: 0")
            return

        raw_scales = np.asarray([candidate["raw_scale"] for candidate in candidates], dtype=np.float64)
        median_scale = float(np.median(raw_scales))
        min_scale = median_scale * 0.72
        max_scale = median_scale * 1.28
        added = 0
        for candidate in candidates:
            clamped_scale = float(np.clip(candidate["raw_scale"], min_scale, max_scale))
            fitted = finalize_sam_body4d_mesh_fit(
                candidate,
                extrinsics_cam[local_idx],
                scene_center,
                clamped_scale,
                max_mesh_faces,
                grounding_plane,
                mesh_ground_offset,
            )
            vertices, faces, color, metrics = fitted
            handle = server.scene.add_mesh_simple(
                name=f"sam_body4d/frame_{frame_idx}/player_{candidate['player_id']}",
                vertices=vertices,
                faces=faces,
                color=color,
                opacity=1.0,
                material="toon5",
                side="double",
                flat_shading=True,
            )
            mesh_handles.append(handle)
            added += 1
            clamp_note = "" if abs(metrics["scale"] - metrics["raw_scale"]) < 1e-8 else f" raw_scale={metrics['raw_scale']:.4f}"
            print(
                f"Added mesh frame={frame_idx} player={candidate['player_id']} "
                f"fit_points={metrics['fit_points']} scale={metrics['scale']:.4f} "
                f"body_extent_ratio={metrics['body_extent_ratio']:.3f} "
                f"ground_shift={metrics['ground_shift']:.4f} "
                f"normal_flipped={metrics['ground_normal_flipped']}{clamp_note}"
            )
        print(f"SAM-Body4D meshes shown for frame {frame_idx}: {added}; median_scale={median_scale:.4f}")

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_points.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    @gui_show_field.on_update
    def _(_) -> None:
        for handle in field_handles:
            handle.visible = gui_show_field.value

    @gui_show_meshes.on_update
    def _(_) -> None:
        update_meshes()

    @gui_mesh_frame.on_update
    def _(_) -> None:
        update_meshes()

    visualize_frames()
    if defer_initial_meshes:
        print("SAM-Body4D initial mesh fit deferred; enable the mesh checkbox or change frame in the GUI.")
    else:
        update_meshes()
    print("Starting viser server...")
    while True:
        time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open an interactive VGGT reconstruction in viser, optionally with fitted SAM-Body4D meshes.")
    parser.add_argument("--frames-dir", type=Path, default=REPO_ROOT / "data" / "frames_10fps")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "outputs" / "vggt" / "birds_eye_full" / "vggt_predictions_compact.npz",
    )
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--conf-threshold", type=float, default=20.0)
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=1,
        help="Decimate reconstructed depth/image grid for a lighter viewer. 1 keeps full VGGT resolution.",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "meshes" / "sam_body4d" / "sam_body4d_transreid_3clusters_overlay" / "mesh_4d_individual",
    )
    parser.add_argument(
        "--focal-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "meshes" / "sam_body4d" / "sam_body4d_transreid_3clusters_overlay" / "focal_4d_individual",
    )
    parser.add_argument("--mesh-min-conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-mesh-fit-vertices", type=int, default=1200)
    parser.add_argument("--max-mesh-faces", type=int, default=6000)
    parser.add_argument("--hide-vggt-points", action="store_true", help="Start with the VGGT point cloud hidden.")
    parser.add_argument("--no-ground-meshes-to-field", action="store_true", help="Do not shift SAM-Body4D meshes onto the synthetic field plane.")
    parser.add_argument("--mesh-ground-offset", type=float, default=0.018, help="Offset above the synthetic field plane after grounding, in VGGT world units.")
    parser.add_argument("--defer-initial-meshes", action="store_true", help="Start the server before fitting the first SAM-Body4D mesh frame.")
    parser.add_argument("--fixed-field-player-size", action="store_true", help="Use one shared SAM-Body4D body extent for normal field-player tracks.")
    parser.add_argument("--goalie-object-ids", default="", help="Comma-separated SAM object IDs to exempt from shared field-player body size.")
    parser.add_argument("--goalie-extent-threshold", type=float, default=1.10, help="Also exempt tracks whose median body extent exceeds this multiplier of the global median.")
    parser.add_argument(
        "--floor-fit-json",
        type=Path,
        default=REPO_ROOT / "outputs" / "vggt" / "birds_eye_full" / "birds_eye_player_locations_vggt.json",
    )
    args = parser.parse_args()

    frame_paths = sorted_frame_paths(args.frames_dir)
    if not frame_paths:
        raise SystemExit(f"No frame_*.jpg files found in {args.frames_dir}")
    if not args.predictions.exists():
        raise SystemExit(f"Missing VGGT predictions: {args.predictions}")

    with np.load(args.predictions) as data:
        frame_indices = data["frame_indices"].astype(int) if "frame_indices" in data else np.arange(len(frame_paths))
        extrinsic = data["extrinsic"].astype(np.float32)
        intrinsic = data["intrinsic"].astype(np.float32)
        depth = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)

    selected_frames = [frame_paths[i] for i in frame_indices.tolist()]
    original_shape = image_shape(selected_frames[0])
    images, _ = load_and_preprocess_images_square([str(path) for path in selected_frames], target_size=depth.shape[1])
    images = images.cpu().numpy().astype(np.float32)

    stride = max(1, int(args.spatial_stride))
    if stride > 1:
        images = images[:, :, ::stride, ::stride]
        depth = depth[:, ::stride, ::stride, :]
        depth_conf = depth_conf[:, ::stride, ::stride]
        intrinsic = scale_intrinsics_for_stride(intrinsic, stride)

    s, _, h, w = images.shape
    pred_dict = {
        "images": images,
        "world_points": np.zeros((s, h, w, 3), dtype=np.float32),
        "world_points_conf": depth_conf,
        "depth": depth,
        "depth_conf": depth_conf,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
    }

    print(f"Loaded {s} VGGT frames from {args.frames_dir}")
    print(f"Depth grid: {h}x{w}; initial confidence percentile: {args.conf_threshold}")
    print(f"Open http://127.0.0.1:{args.port} and use the viser controls to rotate/filter the reconstruction.")
    if args.mesh_root.exists() and args.focal_root.exists():
        print(f"SAM-Body4D mesh layer enabled from {args.mesh_root}")
    else:
        print("SAM-Body4D mesh layer disabled because mesh/focal roots were not found.")
    viser_vggt_mesh_wrapper(
        pred_dict,
        frame_indices=frame_indices,
        original_image_shape=original_shape,
        mesh_root=args.mesh_root,
        focal_root=args.focal_root,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        show_vggt_points=not args.hide_vggt_points,
        mesh_min_conf_percentile=args.mesh_min_conf_percentile,
        max_mesh_fit_vertices=args.max_mesh_fit_vertices,
        max_mesh_faces=args.max_mesh_faces,
        floor_fit_path=args.floor_fit_json,
        ground_meshes_to_field=not args.no_ground_meshes_to_field,
        mesh_ground_offset=args.mesh_ground_offset,
        defer_initial_meshes=args.defer_initial_meshes,
        fixed_field_player_size=args.fixed_field_player_size,
        goalie_ids={item.strip() for item in args.goalie_object_ids.split(",") if item.strip()},
        goalie_extent_threshold=args.goalie_extent_threshold,
    )


if __name__ == "__main__":
    main()
