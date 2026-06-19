from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from fit_floor_homography_from_feature_clicks import arc_points, feature_samples
from render_birds_eye_locations import (
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    draw_floor,
    rgb_to_bgr,
    rounded_floor_points,
    world_to_canvas,
    write_h264,
)


FLOOR_FEATURE_OBJECTS = {
    "left_restraining_line": 1,
    "right_restraining_line": 2,
    "midfield_line": 3,
    "goal_crease": 4,
    "field_outline": 5,
}
LINE_X_BY_FEATURE = {
    "left_restraining_line": 57.5,
    "right_restraining_line": 142.5,
    "midfield_line": 100.0,
}


@dataclass
class PlaneFrame:
    frame_idx: int
    features: dict[str, np.ndarray]


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_npz_mask(mask_dir: Path, frame_idx: int, object_id: int | None, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return np.zeros((height, width), dtype=bool)
    data = np.load(path)
    object_ids = [int(value) for value in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    if object_id is None:
        mask = masks.any(axis=0) if masks.size else np.zeros((height, width), dtype=bool)
    elif object_id in object_ids:
        mask = masks[object_ids.index(object_id)]
    else:
        mask = np.zeros((height, width), dtype=bool)
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def sample_mask_pixels(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def original_pixels_to_vggt(points_xy: np.ndarray, original_shape: tuple[int, int], resolution: int) -> np.ndarray:
    height, width = original_shape
    max_dim = max(width, height)
    left = (max_dim - width) / 2.0
    top = (max_dim - height) / 2.0
    scale = float(resolution) / float(max_dim)
    out = points_xy.astype(np.float64).copy()
    out[:, 0] = (out[:, 0] + left) * scale
    out[:, 1] = (out[:, 1] + top) * scale
    return out


def sample_world_points(
    world_points: np.ndarray,
    confidence: np.ndarray,
    original_points_xy: np.ndarray,
    original_shape: tuple[int, int],
    resolution: int,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(original_points_xy) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    mapped = original_pixels_to_vggt(original_points_xy, original_shape, resolution)
    xs = np.clip(np.round(mapped[:, 0]).astype(np.int32), 0, resolution - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(np.int32), 0, resolution - 1)
    pts = world_points[ys, xs].astype(np.float64)
    conf = confidence[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    return pts[valid], conf[valid]


def fit_plane(points: np.ndarray, max_iters: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 20:
        raise RuntimeError(f"Need at least 20 VGGT floor points to fit a plane, got {len(points)}.")
    keep = np.isfinite(points).all(axis=1)
    pts = points[keep]
    for _ in range(max_iters):
        center = np.median(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        normal = vh[-1]
        normal = normal / max(1e-9, np.linalg.norm(normal))
        dist = np.abs((pts - center) @ normal)
        median = float(np.median(dist))
        mad = float(np.median(np.abs(dist - median))) + 1e-9
        next_keep = dist <= median + 3.0 * 1.4826 * mad
        if next_keep.sum() < 20 or next_keep.sum() == len(pts):
            break
        pts = pts[next_keep]
    center = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    basis_u = vh[0] / np.linalg.norm(vh[0])
    normal = vh[-1] / np.linalg.norm(vh[-1])
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)
    final_dist = np.abs((points - center) @ normal)
    cutoff = float(np.percentile(final_dist[np.isfinite(final_dist)], 90))
    inlier_mask = np.isfinite(final_dist) & (final_dist <= cutoff)
    return center, basis_u, basis_v, inlier_mask


def project_to_plane_coords(points: np.ndarray, center: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray) -> np.ndarray:
    rel = points - center[None, :]
    return np.column_stack([rel @ basis_u, rel @ basis_v]).astype(np.float64)


def params_to_h(params: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [params[6], params[7], 1.0]],
        dtype=np.float64,
    )


def transform_plane_to_floor(H: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    if len(points_uv) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(points_uv.astype(np.float32).reshape(1, -1, 2), H).reshape(-1, 2).astype(np.float64)


def line_points(x: float, y1: float, y2: float, samples: int = 180) -> np.ndarray:
    return np.column_stack([np.full(samples, x), np.linspace(y1, y2, samples)]).astype(np.float64)


def outline_samples() -> np.ndarray:
    return rounded_floor_points(samples_per_corner=64).astype(np.float64)


def crease_samples() -> np.ndarray:
    left = arc_points(12.0, 42.5, 9.25, -90.0, 90.0, 160)
    right = arc_points(188.0, 42.5, 9.25, 90.0, 270.0, 160)
    return np.concatenate([left, right], axis=0).astype(np.float64)


def homography_from_bbox(uv: np.ndarray, flip_x: bool, flip_y: bool, swap: bool) -> np.ndarray:
    data = uv[:, ::-1] if swap else uv
    mn = np.percentile(data, 2, axis=0)
    mx = np.percentile(data, 98, axis=0)
    src = np.asarray([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]], dtype=np.float32)
    x0, x1 = (FLOOR_LENGTH_FT, 0.0) if flip_x else (0.0, FLOOR_LENGTH_FT)
    y0, y1 = (FLOOR_WIDTH_FT, 0.0) if flip_y else (0.0, FLOOR_WIDTH_FT)
    dst = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    if swap:
        swap_h = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H = H @ swap_h
    return H


def fit_plane_to_floor_homography(feature_uv: dict[str, np.ndarray], regularization: float) -> tuple[np.ndarray, dict]:
    all_uv = np.concatenate([uv for uv in feature_uv.values() if len(uv)], axis=0)
    if len(all_uv) < 40:
        raise RuntimeError("Not enough plane-coordinate floor feature samples to align VGGT to the rink.")
    outline_tree = cKDTree(outline_samples())
    crease_tree = cKDTree(crease_samples())
    line_samples_by_feature = feature_samples()

    def residuals(params: np.ndarray, initial_projected: np.ndarray | None = None, initial_H: np.ndarray | None = None) -> np.ndarray:
        H = params_to_h(params)
        values: list[float] = []
        for feature, uv in feature_uv.items():
            if len(uv) == 0:
                continue
            floor = transform_plane_to_floor(H, uv)
            valid = np.isfinite(floor).all(axis=1)
            if not valid.any():
                values.extend([1e3, 1e3])
                continue
            floor = floor[valid]
            if feature in LINE_X_BY_FEATURE:
                values.extend(((floor[:, 0] - LINE_X_BY_FEATURE[feature]) * 1.6).tolist())
                outside_y = np.maximum(0.0, -floor[:, 1]) + np.maximum(0.0, floor[:, 1] - FLOOR_WIDTH_FT)
                values.extend((outside_y * 0.5).tolist())
            elif feature == "goal_crease":
                distances, _ = crease_tree.query(floor, k=1)
                values.extend((distances * 0.9).tolist())
            elif feature == "field_outline":
                distances, _ = outline_tree.query(floor, k=1)
                values.extend((distances * 0.7).tolist())
            else:
                samples = line_samples_by_feature.get(feature)
                if samples is not None:
                    tree = cKDTree(samples)
                    distances, _ = tree.query(floor, k=1)
                    values.extend(distances.tolist())
        if regularization > 0 and initial_projected is not None and initial_H is not None:
            projected = transform_plane_to_floor(H, all_uv[:: max(1, len(all_uv) // 350)])
            valid = np.isfinite(projected).all(axis=1) & np.isfinite(initial_projected).all(axis=1)
            if valid.any():
                values.extend(((projected[valid] - initial_projected[valid]) * regularization).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float64)

    candidates = []
    reg_uv = all_uv[:: max(1, len(all_uv) // 350)]
    for swap in [False, True]:
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                initial_H = homography_from_bbox(all_uv, flip_x=flip_x, flip_y=flip_y, swap=swap)
                initial_projected = transform_plane_to_floor(initial_H, reg_uv)
                result = least_squares(
                    lambda p: residuals(p, initial_projected=initial_projected, initial_H=initial_H),
                    initial_H.flatten()[:8],
                    loss="soft_l1",
                    f_scale=4.0,
                    max_nfev=700,
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                err = residuals(result.x)
                candidates.append((float(np.mean(np.abs(err))) if len(err) else float("inf"), result))
    candidates.sort(key=lambda item: item[0])
    best_error, best = candidates[0]
    H = params_to_h(best.x)
    metrics = {
        "success": bool(best.success),
        "cost": float(best.cost),
        "mean_abs_residual_ft": float(best_error),
        "nfev": int(best.nfev),
        "feature_points": {feature: int(len(points)) for feature, points in feature_uv.items()},
    }
    return H, metrics


def load_vggt(repo_root: Path):
    sys.path.insert(0, str(repo_root.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    return VGGT, load_and_preprocess_images_square, pose_encoding_to_extri_intri, unproject_depth_map_to_point_map


def run_vggt_predictions(args: argparse.Namespace, frame_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    VGGT, load_images, pose_to_cameras, unproject_depth = load_vggt(Path(args.vggt_repo))
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device == "cpu":
        dtype = torch.float32

    model = VGGT.from_pretrained(args.model).to(device)
    model.eval()
    images, _ = load_images([str(path) for path in frame_paths], target_size=args.resolution)
    images = images.to(device)

    with torch.no_grad():
        autocast = torch.cuda.amp.autocast if device == "cuda" else torch.cpu.amp.autocast
        with autocast(dtype=dtype, enabled=device == "cuda"):
            if images.shape[-1] != args.resolution or images.shape[-2] != args.resolution:
                images = F.interpolate(images, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False)
            images_batched = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batched)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_to_cameras(pose_enc, images_batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batched, ps_idx)

    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_np = depth_map.squeeze(0).detach().cpu().numpy()
    conf_np = depth_conf.squeeze(0).detach().cpu().numpy()
    if conf_np.ndim == 4 and conf_np.shape[-1] == 1:
        conf_np = conf_np[..., 0]
    world_np = unproject_depth(depth_np, extrinsic_np, intrinsic_np)
    return extrinsic_np, intrinsic_np, depth_np, conf_np, world_np


def collect_floor_feature_points(
    frame_indices: list[int],
    frames: list[np.ndarray],
    floor_mask_dir: Path,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    resolution: int,
    min_conf: float,
    max_points_per_feature: int,
) -> tuple[np.ndarray, list[PlaneFrame]]:
    all_points = []
    per_frame: list[PlaneFrame] = []
    for local_idx, frame_idx in enumerate(frame_indices):
        frame = frames[local_idx]
        shape = frame.shape[:2]
        features: dict[str, np.ndarray] = {}
        for feature, object_id in FLOOR_FEATURE_OBJECTS.items():
            mask = load_npz_mask(floor_mask_dir, frame_idx, object_id, shape)
            if feature == "field_outline":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 90)
                mask = mask & yellow
            points_xy = sample_mask_pixels(mask, max_points_per_feature, seed=frame_idx * 1009 + object_id)
            pts3d, _ = sample_world_points(world_points[local_idx], depth_conf[local_idx], points_xy, shape, resolution, min_conf)
            if len(pts3d):
                features[feature] = pts3d
                if feature != "field_outline":
                    all_points.append(pts3d)
        per_frame.append(PlaneFrame(frame_idx=frame_idx, features=features))
    if not all_points:
        raise RuntimeError("No VGGT 3D points landed on the SAM2 floor feature masks.")
    return np.concatenate(all_points, axis=0), per_frame


def load_player_frames(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    return {int(frame["frame"]): frame for frame in data.get("frames", [])}, data.get("team_colors", {})


def player_floor_point_from_mask(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    ys, _ = np.where(mask)
    y_cut = np.quantile(ys, 0.90)
    bottom = mask & (np.indices(mask.shape)[0] >= y_cut)
    return sample_mask_pixels(bottom, max_points, seed)


def render_birds_eye(frames_meta: list[dict], output_video: Path, fps: float, width: int, height: int, margin: int, trail_frames: int) -> None:
    rendered = []
    trails: dict[int, list[tuple[float, float, tuple[int, int, int]]]] = {}
    for item in frames_meta:
        canvas = draw_floor(width, height, margin)
        frame_idx = int(item["frame"])
        cv2.putText(canvas, f"VGGT bird's-eye frame {frame_idx}", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
        for det in item["players"]:
            obj_id = int(det["object_id"])
            x, y = float(det["floor_xy_ft"][0]), float(det["floor_xy_ft"][1])
            color = rgb_to_bgr(det.get("team_color", [220, 220, 220]))
            trails.setdefault(obj_id, []).append((x, y, color))
            trails[obj_id] = trails[obj_id][-trail_frames:]
        for obj_id, points in trails.items():
            for age, (x, y, color) in enumerate(points):
                alpha = (age + 1) / max(1, len(points))
                muted = tuple(int(channel * (0.35 + 0.65 * alpha)) for channel in color)
                cv2.circle(canvas, world_to_canvas(x, y, width, height, margin), 4, muted, -1, cv2.LINE_AA)
        for det in item["players"]:
            obj_id = int(det["object_id"])
            x, y = float(det["floor_xy_ft"][0]), float(det["floor_xy_ft"][1])
            color = rgb_to_bgr(det.get("team_color", [220, 220, 220]))
            px, py = world_to_canvas(x, y, width, height, margin)
            cv2.circle(canvas, (px, py), 8, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 8, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(obj_id), (px + 9, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
        rendered.append(canvas)
    write_h264(rendered, output_video, fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--model", default="facebook/VGGT-1B")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--sam3-json", default="outputs/sam3/team_classification/sam3_team_transreid_3clusters_detections.json")
    parser.add_argument("--player-mask-dir", default="outputs/sam3/text/sam3_text_player_instance_masks")
    parser.add_argument("--floor-mask-dir", default="outputs/sam2/floor_features/sam2_floor_feature_instance_masks_with_outline")
    parser.add_argument("--output-dir", default="outputs/vggt/birds_eye")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-depth-conf", type=float, default=3.0)
    parser.add_argument("--max-floor-points-per-feature", type=int, default=220)
    parser.add_argument("--max-player-points", type=int, default=80)
    parser.add_argument("--alignment-regularization", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=54)
    parser.add_argument("--trail-frames", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths_all = sorted_frame_paths(Path(args.frames_dir))
    selected = list(range(0, len(frame_paths_all), max(1, args.frame_stride)))
    if args.max_frames:
        selected = selected[: args.max_frames]
    frame_paths = [frame_paths_all[idx] for idx in selected]
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frame_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("Failed to read one or more input frames.")

    extrinsic, intrinsic, depth_map, depth_conf, world_points = run_vggt_predictions(args, frame_paths)
    np.savez_compressed(
        output_dir / "vggt_predictions_compact.npz",
        frame_indices=np.asarray(selected, dtype=np.int32),
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_conf=depth_conf,
    )

    floor_points, plane_frames = collect_floor_feature_points(
        selected,
        frames,
        Path(args.floor_mask_dir),
        world_points,
        depth_conf,
        args.resolution,
        args.min_depth_conf,
        args.max_floor_points_per_feature,
    )
    center, basis_u, basis_v, plane_inliers = fit_plane(floor_points)

    feature_uv: dict[str, list[np.ndarray]] = {}
    for plane_frame in plane_frames:
        for feature, pts3d in plane_frame.features.items():
            uv = project_to_plane_coords(pts3d, center, basis_u, basis_v)
            feature_uv.setdefault(feature, []).append(uv)
    feature_uv_merged = {feature: np.concatenate(values, axis=0) for feature, values in feature_uv.items() if values}
    H_plane_to_floor, alignment_metrics = fit_plane_to_floor_homography(feature_uv_merged, args.alignment_regularization)

    player_by_frame, _ = load_player_frames(Path(args.sam3_json))
    output_frames = []
    for local_idx, frame_idx in enumerate(selected):
        frame = frames[local_idx]
        shape = frame.shape[:2]
        metadata = player_by_frame.get(frame_idx, {"object_ids": [], "boxes": [], "team_colors": [], "teams": []})
        object_ids = [int(value) for value in metadata.get("object_ids", [])]
        players = []
        for det_idx, obj_id in enumerate(object_ids):
            mask = load_npz_mask(Path(args.player_mask_dir), frame_idx, obj_id, shape)
            pixels = player_floor_point_from_mask(mask, args.max_player_points, seed=frame_idx * 1009 + obj_id)
            if len(pixels) == 0 and det_idx < len(metadata.get("boxes", [])):
                x1, y1, x2, y2 = metadata["boxes"][det_idx]
                pixels = np.asarray([[(x1 + x2) * 0.5, y2]], dtype=np.float32)
            pts3d, conf = sample_world_points(world_points[local_idx], depth_conf[local_idx], pixels, shape, args.resolution, args.min_depth_conf)
            if len(pts3d) == 0:
                continue
            uv = project_to_plane_coords(pts3d, center, basis_u, basis_v)
            floor_xy = transform_plane_to_floor(H_plane_to_floor, np.median(uv, axis=0, keepdims=True))[0]
            floor_xy[0] = np.clip(floor_xy[0], 0.0, FLOOR_LENGTH_FT)
            floor_xy[1] = np.clip(floor_xy[1], 0.0, FLOOR_WIDTH_FT)
            team_color = metadata.get("team_colors", [[220, 220, 220]] * len(object_ids))[det_idx]
            players.append(
                {
                    "object_id": obj_id,
                    "team": metadata.get("teams", ["unknown"] * len(object_ids))[det_idx],
                    "team_color": team_color,
                    "floor_xy_ft": [float(floor_xy[0]), float(floor_xy[1])],
                    "source": "vggt_depth_player_mask_bottom",
                    "sampled_points": int(len(pts3d)),
                    "median_depth_conf": float(np.median(conf)) if len(conf) else 0.0,
                }
            )
        output_frames.append({"frame": int(frame_idx), "players": players})

    output_json = output_dir / "birds_eye_player_locations_vggt.json"
    output_video = output_dir / "birds_eye_player_locations_vggt_h264.mp4"
    result = {
        "schema": "vggt_birds_eye_v1",
        "model": args.model,
        "frames_dir": args.frames_dir,
        "frame_indices": selected,
        "vggt_resolution": args.resolution,
        "floor_mask_dir": args.floor_mask_dir,
        "player_mask_dir": args.player_mask_dir,
        "sam3_json": args.sam3_json,
        "floor_plane": {
            "center": center.tolist(),
            "basis_u": basis_u.tolist(),
            "basis_v": basis_v.tolist(),
            "floor_points": int(len(floor_points)),
            "plane_inliers": int(np.asarray(plane_inliers).sum()),
        },
        "plane_to_floor_homography": H_plane_to_floor.tolist(),
        "alignment_metrics": alignment_metrics,
        "frames": output_frames,
    }
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    render_birds_eye(output_frames, output_video, args.fps / max(1, args.frame_stride), args.width, args.height, args.margin, args.trail_frames)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_video": str(output_video),
                "frames": len(output_frames),
                "players": sum(len(frame["players"]) for frame in output_frames),
                "alignment_mean_abs_residual_ft": alignment_metrics["mean_abs_residual_ft"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
