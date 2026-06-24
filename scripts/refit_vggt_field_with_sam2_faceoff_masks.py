from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = REPO_ROOT / "third_party" / "VGGT"
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from fit_floor_homography_from_feature_clicks import feature_samples  # noqa: E402
from run_vggt_birds_eye import (  # noqa: E402
    FLOOR_FEATURE_OBJECTS,
    LINE_X_BY_FEATURE,
    fit_plane,
    load_npz_mask,
    params_to_h,
    project_to_plane_coords,
    sample_mask_pixels,
    sample_world_points,
    transform_plane_to_floor,
)
from render_birds_eye_locations import FLOOR_WIDTH_FT, dense_floor_model, rounded_floor_points  # noqa: E402


FACEOFF_OBJECTS = {
    "right_upper": {"object_id": 6, "world": [157.5, 70.0]},
    "right_lower": {"object_id": 7, "world": [157.5, 15.0]},
    "left_upper": {"object_id": 8, "world": [42.5, 70.0]},
    "left_lower": {"object_id": 9, "world": [42.5, 15.0]},
}


def h_to_params(H: np.ndarray) -> np.ndarray:
    H = H / H[2, 2]
    return np.asarray([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]], dtype=np.float64)


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homog @ H.T
    denom = np.where(np.abs(out[:, 2]) < 1e-9, np.nan, out[:, 2])
    return out[:, :2] / denom[:, None]


def affine_lstsq(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    X = np.column_stack([src, np.ones(len(src), dtype=np.float64)])
    coeff_x, *_ = np.linalg.lstsq(X, tgt[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(X, tgt[:, 1], rcond=None)
    A = np.asarray([[coeff_x[0], coeff_x[1]], [coeff_y[0], coeff_y[1]]], dtype=np.float64)
    t = np.asarray([coeff_x[2], coeff_y[2]], dtype=np.float64)
    return A, t


def initial_transform_params(model: str, source: np.ndarray, target: np.ndarray, old_H: np.ndarray) -> np.ndarray:
    if model == "homography":
        return h_to_params(old_H)
    A, t = affine_lstsq(source, target)
    if model == "affine":
        return np.asarray([A[0, 0], A[0, 1], A[1, 0], A[1, 1], t[0], t[1]], dtype=np.float64)
    if model == "similarity":
        a = 0.5 * (A[0, 0] + A[1, 1])
        b = 0.5 * (A[1, 0] - A[0, 1])
        return np.asarray([a, b, t[0], t[1]], dtype=np.float64)
    if model == "anisotropic":
        theta = float(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))
        c = np.cos(theta)
        s = np.sin(theta)
        sx = max(1e-8, float(c * A[0, 0] + s * A[1, 0]))
        sy = max(1e-8, float(-s * A[0, 1] + c * A[1, 1]))
        return np.asarray([np.log(sx), np.log(sy), theta, t[0], t[1]], dtype=np.float64)
    raise ValueError(f"Unknown transform model: {model}")


def params_to_transform(model: str, params: np.ndarray) -> np.ndarray:
    if model == "homography":
        return params_to_h(params)
    if model == "affine":
        a, b, c, d, tx, ty = params
        return np.asarray([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    if model == "similarity":
        a, b, tx, ty = params
        return np.asarray([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    if model == "anisotropic":
        log_sx, log_sy, theta, tx, ty = params
        sx = float(np.exp(log_sx))
        sy = float(np.exp(log_sy))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        return np.asarray([[sx * c, -sy * s, tx], [sx * s, sy * c, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unknown transform model: {model}")


def sample_mask_world_points(
    mask: np.ndarray,
    world_points_frame: np.ndarray,
    confidence_frame: np.ndarray,
    shape: tuple[int, int],
    min_conf: float,
    max_points: int,
    seed: int,
) -> np.ndarray:
    pixels = sample_mask_pixels(mask, max_points=max_points, seed=seed)
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts3d, _ = sample_world_points(world_points_frame, confidence_frame, pixels, shape, world_points_frame.shape[0], min_conf)
    return pts3d


def pixel_to_plane_uv(
    pixel_xy: np.ndarray,
    shape: tuple[int, int],
    resolution: int,
    intrinsic_frame: np.ndarray,
    extrinsic_frame: np.ndarray,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray | None:
    from run_vggt_birds_eye import original_pixels_to_vggt

    mapped = original_pixels_to_vggt(pixel_xy.reshape(1, 2), shape, resolution)[0]
    normal = np.cross(basis_u, basis_v)
    normal = normal / max(1e-9, float(np.linalg.norm(normal)))
    K_inv = np.linalg.inv(intrinsic_frame.astype(np.float64))
    ray_camera = K_inv @ np.asarray([mapped[0], mapped[1], 1.0], dtype=np.float64)
    rotation = extrinsic_frame[:, :3].astype(np.float64)
    translation = extrinsic_frame[:, 3].astype(np.float64)
    camera_center = -rotation.T @ translation
    ray_world = rotation.T @ ray_camera
    denom = float(normal @ ray_world)
    if abs(denom) < 1e-9:
        return None
    distance = float(normal @ (center - camera_center) / denom)
    if not np.isfinite(distance) or distance <= 0:
        return None
    point = camera_center + distance * ray_world
    return project_to_plane_coords(point.reshape(1, 3), center, basis_u, basis_v)[0]


def collect_feature_uv(
    frame_indices: np.ndarray,
    frames_dir: Path,
    floor_mask_dir: Path,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    min_conf: float,
    max_points_per_feature: int,
) -> dict[str, np.ndarray]:
    merged: dict[str, list[np.ndarray]] = {}
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame = cv2.imread(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        shape = frame.shape[:2]
        for feature, object_id in FLOOR_FEATURE_OBJECTS.items():
            mask = load_npz_mask(floor_mask_dir, int(frame_idx), object_id, shape)
            if feature == "field_outline":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 90)
                mask = mask & yellow
            pts3d = sample_mask_world_points(
                mask,
                world_points[local_idx],
                depth_conf[local_idx],
                shape,
                min_conf,
                max_points_per_feature,
                seed=int(frame_idx) * 1009 + object_id,
            )
            if len(pts3d):
                merged.setdefault(feature, []).append(project_to_plane_coords(pts3d, center, basis_u, basis_v))
    return {feature: np.concatenate(values, axis=0) for feature, values in merged.items() if values}


def collect_faceoff_points(
    frame_indices: np.ndarray,
    frames_dir: Path,
    faceoff_mask_dir: Path,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    min_conf: float,
    max_points_per_mask: int,
    min_area: int,
    max_area: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    uv_points = []
    targets = []
    metrics = []
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        shape = frame.shape[:2]
        for name, spec in FACEOFF_OBJECTS.items():
            mask = load_npz_mask(faceoff_mask_dir, int(frame_idx), int(spec["object_id"]), shape)
            area = int(mask.sum())
            record = {"frame": int(frame_idx), "landmark": name, "object_id": int(spec["object_id"]), "area": area, "used": False}
            if area < min_area or area > max_area:
                record["reason"] = "area_filter"
                metrics.append(record)
                continue
            ys, xs = np.where(mask)
            pixel = np.asarray([float(np.median(xs)), float(np.median(ys))], dtype=np.float64)
            median_uv = pixel_to_plane_uv(
                pixel,
                shape,
                world_points.shape[1],
                intrinsic[local_idx],
                extrinsic[local_idx],
                center,
                basis_u,
                basis_v,
            )
            if median_uv is None:
                record["reason"] = "ray_plane_failed"
                metrics.append(record)
                continue
            uv_points.append(median_uv)
            targets.append(spec["world"])
            record.update({"used": True, "sampled_points": int(area), "target": spec["world"], "image_centroid": pixel.tolist(), "plane_uv": median_uv.tolist(), "method": "sam2_mask_centroid_camera_ray_floor_plane"})
            metrics.append(record)
    if not uv_points:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), metrics
    return np.asarray(uv_points, dtype=np.float64), np.asarray(targets, dtype=np.float64), metrics


def fit_homography_with_points(
    feature_uv: dict[str, np.ndarray],
    point_uv: np.ndarray,
    point_targets: np.ndarray,
    old_H: np.ndarray,
    point_weight: float,
    prior_weight: float,
    transform_model: str,
) -> tuple[np.ndarray, dict]:
    outline_tree = cKDTree(rounded_floor_points(samples_per_corner=64).astype(np.float64))
    line_samples_by_feature = feature_samples()
    prior_floor = dense_floor_model(step_ft=4.0).astype(np.float64)
    prior_uv = apply_homography(np.linalg.inv(old_H), prior_floor)
    valid_prior = np.isfinite(prior_uv).all(axis=1)
    prior_uv = prior_uv[valid_prior]
    prior_old = transform_plane_to_floor(old_H, prior_uv)
    initial_params = initial_transform_params(transform_model, prior_uv, prior_old, old_H)

    def residuals(params: np.ndarray) -> np.ndarray:
        H = params_to_transform(transform_model, params)
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
            elif feature == "field_outline":
                distances, _ = outline_tree.query(floor, k=1)
                values.extend((distances * 0.7).tolist())
            else:
                samples = line_samples_by_feature.get(feature)
                if samples is not None:
                    tree = cKDTree(samples)
                    distances, _ = tree.query(floor, k=1)
                    values.extend(distances.tolist())
        if len(point_uv):
            pred = transform_plane_to_floor(H, point_uv)
            valid = np.isfinite(pred).all(axis=1)
            if valid.any():
                values.extend(((pred[valid] - point_targets[valid]) * point_weight).reshape(-1).tolist())
            if (~valid).any():
                values.extend([1e4] * int((~valid).sum()) * 2)
        if prior_weight > 0:
            pred_prior = transform_plane_to_floor(H, prior_uv)
            valid = np.isfinite(pred_prior).all(axis=1) & np.isfinite(prior_old).all(axis=1)
            if valid.any():
                values.extend(((pred_prior[valid] - prior_old[valid]) * prior_weight).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residuals,
        initial_params,
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=1200,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    H = params_to_transform(transform_model, result.x)
    point_errors = np.linalg.norm(transform_plane_to_floor(H, point_uv) - point_targets, axis=1) if len(point_uv) else np.zeros(0)
    old_point_errors = np.linalg.norm(transform_plane_to_floor(old_H, point_uv) - point_targets, axis=1) if len(point_uv) else np.zeros(0)
    prior_drift = np.linalg.norm(transform_plane_to_floor(H, prior_uv) - prior_old, axis=1)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "point_constraints": int(len(point_uv)),
        "old_mean_point_error_ft": float(np.mean(old_point_errors)) if len(old_point_errors) else None,
        "new_mean_point_error_ft": float(np.mean(point_errors)) if len(point_errors) else None,
        "old_max_point_error_ft": float(np.max(old_point_errors)) if len(old_point_errors) else None,
        "new_max_point_error_ft": float(np.max(point_errors)) if len(point_errors) else None,
        "mean_prior_drift_ft": float(np.mean(prior_drift)),
        "max_prior_drift_ft": float(np.max(prior_drift)),
        "feature_points": {feature: int(len(points)) for feature, points in feature_uv.items()},
        "point_weight": float(point_weight),
        "prior_weight": float(prior_weight),
        "transform_model": transform_model,
        "transform_matrix_type": "projective_homography" if transform_model == "homography" else "affine_3x3",
    }
    return H, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit VGGT field alignment using SAM2-tracked field masks and faceoff point masks.")
    parser.add_argument("--vggt-npz", default="outputs/vggt/birds_eye_full/vggt_predictions_compact.npz")
    parser.add_argument("--base-fit-json", default="outputs/vggt/birds_eye_full/birds_eye_player_locations_vggt.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--floor-mask-dir", default="outputs/sam2/floor_features/sam2_floor_feature_instance_masks_with_outline")
    parser.add_argument("--faceoff-mask-dir", default="outputs/sam2/floor_features/sam2_faceoff_corner_instance_masks")
    parser.add_argument("--output", default="outputs/vggt/birds_eye_full/birds_eye_player_locations_vggt_sam2_faceoff_refined.json")
    parser.add_argument("--min-depth-conf", type=float, default=3.0)
    parser.add_argument("--max-feature-points", type=int, default=320)
    parser.add_argument("--max-faceoff-points", type=int, default=160)
    parser.add_argument("--min-faceoff-area", type=int, default=8)
    parser.add_argument("--max-faceoff-area", type=int, default=10000)
    parser.add_argument("--point-weight", type=float, default=14.0)
    parser.add_argument("--prior-weight", type=float, default=0.08)
    parser.add_argument("--transform-model", choices=["similarity", "anisotropic", "affine", "homography"], default="homography")
    args = parser.parse_args()

    base = json.loads(Path(args.base_fit_json).read_text())
    with np.load(args.vggt_npz) as data:
        frame_indices = data["frame_indices"].astype(int) if "frame_indices" in data else np.arange(data["depth_map"].shape[0])
        extrinsic = data["extrinsic"].astype(np.float32)
        intrinsic = data["intrinsic"].astype(np.float32)
        depth = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    floor_points = []
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame = cv2.imread(str(Path(args.frames_dir) / f"frame_{frame_idx:05d}.jpg"), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        shape = frame.shape[:2]
        for feature, object_id in FLOOR_FEATURE_OBJECTS.items():
            if feature == "field_outline":
                continue
            mask = load_npz_mask(Path(args.floor_mask_dir), int(frame_idx), object_id, shape)
            pts = sample_mask_world_points(
                mask,
                world_points[local_idx],
                depth_conf[local_idx],
                shape,
                args.min_depth_conf,
                args.max_feature_points,
                seed=int(frame_idx) * 1009 + object_id,
            )
            if len(pts):
                floor_points.append(pts)
    if not floor_points:
        raise RuntimeError("No floor points found from existing SAM2 feature masks.")
    center, basis_u, basis_v, plane_inliers = fit_plane(np.concatenate(floor_points, axis=0))

    feature_uv = collect_feature_uv(
        frame_indices,
        Path(args.frames_dir),
        Path(args.floor_mask_dir),
        world_points,
        depth_conf,
        center,
        basis_u,
        basis_v,
        args.min_depth_conf,
        args.max_feature_points,
    )
    point_uv, point_targets, point_metrics = collect_faceoff_points(
        frame_indices,
        Path(args.frames_dir),
        Path(args.faceoff_mask_dir),
        world_points,
        depth_conf,
        intrinsic,
        extrinsic,
        center,
        basis_u,
        basis_v,
        args.min_depth_conf,
        args.max_faceoff_points,
        args.min_faceoff_area,
        args.max_faceoff_area,
    )
    old_H = np.asarray(base["plane_to_floor_homography"], dtype=np.float64)
    H, metrics = fit_homography_with_points(feature_uv, point_uv, point_targets, old_H, args.point_weight, args.prior_weight, args.transform_model)

    output = dict(base)
    output["floor_plane"] = {
        "center": center.tolist(),
        "basis_u": basis_u.tolist(),
        "basis_v": basis_v.tolist(),
        "floor_points": int(sum(len(points) for points in floor_points)),
        "plane_inliers": int(np.asarray(plane_inliers).sum()),
    }
    output["plane_to_floor_homography"] = H.tolist()
    output["sam2_faceoff_refinement"] = {
        "schema": "vggt_sam2_faceoff_refinement_v1",
        "source_base_fit_json": args.base_fit_json,
        "floor_mask_dir": args.floor_mask_dir,
        "faceoff_mask_dir": args.faceoff_mask_dir,
        "transform_model": args.transform_model,
        "metrics": metrics,
        "point_metrics": point_metrics,
    }
    output["alignment_metrics"] = dict(base.get("alignment_metrics", {}))
    output["alignment_metrics"].update(metrics)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(out_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
