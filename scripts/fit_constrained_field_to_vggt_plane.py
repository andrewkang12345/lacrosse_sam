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
from refit_vggt_field_with_sam2_faceoff_masks import (  # noqa: E402
    FACEOFF_OBJECTS,
    affine_lstsq,
    collect_faceoff_points,
    collect_feature_uv,
    initial_transform_params,
    load_npz_mask,
    params_to_transform,
    sample_mask_world_points,
)
from render_birds_eye_locations import dense_floor_model  # noqa: E402
from run_vggt_birds_eye import FLOOR_FEATURE_OBJECTS, fit_plane, project_to_plane_coords, transform_plane_to_floor  # noqa: E402


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homog @ H.T
    denom = np.where(np.abs(out[:, 2]) < 1e-9, np.nan, out[:, 2])
    return out[:, :2] / denom[:, None]


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    return apply_homography(H, points)


def collect_plane_from_floor_masks(
    frame_indices: np.ndarray,
    frames_dir: Path,
    floor_mask_dir: Path,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    min_conf: float,
    max_points_per_feature: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    floor_points = []
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame = cv2.imread(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        shape = frame.shape[:2]
        for feature, object_id in FLOOR_FEATURE_OBJECTS.items():
            if feature == "field_outline":
                continue
            mask = load_npz_mask(floor_mask_dir, int(frame_idx), object_id, shape)
            pts = sample_mask_world_points(
                mask,
                world_points[local_idx],
                depth_conf[local_idx],
                shape,
                min_conf,
                max_points_per_feature,
                seed=int(frame_idx) * 1009 + object_id,
            )
            if len(pts):
                floor_points.append(pts)
    if not floor_points:
        raise RuntimeError("No floor points found from SAM2 feature masks.")
    merged = np.concatenate(floor_points, axis=0)
    center, basis_u, basis_v, inliers = fit_plane(merged)
    return center, basis_u, basis_v, int(len(merged)), int(np.asarray(inliers).sum())


def initial_floor_to_plane_params(model: str, old_H_plane_to_floor: np.ndarray) -> np.ndarray:
    dense_floor = dense_floor_model(step_ft=4.0).astype(np.float64)
    old_floor_to_plane = np.linalg.inv(old_H_plane_to_floor)
    old_plane = transform_points(old_floor_to_plane, dense_floor)
    if model == "homography":
        old_floor_to_plane = old_floor_to_plane / old_floor_to_plane[2, 2]
        return np.asarray(
            [
                old_floor_to_plane[0, 0],
                old_floor_to_plane[0, 1],
                old_floor_to_plane[0, 2],
                old_floor_to_plane[1, 0],
                old_floor_to_plane[1, 1],
                old_floor_to_plane[1, 2],
                old_floor_to_plane[2, 0],
                old_floor_to_plane[2, 1],
            ],
            dtype=np.float64,
        )
    return initial_transform_params(model, dense_floor, old_plane, old_floor_to_plane)


def fit_field_to_plane(
    feature_uv: dict[str, np.ndarray],
    point_uv: np.ndarray,
    point_targets: np.ndarray,
    old_H_plane_to_floor: np.ndarray,
    model: str,
    point_weight: float,
    prior_weight: float,
    loss_f_scale: float,
) -> tuple[np.ndarray, dict]:
    samples_by_feature = feature_samples()
    initial_params = initial_floor_to_plane_params(model, old_H_plane_to_floor)
    dense_floor = dense_floor_model(step_ft=4.0).astype(np.float64)
    old_floor_to_plane = np.linalg.inv(old_H_plane_to_floor)
    old_dense_plane = transform_points(old_floor_to_plane, dense_floor)

    def residuals(params: np.ndarray) -> np.ndarray:
        H_floor_to_plane = params_to_transform(model, params)
        values: list[float] = []
        for feature, observed_uv in feature_uv.items():
            samples = samples_by_feature.get(feature)
            if samples is None or len(observed_uv) == 0:
                continue
            model_uv = transform_points(H_floor_to_plane, samples)
            valid = np.isfinite(model_uv).all(axis=1)
            if not valid.any():
                values.extend([1e3] * min(len(observed_uv), 50))
                continue
            tree = cKDTree(model_uv[valid])
            distances, _ = tree.query(observed_uv, k=1)
            if feature == "field_outline":
                distances = distances * 0.7
            elif feature == "goal_crease":
                distances = distances * 0.9
            values.extend(np.minimum(distances, np.percentile(distances, 90)).tolist())
        if len(point_uv):
            pred_points = transform_points(H_floor_to_plane, point_targets)
            valid = np.isfinite(pred_points).all(axis=1)
            if valid.any():
                values.extend(((pred_points[valid] - point_uv[valid]) * point_weight).reshape(-1).tolist())
            if (~valid).any():
                values.extend([1e3] * int((~valid).sum()) * 2)
        if prior_weight > 0:
            dense_plane = transform_points(H_floor_to_plane, dense_floor)
            valid = np.isfinite(dense_plane).all(axis=1) & np.isfinite(old_dense_plane).all(axis=1)
            if valid.any():
                values.extend(((dense_plane[valid] - old_dense_plane[valid]) * prior_weight).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residuals,
        initial_params,
        loss="soft_l1",
        f_scale=loss_f_scale,
        max_nfev=1200,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    H_floor_to_plane = params_to_transform(model, result.x)
    H_plane_to_floor = np.linalg.inv(H_floor_to_plane)
    H_plane_to_floor = H_plane_to_floor / H_plane_to_floor[2, 2]

    point_errors = np.linalg.norm(transform_plane_to_floor(H_plane_to_floor, point_uv) - point_targets, axis=1) if len(point_uv) else np.zeros(0)
    old_point_errors = np.linalg.norm(transform_plane_to_floor(old_H_plane_to_floor, point_uv) - point_targets, axis=1) if len(point_uv) else np.zeros(0)
    old_dense_floor = transform_plane_to_floor(old_H_plane_to_floor, old_dense_plane)
    new_dense_floor = transform_plane_to_floor(H_plane_to_floor, old_dense_plane)
    drift = np.linalg.norm(new_dense_floor - old_dense_floor, axis=1)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "transform_model": model,
        "transform_matrix_type": "projective_homography" if model == "homography" else "affine_3x3",
        "point_constraints": int(len(point_uv)),
        "old_mean_point_error_ft": float(np.mean(old_point_errors)) if len(old_point_errors) else None,
        "new_mean_point_error_ft": float(np.mean(point_errors)) if len(point_errors) else None,
        "old_max_point_error_ft": float(np.max(old_point_errors)) if len(old_point_errors) else None,
        "new_max_point_error_ft": float(np.max(point_errors)) if len(point_errors) else None,
        "mean_prior_drift_ft": float(np.mean(drift)),
        "max_prior_drift_ft": float(np.max(drift)),
        "feature_points": {feature: int(len(points)) for feature, points in feature_uv.items()},
        "point_weight": float(point_weight),
        "prior_weight": float(prior_weight),
        "loss_f_scale": float(loss_f_scale),
    }
    return H_plane_to_floor, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a non-projective synthetic field transform directly into the VGGT floor plane.")
    parser.add_argument("--vggt-npz", default="outputs/vggt/birds_eye_undistorted/vggt_predictions_compact.npz")
    parser.add_argument("--base-fit-json", default="outputs/vggt/birds_eye_undistorted/birds_eye_player_locations_vggt.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps_undistorted")
    parser.add_argument("--floor-mask-dir", default="outputs/sam2/floor_features/sam2_floor_feature_instance_masks_with_outline_undistorted")
    parser.add_argument("--faceoff-mask-dir", default="outputs/sam2/floor_features/sam2_faceoff_corner_instance_masks_undistorted")
    parser.add_argument("--output", default="outputs/vggt/birds_eye_undistorted/birds_eye_player_locations_vggt_field_to_plane_similarity.json")
    parser.add_argument("--transform-model", choices=["similarity", "anisotropic", "affine", "homography"], default="similarity")
    parser.add_argument("--min-depth-conf", type=float, default=3.0)
    parser.add_argument("--max-feature-points", type=int, default=320)
    parser.add_argument("--max-faceoff-points", type=int, default=160)
    parser.add_argument("--min-faceoff-area", type=int, default=8)
    parser.add_argument("--max-faceoff-area", type=int, default=10000)
    parser.add_argument("--point-weight", type=float, default=30.0)
    parser.add_argument("--prior-weight", type=float, default=0.02)
    parser.add_argument("--loss-f-scale", type=float, default=0.25)
    args = parser.parse_args()

    base = json.loads(Path(args.base_fit_json).read_text())
    with np.load(args.vggt_npz) as data:
        frame_indices = data["frame_indices"].astype(int) if "frame_indices" in data else np.arange(data["depth_map"].shape[0])
        extrinsic = data["extrinsic"].astype(np.float32)
        intrinsic = data["intrinsic"].astype(np.float32)
        depth = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    center, basis_u, basis_v, floor_point_count, plane_inliers = collect_plane_from_floor_masks(
        frame_indices,
        Path(args.frames_dir),
        Path(args.floor_mask_dir),
        world_points,
        depth_conf,
        args.min_depth_conf,
        args.max_feature_points,
    )
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
    H, metrics = fit_field_to_plane(
        feature_uv,
        point_uv,
        point_targets,
        np.asarray(base["plane_to_floor_homography"], dtype=np.float64),
        args.transform_model,
        args.point_weight,
        args.prior_weight,
        args.loss_f_scale,
    )

    output = dict(base)
    output["floor_plane"] = {
        "center": center.tolist(),
        "basis_u": basis_u.tolist(),
        "basis_v": basis_v.tolist(),
        "floor_points": int(floor_point_count),
        "plane_inliers": int(plane_inliers),
    }
    output["plane_to_floor_homography"] = H.tolist()
    output["constrained_field_fit"] = {
        "schema": "vggt_constrained_field_fit_v1",
        "source_base_fit_json": args.base_fit_json,
        "floor_mask_dir": args.floor_mask_dir,
        "faceoff_mask_dir": args.faceoff_mask_dir,
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
