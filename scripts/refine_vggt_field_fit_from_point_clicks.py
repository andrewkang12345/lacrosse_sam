from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = REPO_ROOT / "third_party" / "VGGT"
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from run_vggt_birds_eye import original_pixels_to_vggt, params_to_h, project_to_plane_coords, transform_plane_to_floor  # noqa: E402
from render_birds_eye_locations import dense_floor_model  # noqa: E402


def h_to_params(H: np.ndarray) -> np.ndarray:
    H = H / H[2, 2]
    return np.asarray([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]], dtype=np.float64)


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homog @ H.T
    denom = np.where(np.abs(out[:, 2]) < 1e-9, np.nan, out[:, 2])
    return out[:, :2] / denom[:, None]


def apply_affine(params: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    a, b, c, d, tx, ty = params
    pts = np.asarray(points_xy, dtype=np.float64)
    return np.column_stack([a * pts[:, 0] + b * pts[:, 1] + tx, c * pts[:, 0] + d * pts[:, 1] + ty])


def sample_click_plane_uv(
    world_points_frame: np.ndarray,
    confidence_frame: np.ndarray,
    intrinsic_frame: np.ndarray,
    extrinsic_frame: np.ndarray,
    image_xy: np.ndarray,
    original_shape: tuple[int, int],
    plane_center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    radius: int,
    min_conf_percentile: float,
) -> tuple[np.ndarray | None, dict]:
    resolution = world_points_frame.shape[0]
    mapped = original_pixels_to_vggt(image_xy.reshape(1, 2), original_shape, resolution)[0]
    normal = np.cross(basis_u, basis_v)
    normal = normal / max(1e-9, float(np.linalg.norm(normal)))

    K_inv = np.linalg.inv(intrinsic_frame.astype(np.float64))
    ray_camera = K_inv @ np.asarray([mapped[0], mapped[1], 1.0], dtype=np.float64)
    rotation = extrinsic_frame[:, :3].astype(np.float64)
    translation = extrinsic_frame[:, 3].astype(np.float64)
    camera_center = -rotation.T @ translation
    ray_world = rotation.T @ ray_camera
    denom = float(normal @ ray_world)
    if abs(denom) > 1e-8:
        scale = float(normal @ (plane_center - camera_center) / denom)
        if np.isfinite(scale) and scale > 0:
            point = camera_center + scale * ray_world
            uv = project_to_plane_coords(point.reshape(1, 3), plane_center, basis_u, basis_v)[0]
            return uv, {
                "samples": 1,
                "method": "camera_ray_floor_plane_intersection",
                "mapped_vggt_xy": [float(mapped[0]), float(mapped[1])],
                "ray_scale": scale,
            }

    cx = int(np.clip(round(mapped[0]), 0, resolution - 1))
    cy = int(np.clip(round(mapped[1]), 0, resolution - 1))
    xs = np.arange(max(0, cx - radius), min(resolution, cx + radius + 1), dtype=np.int32)
    ys = np.arange(max(0, cy - radius), min(resolution, cy + radius + 1), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = world_points_frame[grid_y.reshape(-1), grid_x.reshape(-1)].astype(np.float64)
    conf = confidence_frame[grid_y.reshape(-1), grid_x.reshape(-1)].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    if not valid.any():
        return None, {"samples": 0, "reason": "no_finite_vggt_points"}
    frame_conf_cut = float(np.percentile(confidence_frame.reshape(-1), min_conf_percentile))
    valid &= conf >= frame_conf_cut
    if not valid.any():
        return None, {"samples": 0, "reason": "below_confidence_cutoff", "conf_cutoff": frame_conf_cut}
    pts = pts[valid]
    conf = conf[valid]
    distances = np.abs((pts - plane_center[None, :]) @ normal)
    keep_count = max(3, min(12, len(pts)))
    keep = np.argsort(distances)[:keep_count]
    selected = pts[keep]
    uv = project_to_plane_coords(selected, plane_center, basis_u, basis_v)
    return np.median(uv, axis=0), {
        "samples": int(len(selected)),
        "method": "depth_window_floor_nearest_fallback",
        "conf_cutoff": frame_conf_cut,
        "mean_plane_distance": float(np.mean(distances[keep])),
        "mean_confidence": float(np.mean(conf[keep])),
        "mapped_vggt_xy": [float(mapped[0]), float(mapped[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine VGGT plane-to-field fit using exact clicked field points.")
    parser.add_argument("--vggt-npz", default="outputs/vggt/birds_eye_full/vggt_predictions_compact.npz")
    parser.add_argument("--base-fit-json", default="outputs/vggt/birds_eye_full/birds_eye_player_locations_vggt.json")
    parser.add_argument("--clicks", default="outputs/floor/clicks/floor_faceoff_corner_clicks.json")
    parser.add_argument("--output", default="outputs/vggt/birds_eye_full/birds_eye_player_locations_vggt_faceoff_refined.json")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--click-window-radius", type=int, default=4)
    parser.add_argument("--click-weight", type=float, default=8.0)
    parser.add_argument("--prior-weight", type=float, default=0.18)
    parser.add_argument("--min-conf-percentile", type=float, default=10.0)
    parser.add_argument("--fit-mode", choices=["affine_correction", "direct_homography"], default="affine_correction")
    args = parser.parse_args()

    base = json.loads(Path(args.base_fit_json).read_text())
    click_data = json.loads(Path(args.clicks).read_text())
    clicks = click_data.get("clicks", [])
    if not clicks:
        raise RuntimeError(f"No clicks found in {args.clicks}")

    with np.load(args.vggt_npz) as data:
        frame_indices = data["frame_indices"].astype(int) if "frame_indices" in data else np.arange(data["depth_map"].shape[0])
        extrinsic = data["extrinsic"].astype(np.float32)
        intrinsic = data["intrinsic"].astype(np.float32)
        depth = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    frame_lookup = {int(frame): i for i, frame in enumerate(frame_indices.tolist())}

    plane = base["floor_plane"]
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    old_H = np.asarray(base["plane_to_floor_homography"], dtype=np.float64)

    click_uv = []
    click_floor = []
    click_metrics = []
    original_shape = (args.image_height, args.image_width)
    for click in clicks:
        frame = int(click["frame"])
        if frame not in frame_lookup:
            click_metrics.append({"frame": frame, "landmark_id": click.get("landmark_id"), "used": False, "reason": "frame_not_in_vggt"})
            continue
        local_idx = frame_lookup[frame]
        image_xy = np.asarray([float(click["image"]["x"]), float(click["image"]["y"])], dtype=np.float64)
        uv, metrics = sample_click_plane_uv(
            world_points[local_idx],
            depth_conf[local_idx],
            intrinsic[local_idx],
            extrinsic[local_idx],
            image_xy,
            original_shape,
            center,
            basis_u,
            basis_v,
            args.click_window_radius,
            args.min_conf_percentile,
        )
        metrics.update({"frame": frame, "landmark_id": click.get("landmark_id"), "image": click["image"], "world": click["world"]})
        if uv is None:
            metrics["used"] = False
            click_metrics.append(metrics)
            continue
        metrics["used"] = True
        click_uv.append(uv)
        click_floor.append([float(click["world"]["x"]), float(click["world"]["y"])])
        click_metrics.append(metrics)

    if len(click_uv) < 2:
        raise RuntimeError(f"Need at least two usable clicked VGGT points; got {len(click_uv)}")

    click_uv = np.asarray(click_uv, dtype=np.float64)
    click_floor = np.asarray(click_floor, dtype=np.float64)

    prior_floor = dense_floor_model(step_ft=4.0).astype(np.float64)
    H_floor_to_plane = np.linalg.inv(old_H)
    prior_uv = apply_homography(H_floor_to_plane, prior_floor)
    valid_prior = np.isfinite(prior_uv).all(axis=1)
    prior_uv = prior_uv[valid_prior]
    prior_floor = prior_floor[valid_prior]

    old_click_pred = transform_plane_to_floor(old_H, click_uv)
    old_click_errors = np.linalg.norm(old_click_pred - click_floor, axis=1)

    if args.fit_mode == "direct_homography":
        def residuals(params: np.ndarray) -> np.ndarray:
            H = params_to_h(params)
            values = []
            pred_click = transform_plane_to_floor(H, click_uv)
            valid_click = np.isfinite(pred_click).all(axis=1)
            if valid_click.any():
                values.extend(((pred_click[valid_click] - click_floor[valid_click]) * args.click_weight).reshape(-1).tolist())
            if (~valid_click).any():
                values.extend([1e4] * int((~valid_click).sum()) * 2)
            pred_prior = transform_plane_to_floor(H, prior_uv)
            valid_prior_res = np.isfinite(pred_prior).all(axis=1)
            if valid_prior_res.any():
                values.extend(((pred_prior[valid_prior_res] - prior_floor[valid_prior_res]) * args.prior_weight).reshape(-1).tolist())
            return np.asarray(values, dtype=np.float64)

        result = least_squares(
            residuals,
            h_to_params(old_H),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        new_H = params_to_h(result.x)
        correction = None
    else:
        old_click_floor = transform_plane_to_floor(old_H, click_uv)
        old_prior_floor = transform_plane_to_floor(old_H, prior_uv)

        def residuals(params: np.ndarray) -> np.ndarray:
            values = []
            pred_click = apply_affine(params, old_click_floor)
            values.extend(((pred_click - click_floor) * args.click_weight).reshape(-1).tolist())
            pred_prior = apply_affine(params, old_prior_floor)
            values.extend(((pred_prior - old_prior_floor) * args.prior_weight).reshape(-1).tolist())
            return np.asarray(values, dtype=np.float64)

        result = least_squares(
            residuals,
            np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        a, b, c, d, tx, ty = result.x
        correction = np.asarray([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
        new_H = correction @ old_H
    new_click_pred = transform_plane_to_floor(new_H, click_uv)
    new_click_errors = np.linalg.norm(new_click_pred - click_floor, axis=1)
    prior_new = transform_plane_to_floor(new_H, prior_uv)
    prior_old = transform_plane_to_floor(old_H, prior_uv)
    prior_drift = np.linalg.norm(prior_new - prior_old, axis=1)

    output = dict(base)
    output["plane_to_floor_homography"] = new_H.tolist()
    output["faceoff_corner_refinement"] = {
        "schema": "vggt_faceoff_corner_refinement_v1",
        "source_base_fit_json": args.base_fit_json,
        "source_clicks": args.clicks,
        "usable_clicks": int(len(click_uv)),
        "total_clicks": int(len(clicks)),
        "click_weight": float(args.click_weight),
        "prior_weight": float(args.prior_weight),
        "fit_mode": args.fit_mode,
        "floor_space_affine_correction": None if correction is None else correction.tolist(),
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "old_mean_click_error_ft": float(np.mean(old_click_errors)),
        "new_mean_click_error_ft": float(np.mean(new_click_errors)),
        "old_max_click_error_ft": float(np.max(old_click_errors)),
        "new_max_click_error_ft": float(np.max(new_click_errors)),
        "mean_prior_drift_ft": float(np.mean(prior_drift)),
        "max_prior_drift_ft": float(np.max(prior_drift)),
        "click_metrics": click_metrics,
    }
    output["alignment_metrics"] = dict(base.get("alignment_metrics", {}))
    output["alignment_metrics"]["faceoff_refined_mean_click_error_ft"] = float(np.mean(new_click_errors))
    output["alignment_metrics"]["faceoff_refined_mean_prior_drift_ft"] = float(np.mean(prior_drift))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "usable_clicks": int(len(click_uv)),
                "old_mean_click_error_ft": float(np.mean(old_click_errors)),
                "new_mean_click_error_ft": float(np.mean(new_click_errors)),
                "old_max_click_error_ft": float(np.max(old_click_errors)),
                "new_max_click_error_ft": float(np.max(new_click_errors)),
                "mean_prior_drift_ft": float(np.mean(prior_drift)),
                "max_prior_drift_ft": float(np.max(prior_drift)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
