from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from nll_field_geometry import circle_points, goal_crease_samples, line_points
from render_birds_eye_locations import (
    CORNER_RADIUS_FT,
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    fit_homographies,
    rounded_floor_points,
)


def floor_feature_samples() -> np.ndarray:
    points: list[np.ndarray] = []
    points.append(rounded_floor_points(samples_per_corner=64).astype(np.float64))
    for y in [0.0, FLOOR_WIDTH_FT]:
        points.append(line_points(CORNER_RADIUS_FT, y, FLOOR_LENGTH_FT - CORNER_RADIUS_FT, y, samples=220))
    for x in [12.0, 57.5, 100.0, 142.5, 188.0]:
        points.append(line_points(x, 0.0, x, FLOOR_WIDTH_FT, samples=140))
    points.append(circle_points(100.0, 42.5, 11.0, samples=180))
    points.append(goal_crease_samples(arc_samples=180, chord_samples=60))
    for x in [12.0, 188.0]:
        points.append(line_points(x, 40.125, x, 44.875, samples=24))
    for x, y in [(100.0, 42.5), (42.5, 15.0), (42.5, 70.0), (157.5, 15.0), (157.5, 70.0)]:
        points.append(circle_points(x, y, 0.8, samples=24))
    return np.concatenate(points, axis=0)


def params_to_h(params: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )


def h_to_params(H: np.ndarray) -> np.ndarray:
    H = H / H[2, 2]
    return np.asarray([H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]], dtype=np.float64)


def project_world(H_world_to_image: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    pts = cv2.perspectiveTransform(world_points.astype(np.float32).reshape(1, -1, 2), H_world_to_image).reshape(-1, 2)
    return pts.astype(np.float64)


def nearest_feature_residuals(
    H_world_to_image: np.ndarray,
    image_clicks: np.ndarray,
    world_samples: np.ndarray,
    max_sample_distance_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = project_world(H_world_to_image, world_samples)
    finite = np.isfinite(projected).all(axis=1)
    projected = projected[finite]
    sample_indices = np.nonzero(finite)[0]
    if len(projected) == 0:
        return np.full((len(image_clicks), 2), 1e4), np.zeros((len(image_clicks),), dtype=int), np.full((len(image_clicks),), 1e4)
    diff = image_clicks[:, None, :] - projected[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = dist2.argmin(axis=1)
    residuals = diff[np.arange(len(image_clicks)), nearest]
    distances = np.sqrt(dist2[np.arange(len(image_clicks)), nearest])
    if max_sample_distance_px > 0:
        scale = np.maximum(1.0, distances / max_sample_distance_px)
        residuals = residuals / scale[:, None]
    return residuals, sample_indices[nearest], distances


def refine_frame(
    initial_H_image_to_world: np.ndarray,
    image_clicks: np.ndarray,
    world_samples: np.ndarray,
    max_nfev: int,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    initial_H_world_to_image = np.linalg.inv(initial_H_image_to_world)
    p0 = h_to_params(initial_H_world_to_image)
    initial_projected = project_world(initial_H_world_to_image, world_samples[::25])

    def residual_fn(params: np.ndarray) -> np.ndarray:
        H = params_to_h(params)
        click_residuals, _, _ = nearest_feature_residuals(H, image_clicks, world_samples, max_sample_distance_px=80.0)
        residuals = click_residuals.reshape(-1)
        if regularization > 0:
            projected = project_world(H, world_samples[::25])
            valid = np.isfinite(projected).all(axis=1) & np.isfinite(initial_projected).all(axis=1)
            if valid.any():
                residuals = np.concatenate([residuals, (projected[valid] - initial_projected[valid]).reshape(-1) * regularization])
        return residuals

    result = least_squares(
        residual_fn,
        p0,
        loss="soft_l1",
        f_scale=12.0,
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    H_world_to_image = params_to_h(result.x)
    residuals, nearest, distances = nearest_feature_residuals(H_world_to_image, image_clicks, world_samples, max_sample_distance_px=0.0)
    H_image_to_world = np.linalg.inv(H_world_to_image)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "mean_error_px": float(distances.mean()) if len(distances) else 0.0,
        "max_error_px": float(distances.max()) if len(distances) else 0.0,
        "errors_px": [float(v) for v in distances.tolist()],
        "nearest_sample_indices": [int(v) for v in nearest.tolist()],
        "nfev": int(result.nfev),
    }
    return H_image_to_world, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled-clicks", default="outputs/floor_unlabeled_clicks.json")
    parser.add_argument("--initial-calibration", default="outputs/floor_homography_inferred_frame77.json")
    parser.add_argument("--output", default="outputs/floor_homography_curve_refined.json")
    parser.add_argument("--frame", type=int, default=77)
    parser.add_argument("--max-nfev", type=int, default=5000)
    parser.add_argument("--regularization", type=float, default=0.015)
    parser.add_argument("--ransac-threshold-ft", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unlabeled = json.loads(Path(args.unlabeled_clicks).read_text())
    initial = json.loads(Path(args.initial_calibration).read_text())
    initial_fit = fit_homographies(initial, args.ransac_threshold_ft)[0]
    clicks = [click for click in unlabeled.get("clicks", []) if int(click["frame"]) == args.frame]
    if len(clicks) < 4:
        raise RuntimeError(f"Need at least 4 unlabeled clicks on frame {args.frame}; found {len(clicks)}")
    image_clicks = np.asarray([[click["image"]["x"], click["image"]["y"]] for click in clicks], dtype=np.float64)
    world_samples = floor_feature_samples()
    refined_H, metrics = refine_frame(
        initial_fit.H,
        image_clicks,
        world_samples,
        max_nfev=args.max_nfev,
        regularization=args.regularization,
    )
    out = {
        "schema": "floor_homography_matrix_v1",
        "world_units": "feet",
        "floor": {"length_ft": FLOOR_LENGTH_FT, "width_ft": FLOOR_WIDTH_FT, "corner_radius_ft": CORNER_RADIUS_FT},
        "source_unlabeled_clicks": args.unlabeled_clicks,
        "source_initial_calibration": args.initial_calibration,
        "fit_method": "curve_distance_refinement",
        "homographies": [
            {
                "frame": args.frame,
                "points": len(clicks),
                "inliers": len(clicks),
                "mean_error_px": metrics["mean_error_px"],
                "max_error_px": metrics["max_error_px"],
                "metrics": metrics,
                "matrix_image_to_world": refined_H.tolist(),
            }
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"output": args.output, "points": len(clicks), **metrics}, indent=2))


if __name__ == "__main__":
    main()
