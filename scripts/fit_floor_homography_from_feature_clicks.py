from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from render_birds_eye_locations import (
    CORNER_RADIUS_FT,
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    fit_homographies,
    nearest_fit,
)


def line_points(x1: float, y1: float, x2: float, y2: float, samples: int = 120) -> np.ndarray:
    return np.column_stack([np.linspace(x1, x2, samples), np.linspace(y1, y2, samples)]).astype(np.float64)


def arc_points(cx: float, cy: float, radius: float, start_deg: float, stop_deg: float, samples: int = 100) -> np.ndarray:
    angles = np.deg2rad(np.linspace(start_deg, stop_deg, samples))
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]).astype(np.float64)


def rounded_outline_samples(samples_per_segment: int = 140) -> np.ndarray:
    r = CORNER_RADIUS_FT
    parts = [
        line_points(r, 0.0, FLOOR_LENGTH_FT - r, 0.0, samples_per_segment),
        line_points(FLOOR_LENGTH_FT, r, FLOOR_LENGTH_FT, FLOOR_WIDTH_FT - r, samples_per_segment // 2),
        line_points(FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT, r, FLOOR_WIDTH_FT, samples_per_segment),
        line_points(0.0, FLOOR_WIDTH_FT - r, 0.0, r, samples_per_segment // 2),
        arc_points(r, r, r, 180.0, 270.0, 80),
        arc_points(FLOOR_LENGTH_FT - r, r, r, 270.0, 360.0, 80),
        arc_points(FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT - r, r, 0.0, 90.0, 80),
        arc_points(r, FLOOR_WIDTH_FT - r, r, 90.0, 180.0, 80),
    ]
    return np.concatenate(parts, axis=0)


def crease_samples() -> np.ndarray:
    radius = 9.25
    lower = 42.5 - radius
    upper = 42.5 + radius
    left = [
        arc_points(12.0, 42.5, radius, -90.0, 90.0, 140),
        line_points(0.0, lower, 12.0, lower, 60),
        line_points(0.0, upper, 12.0, upper, 60),
        line_points(0.0, lower, 0.0, upper, 80),
        line_points(12.0, lower, 12.0, upper, 50),
    ]
    right = [
        arc_points(188.0, 42.5, radius, 90.0, 270.0, 140),
        line_points(188.0, lower, 200.0, lower, 60),
        line_points(188.0, upper, 200.0, upper, 60),
        line_points(200.0, lower, 200.0, upper, 80),
        line_points(188.0, lower, 188.0, upper, 50),
    ]
    return np.concatenate(left + right, axis=0).astype(np.float64)


def feature_samples() -> dict[str, np.ndarray]:
    return {
        "left_restraining_line": line_points(57.5, 0.0, 57.5, FLOOR_WIDTH_FT, 180),
        "right_restraining_line": line_points(142.5, 0.0, 142.5, FLOOR_WIDTH_FT, 180),
        "midfield_line": line_points(100.0, 0.0, 100.0, FLOOR_WIDTH_FT, 180),
        "field_outline": rounded_outline_samples(),
        "goal_crease": crease_samples(),
    }


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
    return cv2.perspectiveTransform(world_points.astype(np.float32).reshape(1, -1, 2), H_world_to_image).reshape(-1, 2).astype(np.float64)


def feature_residual(click: dict, H_world_to_image: np.ndarray, samples_by_feature: dict[str, np.ndarray]) -> tuple[np.ndarray, float, int]:
    samples = samples_by_feature[click["feature"]]
    projected = project_world(H_world_to_image, samples)
    image = np.asarray([click["image"]["x"], click["image"]["y"]], dtype=np.float64)
    finite = np.isfinite(projected).all(axis=1)
    if not finite.any():
        return np.asarray([1e4, 1e4], dtype=np.float64), 1e4, -1
    projected_valid = projected[finite]
    diff = image[None, :] - projected_valid
    dist2 = np.sum(diff * diff, axis=1)
    idx = int(dist2.argmin())
    residual = diff[idx]
    distance = float(np.sqrt(dist2[idx]))
    sample_indices = np.nonzero(finite)[0]
    return residual, distance, int(sample_indices[idx])


def refine_frame(
    frame: int,
    clicks: list[dict],
    initial_H_image_to_world: np.ndarray,
    samples_by_feature: dict[str, np.ndarray],
    max_nfev: int,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    initial_H_world_to_image = np.linalg.inv(initial_H_image_to_world)
    p0 = h_to_params(initial_H_world_to_image)
    all_samples = np.concatenate(list(samples_by_feature.values()), axis=0)
    regularizer_samples = all_samples[:: max(1, len(all_samples) // 300)]
    initial_projected = project_world(initial_H_world_to_image, regularizer_samples)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        H = params_to_h(params)
        residuals = []
        for click in clicks:
            residual, distance, _ = feature_residual(click, H, samples_by_feature)
            scale = max(1.0, distance / 90.0)
            residuals.extend((residual / scale).tolist())
        if regularization > 0:
            projected = project_world(H, regularizer_samples)
            valid = np.isfinite(projected).all(axis=1) & np.isfinite(initial_projected).all(axis=1)
            if valid.any():
                residuals.extend(((projected[valid] - initial_projected[valid]) * regularization).reshape(-1).tolist())
        return np.asarray(residuals, dtype=np.float64)

    result = least_squares(
        residual_fn,
        p0,
        loss="soft_l1",
        f_scale=10.0,
        max_nfev=max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    H_world_to_image = params_to_h(result.x)
    H_image_to_world = np.linalg.inv(H_world_to_image)

    errors = []
    nearest = []
    for click in clicks:
        _, distance, sample_idx = feature_residual(click, H_world_to_image, samples_by_feature)
        errors.append(distance)
        nearest.append(sample_idx)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "mean_error_px": float(np.mean(errors)) if errors else 0.0,
        "max_error_px": float(np.max(errors)) if errors else 0.0,
        "errors_px": [float(v) for v in errors],
        "nearest_sample_indices": nearest,
        "nfev": int(result.nfev),
        "features": {feature: sum(1 for click in clicks if click["feature"] == feature) for feature in sorted({click["feature"] for click in clicks})},
    }
    return H_image_to_world, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-clicks", default="outputs/floor_feature_clicks.json")
    parser.add_argument("--initial-calibration", default="outputs/floor_homography_dynamic.json")
    parser.add_argument("--output", default="outputs/floor_homography_feature_fit.json")
    parser.add_argument("--reference-frame", type=int, default=78)
    parser.add_argument("--max-nfev", type=int, default=5000)
    parser.add_argument("--regularization", type=float, default=0.012)
    parser.add_argument("--ransac-threshold-ft", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_data = json.loads(Path(args.feature_clicks).read_text())
    initial_data = json.loads(Path(args.initial_calibration).read_text())
    initial_fits = fit_homographies(initial_data, args.ransac_threshold_ft)
    samples_by_feature = feature_samples()

    by_frame: dict[int, list[dict]] = {}
    valid_features = set(samples_by_feature)
    for click in feature_data.get("clicks", []):
        if click.get("feature") not in valid_features:
            continue
        by_frame.setdefault(int(click["frame"]), []).append(click)

    homographies = []
    for frame in sorted(by_frame):
        clicks = by_frame[frame]
        if len(clicks) < 4 or len({click["feature"] for click in clicks}) < 2:
            continue
        initial_fit = nearest_fit(initial_fits, frame)
        H, metrics = refine_frame(
            frame,
            clicks,
            initial_fit.H,
            samples_by_feature,
            max_nfev=args.max_nfev,
            regularization=args.regularization,
        )
        homographies.append(
            {
                "frame": frame,
                "points": len(clicks),
                "inliers": len(clicks),
                "mean_error_px": metrics["mean_error_px"],
                "max_error_px": metrics["max_error_px"],
                "metrics": metrics,
                "matrix_image_to_world": H.tolist(),
            }
        )

    if not homographies:
        raise RuntimeError("No usable feature-click frames. Need at least 4 clicks and at least 2 feature categories on a frame.")

    # Put the requested reference frame first when present, so older callers that
    # read the first matrix still get the intended base calibration.
    homographies.sort(key=lambda item: (0 if item["frame"] == args.reference_frame else 1, abs(item["frame"] - args.reference_frame), item["frame"]))

    output = {
        "schema": "floor_homography_matrix_v1",
        "world_units": "feet",
        "floor": {"length_ft": FLOOR_LENGTH_FT, "width_ft": FLOOR_WIDTH_FT, "corner_radius_ft": CORNER_RADIUS_FT},
        "source_feature_clicks": args.feature_clicks,
        "source_initial_calibration": args.initial_calibration,
        "fit_method": "coarse_feature_distance_refinement",
        "crease_model": "front semicircle plus rear rectangle, mirrored at both goals",
        "reference_frame": args.reference_frame,
        "homographies": homographies,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": args.output,
                "homography_frames": [item["frame"] for item in homographies],
                "mean_errors_px": {str(item["frame"]): item["mean_error_px"] for item in homographies},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
