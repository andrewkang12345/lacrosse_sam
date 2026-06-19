from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from fit_floor_homography_from_feature_clicks import feature_residual, feature_samples, refine_frame
from render_birds_eye_locations import (
    CORNER_RADIUS_FT,
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    fit_homographies,
    nearest_fit,
)


AUTO_FEATURES = {
    "auto_floor_lines": ["left_restraining_line", "right_restraining_line", "midfield_line", "goal_crease"],
    "auto_white_lines": ["left_restraining_line", "right_restraining_line", "midfield_line", "goal_crease"],
    "auto_all_landmarks": ["left_restraining_line", "right_restraining_line", "midfield_line", "goal_crease", "field_outline"],
}


def arc_points(cx: float, cy: float, radius: float, start_deg: float, stop_deg: float, samples: int = 120) -> np.ndarray:
    angles = np.deg2rad(np.linspace(start_deg, stop_deg, samples))
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]).astype(np.float64)


def crease_side_samples() -> dict[str, np.ndarray]:
    return {
        "left": arc_points(12.0, 42.5, 9.25, -90.0, 90.0, 160),
        "right": arc_points(188.0, 42.5, 9.25, 90.0, 270.0, 160),
    }


def parse_mask_source(value: str) -> tuple[str, Path, int | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mask sources must be feature=path or feature=path#object_id")
    feature, path = value.split("=", 1)
    feature = feature.strip()
    if not feature:
        raise argparse.ArgumentTypeError("empty feature in mask source")
    object_id = None
    if "#" in path:
        path, raw_object_id = path.rsplit("#", 1)
        object_id = int(raw_object_id)
    return feature, Path(path), object_id


def load_mask(mask_dir: Path, frame_idx: int, shape: tuple[int, int], object_id: int | None) -> np.ndarray:
    npz_path = mask_dir / f"{frame_idx:08d}.npz"
    png_path = mask_dir / f"{frame_idx:08d}.png"
    height, width = shape
    if npz_path.exists():
        data = np.load(npz_path)
        object_ids = [int(value) for value in data["object_ids"].tolist()]
        masks = data["masks"].astype(bool)
        if masks.size == 0:
            return np.zeros((height, width), dtype=bool)
        if object_id is not None:
            if object_id not in object_ids:
                return np.zeros((height, width), dtype=bool)
            selected = masks[object_ids.index(object_id)]
        else:
            selected = masks.any(axis=0)
        if selected.shape == (height, width):
            return selected
        return cv2.resize(selected.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if png_path.exists():
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((height, width), dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if object_id is not None:
            return mask == object_id
        return mask > 0
    return np.zeros((height, width), dtype=bool)


def cleaned_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask_u8)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            clean[labels == label] = 1
    if not clean.any():
        return clean.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    return clean.astype(bool)


def sample_mask_pixels(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    coords = np.column_stack([xs, ys]).astype(np.float64)
    if len(coords) <= max_points:
        return coords
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=max_points, replace=False)
    return coords[idx]


def project_world(H_world_to_image: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(world_points.astype(np.float32).reshape(1, -1, 2), H_world_to_image).reshape(-1, 2).astype(np.float64)


def similarity_to_h(params: np.ndarray) -> np.ndarray:
    tx, ty, theta, log_scale = [float(v) for v in params]
    scale = float(np.exp(log_scale))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [scale * c, -scale * s, tx],
            [scale * s, scale * c, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def refine_frame_image_similarity(
    frame: int,
    clicks: list[dict],
    initial_H_image_to_world: np.ndarray,
    samples_by_feature: dict[str, np.ndarray],
    max_nfev: int,
    regularization: float,
) -> tuple[np.ndarray, dict]:
    from scipy.optimize import least_squares

    initial_H_world_to_image = np.linalg.inv(initial_H_image_to_world)
    p0 = np.zeros(4, dtype=np.float64)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        image_adjustment = similarity_to_h(params)
        H_world_to_image = image_adjustment @ initial_H_world_to_image
        residuals = []
        for click in clicks:
            residual, distance, _ = feature_residual(click, H_world_to_image, samples_by_feature)
            scale = max(1.0, distance / 90.0)
            residuals.extend((residual / scale).tolist())
        if regularization > 0:
            tx, ty, theta, log_scale = params
            residuals.extend(
                [
                    float(tx) * regularization,
                    float(ty) * regularization,
                    float(theta) * regularization * 220.0,
                    float(log_scale) * regularization * 220.0,
                ]
            )
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
    image_adjustment = similarity_to_h(result.x)
    H_world_to_image = image_adjustment @ initial_H_world_to_image
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
        "image_similarity_adjustment": {
            "tx_px": float(result.x[0]),
            "ty_px": float(result.x[1]),
            "rotation_deg": float(np.rad2deg(result.x[2])),
            "scale": float(np.exp(result.x[3])),
        },
    }
    return H_image_to_world, metrics


def fit_line_primitive(mask: np.ndarray, max_points: int, seed: int) -> dict | None:
    points = sample_mask_pixels(mask, max_points, seed)
    if len(points) < 12:
        return None
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)]
    a = -vy
    b = vx
    norm = float(np.hypot(a, b))
    if norm < 1e-6:
        return None
    a /= norm
    b /= norm
    c = -(a * x0 + b * y0)
    distances = np.abs(points @ np.asarray([a, b], dtype=np.float64) + c)
    return {
        "type": "line",
        "line": [float(a), float(b), float(c)],
        "points": int(len(points)),
        "mean_fit_error_px": float(np.mean(distances)),
        "max_fit_error_px": float(np.max(distances)),
    }


def line_from_points(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(points) < 2:
        return None
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)]
    a = -vy
    b = vx
    norm = float(np.hypot(a, b))
    if norm < 1e-6:
        return None
    a /= norm
    b /= norm
    c = -(a * x0 + b * y0)
    direction_angle = float(np.arctan2(-a, b))
    return np.asarray([a, b, c], dtype=np.float64), direction_angle


def rotation_about(point: np.ndarray, theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x, y = float(point[0]), float(point[1])
    return np.asarray(
        [
            [c, -s, x - c * x + s * y],
            [s, c, y - s * x - c * y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def refine_single_line_primitive(
    frame: int,
    primitive: dict,
    initial_H_image_to_world: np.ndarray,
    samples_by_feature: dict[str, np.ndarray],
    max_rotation_deg: float,
    max_translation_px: float,
) -> tuple[np.ndarray, dict] | None:
    feature = primitive["feature"]
    if primitive["type"] != "line" or feature not in samples_by_feature:
        return None
    initial_H_world_to_image = np.linalg.inv(initial_H_image_to_world)
    projected = project_world(initial_H_world_to_image, samples_by_feature[feature])
    valid = np.isfinite(projected).all(axis=1)
    if valid.sum() < 2:
        return None
    projected = projected[valid]
    fitted = line_from_points(projected)
    if fitted is None:
        return None
    _, model_angle = fitted
    target_line = np.asarray(primitive["line"], dtype=np.float64)
    target_angle = float(np.arctan2(-target_line[0], target_line[1]))
    angle_delta = target_angle - model_angle
    angle_delta = float(np.arctan2(np.sin(angle_delta), np.cos(angle_delta)))
    center = projected.mean(axis=0)
    rotate = rotation_about(center, angle_delta)
    rotated_center = cv2.perspectiveTransform(center.astype(np.float32).reshape(1, 1, 2), rotate).reshape(2).astype(np.float64)
    signed_distance = float(target_line[0] * rotated_center[0] + target_line[1] * rotated_center[1] + target_line[2])
    translate = np.asarray(
        [
            [1.0, 0.0, -signed_distance * target_line[0]],
            [0.0, 1.0, -signed_distance * target_line[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation_vec = np.asarray([-signed_distance * target_line[0], -signed_distance * target_line[1]], dtype=np.float64)
    if abs(float(np.rad2deg(angle_delta))) > max_rotation_deg or float(np.linalg.norm(translation_vec)) > max_translation_px:
        return None
    adjustment = translate @ rotate
    H_world_to_image = adjustment @ initial_H_world_to_image
    H_image_to_world = np.linalg.inv(H_world_to_image)
    residuals, errors, feature_counts = primitive_residuals(H_world_to_image, [primitive], samples_by_feature, sample_stride=4)
    metrics = {
        "success": True,
        "cost": float(np.sum(np.square(residuals)) * 0.5) if residuals else 0.0,
        "mean_error_px": float(np.mean(errors)) if errors else 0.0,
        "max_error_px": float(np.max(errors)) if errors else 0.0,
        "nfev": 0,
        "features": feature_counts,
        "primitives": [
            {
                "feature": primitive["feature"],
                "type": primitive["type"],
                "points": int(primitive["points"]),
                "mean_fit_error_px": primitive.get("mean_fit_error_px"),
            }
        ],
        "single_line_adjustment": {
            "rotation_deg": float(np.rad2deg(angle_delta)),
            "translation_px": [float(translation_vec[0]), float(translation_vec[1])],
        },
    }
    return H_image_to_world, metrics


def fit_mask_point_primitive(mask: np.ndarray, max_points: int, seed: int) -> dict | None:
    points = sample_mask_pixels(mask, max_points, seed)
    if len(points) < 12:
        return None
    return {
        "type": "mask_points",
        "points": int(len(points)),
        "tree": cKDTree(points),
        "samples": points,
    }


def choose_crease_side(mask_points: np.ndarray, initial_H_image_to_world: np.ndarray) -> tuple[str, np.ndarray]:
    H_world_to_image = np.linalg.inv(initial_H_image_to_world)
    tree = cKDTree(mask_points)
    best_side = "right"
    best_samples = crease_side_samples()[best_side]
    best_distance = float("inf")
    for side, world_samples in crease_side_samples().items():
        projected = project_world(H_world_to_image, world_samples)
        valid = np.isfinite(projected).all(axis=1)
        if not valid.any():
            continue
        distances, _ = tree.query(projected[valid], k=1)
        score = float(np.median(distances))
        if score < best_distance:
            best_distance = score
            best_side = side
            best_samples = world_samples
    return best_side, best_samples


def primitive_from_mask_source(
    feature: str,
    mask_dir: Path,
    object_id: int | None,
    frame_idx: int,
    frame_shape: tuple[int, int],
    initial_H_image_to_world: np.ndarray,
    max_points_per_source: int,
    min_component_area: int,
) -> dict | None:
    mask = cleaned_mask(load_mask(mask_dir, frame_idx, frame_shape, object_id), min_component_area)
    if not mask.any():
        return None
    if feature in {"left_restraining_line", "right_restraining_line", "midfield_line"}:
        primitive = fit_line_primitive(mask, max_points_per_source, seed=frame_idx * 1009 + len(feature))
    elif feature == "goal_crease":
        primitive = fit_mask_point_primitive(mask, max_points_per_source, seed=frame_idx * 1009 + len(feature))
        if primitive is not None:
            side, world_samples = choose_crease_side(primitive["samples"], initial_H_image_to_world)
            primitive["world_samples"] = world_samples
            primitive["crease_side"] = side
    else:
        primitive = fit_mask_point_primitive(mask, max_points_per_source, seed=frame_idx * 1009 + len(feature))
    if primitive is None:
        return None
    primitive["feature"] = feature
    primitive["object_id"] = object_id
    return primitive


def primitive_residuals(
    H_world_to_image: np.ndarray,
    primitives: list[dict],
    samples_by_feature: dict[str, np.ndarray],
    sample_stride: int,
) -> tuple[list[float], list[float], dict[str, int]]:
    residuals: list[float] = []
    errors: list[float] = []
    feature_counts: dict[str, int] = {}
    for primitive in primitives:
        feature = primitive["feature"]
        if feature not in samples_by_feature:
            continue
        world_samples = primitive.get("world_samples", samples_by_feature[feature])[::sample_stride]
        projected = project_world(H_world_to_image, world_samples)
        valid = np.isfinite(projected).all(axis=1)
        if not valid.any():
            continue
        projected = projected[valid]
        if primitive["type"] == "line":
            a, b, c = primitive["line"]
            d = projected @ np.asarray([a, b], dtype=np.float64) + float(c)
            residuals.extend(d.tolist())
            errors.extend(np.abs(d).tolist())
            feature_counts[feature] = feature_counts.get(feature, 0) + len(d)
        else:
            distances, nearest = primitive["tree"].query(projected, k=1)
            nearest_points = primitive["samples"][nearest]
            diff = projected - nearest_points
            residuals.extend(diff.reshape(-1).tolist())
            errors.extend(distances.tolist())
            feature_counts[feature] = feature_counts.get(feature, 0) + len(distances)
    return residuals, errors, feature_counts


def homography_projection_ok(H_image_to_world: np.ndarray, samples_by_feature: dict[str, np.ndarray], max_abs_px: float = 8000.0) -> bool:
    try:
        H_world_to_image = np.linalg.inv(H_image_to_world)
    except np.linalg.LinAlgError:
        return False
    all_samples = np.concatenate(list(samples_by_feature.values()), axis=0)
    projected = project_world(H_world_to_image, all_samples[:: max(1, len(all_samples) // 400)])
    return bool(np.isfinite(projected).all() and np.max(np.abs(projected)) <= max_abs_px)


def refine_frame_primitive_image_similarity(
    frame: int,
    primitives: list[dict],
    initial_H_image_to_world: np.ndarray,
    samples_by_feature: dict[str, np.ndarray],
    max_nfev: int,
    regularization: float,
    sample_stride: int,
) -> tuple[np.ndarray, dict]:
    from scipy.optimize import least_squares

    initial_H_world_to_image = np.linalg.inv(initial_H_image_to_world)
    p0 = np.zeros(4, dtype=np.float64)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        image_adjustment = similarity_to_h(params)
        H_world_to_image = image_adjustment @ initial_H_world_to_image
        residuals, _, _ = primitive_residuals(H_world_to_image, primitives, samples_by_feature, sample_stride)
        if regularization > 0:
            tx, ty, theta, log_scale = params
            residuals.extend(
                [
                    float(tx) * regularization,
                    float(ty) * regularization,
                    float(theta) * regularization * 220.0,
                    float(log_scale) * regularization * 220.0,
                ]
            )
        return np.asarray(residuals, dtype=np.float64)

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
    image_adjustment = similarity_to_h(result.x)
    H_world_to_image = image_adjustment @ initial_H_world_to_image
    H_image_to_world = np.linalg.inv(H_world_to_image)
    _, errors, feature_counts = primitive_residuals(H_world_to_image, primitives, samples_by_feature, sample_stride)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "mean_error_px": float(np.mean(errors)) if errors else 0.0,
        "max_error_px": float(np.max(errors)) if errors else 0.0,
        "nfev": int(result.nfev),
        "features": feature_counts,
        "primitives": [
            {
                "feature": item["feature"],
                "type": item["type"],
                "points": int(item["points"]),
                "mean_fit_error_px": item.get("mean_fit_error_px"),
            }
            for item in primitives
        ],
        "image_similarity_adjustment": {
            "tx_px": float(result.x[0]),
            "ty_px": float(result.x[1]),
            "rotation_deg": float(np.rad2deg(result.x[2])),
            "scale": float(np.exp(result.x[3])),
        },
    }
    return H_image_to_world, metrics


def build_feature_trees(
    H_image_to_world: np.ndarray,
    samples_by_feature: dict[str, np.ndarray],
    candidate_features: list[str],
) -> tuple[cKDTree, list[str]]:
    H_world_to_image = np.linalg.inv(H_image_to_world)
    image_samples = []
    sample_features = []
    for feature in candidate_features:
        projected = project_world(H_world_to_image, samples_by_feature[feature])
        valid = np.isfinite(projected).all(axis=1)
        if not valid.any():
            continue
        image_samples.append(projected[valid])
        sample_features.extend([feature] * int(valid.sum()))
    if not image_samples:
        raise RuntimeError("No projectable feature samples for landmark-mask assignment.")
    all_image_samples = np.concatenate(image_samples, axis=0)
    return cKDTree(all_image_samples), sample_features


def clicks_from_mask_source(
    feature: str,
    mask_dir: Path,
    object_id: int | None,
    frame_idx: int,
    frame_shape: tuple[int, int],
    initial_H: np.ndarray,
    samples_by_feature: dict[str, np.ndarray],
    excluded_features: set[str],
    max_points_per_source: int,
    max_assignment_distance_px: float,
    min_component_area: int,
) -> list[dict]:
    raw_mask = load_mask(mask_dir, frame_idx, frame_shape, object_id)
    mask = cleaned_mask(raw_mask, min_component_area)
    pixels = sample_mask_pixels(mask, max_points_per_source, seed=frame_idx * 1009 + len(feature))
    if len(pixels) == 0:
        return []

    if feature in AUTO_FEATURES:
        candidates = [item for item in AUTO_FEATURES[feature] if item not in excluded_features]
    elif feature in samples_by_feature and feature not in excluded_features:
        candidates = [feature]
    else:
        raise ValueError(f"Unknown or excluded feature source: {feature}")
    if not candidates:
        return []

    tree, sample_features = build_feature_trees(initial_H, samples_by_feature, candidates)
    distances, nearest = tree.query(pixels, k=1)
    clicks = []
    for point, distance, sample_idx in zip(pixels, distances, nearest):
        if float(distance) > max_assignment_distance_px:
            continue
        assigned = sample_features[int(sample_idx)]
        clicks.append({"frame": frame_idx, "feature": assigned, "image": {"x": float(point[0]), "y": float(point[1])}})
    return clicks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--initial-calibration", default="outputs/floor_homography_feature_no_outline_dynamic.json")
    parser.add_argument("--feature-clicks", default="outputs/floor_feature_clicks_no_outline.json")
    parser.add_argument("--output", default="outputs/floor_homography_tracked_landmarks_dynamic.json")
    parser.add_argument(
        "--mask-source",
        action="append",
        type=parse_mask_source,
        default=[],
        help="Repeatable tracked mask source as feature=path or feature=path#object_id.",
    )
    parser.add_argument("--exclude-features", default="", help="Comma-separated feature IDs to ignore, e.g. field_outline")
    parser.add_argument("--max-points-per-source", type=int, default=220)
    parser.add_argument("--max-assignment-distance-px", type=float, default=38.0)
    parser.add_argument("--min-component-area", type=int, default=18)
    parser.add_argument("--min-points", type=int, default=18)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--refine-mode", choices=["full_homography", "image_similarity", "primitive_image_similarity"], default="full_homography")
    parser.add_argument("--primitive-sample-stride", type=int, default=5)
    parser.add_argument("--max-single-line-rotation-deg", type=float, default=18.0)
    parser.add_argument("--max-single-line-translation-px", type=float, default=120.0)
    parser.add_argument("--max-nfev", type=int, default=900)
    parser.add_argument("--regularization", type=float, default=0.018)
    parser.add_argument("--ransac-threshold-ft", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mask_source:
        raise RuntimeError("Provide at least one --mask-source feature=path from landmark tracking.")

    initial_data = json.loads(Path(args.initial_calibration).read_text())
    initial_fits = fit_homographies(initial_data, args.ransac_threshold_ft)
    samples_by_feature = feature_samples()
    excluded_features = {item.strip() for item in args.exclude_features.split(",") if item.strip()}
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames found in {args.frames_dir}")

    homographies = []
    assignment_metrics = []
    all_generated_clicks = []
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        initial_fit = nearest_fit(initial_fits, frame_idx)
        frame_shape = frame.shape[:2]
        if args.refine_mode == "primitive_image_similarity":
            primitives = []
            source_counts = {}
            for feature, mask_dir, object_id in args.mask_source:
                if feature in excluded_features:
                    continue
                primitive = primitive_from_mask_source(
                    feature,
                    mask_dir,
                    object_id,
                    frame_idx,
                    frame_shape,
                    initial_fit.H,
                    args.max_points_per_source,
                    args.min_component_area,
                )
                suffix = f"#{object_id}" if object_id is not None else ""
                source_counts[f"{feature}={mask_dir}{suffix}"] = 0 if primitive is None else int(primitive["points"])
                if primitive is not None:
                    primitives.append(primitive)
            primitive_features = {item["feature"] for item in primitives}
            primitive_points = sum(int(item["points"]) for item in primitives)
            if primitive_points < args.min_points or len(primitive_features) < args.min_features:
                homographies.append(
                    {
                        "frame": frame_idx,
                        "points": int(initial_fit.points),
                        "inliers": int(initial_fit.inliers),
                        "mean_error_px": float(initial_fit.mean_error_ft),
                        "matrix_image_to_world": initial_fit.H.tolist(),
                        "fallback": True,
                        "fallback_reason": "not_enough_tracked_landmark_primitives",
                    }
                )
                assignment_metrics.append(
                    {
                        "frame": frame_idx,
                        "points": primitive_points,
                        "features": {feature: sum(int(item["points"]) for item in primitives if item["feature"] == feature) for feature in sorted(primitive_features)},
                        "sources": source_counts,
                        "fallback": True,
                    }
                )
                continue
            single_line = (
                refine_single_line_primitive(
                    frame_idx,
                    primitives[0],
                    initial_fit.H,
                    samples_by_feature,
                    max_rotation_deg=args.max_single_line_rotation_deg,
                    max_translation_px=args.max_single_line_translation_px,
                )
                if len(primitives) == 1
                else None
            )
            if single_line is not None:
                H, metrics = single_line
            elif len(primitives) == 1:
                homographies.append(
                    {
                        "frame": frame_idx,
                        "points": int(initial_fit.points),
                        "inliers": int(initial_fit.inliers),
                        "mean_error_px": float(initial_fit.mean_error_ft),
                        "matrix_image_to_world": initial_fit.H.tolist(),
                        "fallback": True,
                        "fallback_reason": "single_line_correction_exceeded_limits",
                    }
                )
                assignment_metrics.append(
                    {
                        "frame": frame_idx,
                        "points": primitive_points,
                        "features": {feature: sum(int(item["points"]) for item in primitives if item["feature"] == feature) for feature in sorted(primitive_features)},
                        "sources": source_counts,
                        "fallback": True,
                    }
                )
                continue
            else:
                H, metrics = refine_frame_primitive_image_similarity(
                    frame_idx,
                    primitives,
                    initial_fit.H,
                    samples_by_feature,
                    max_nfev=args.max_nfev,
                    regularization=args.regularization,
                    sample_stride=max(1, args.primitive_sample_stride),
                )
            if not homography_projection_ok(H, samples_by_feature):
                homographies.append(
                    {
                        "frame": frame_idx,
                        "points": int(initial_fit.points),
                        "inliers": int(initial_fit.inliers),
                        "mean_error_px": float(initial_fit.mean_error_ft),
                        "matrix_image_to_world": initial_fit.H.tolist(),
                        "fallback": True,
                        "fallback_reason": "refined_projection_out_of_render_bounds",
                    }
                )
                assignment_metrics.append(
                    {
                        "frame": frame_idx,
                        "points": primitive_points,
                        "features": metrics.get("features", {}),
                        "sources": source_counts,
                        "fallback": True,
                    }
                )
                continue
            homographies.append(
                {
                    "frame": frame_idx,
                    "points": primitive_points,
                    "inliers": primitive_points,
                    "mean_error_px": metrics["mean_error_px"],
                    "max_error_px": metrics["max_error_px"],
                    "matrix_image_to_world": H.tolist(),
                    "tracked_landmark_metrics": metrics,
                }
            )
            assignment_metrics.append({"frame": frame_idx, "points": primitive_points, "features": metrics["features"], "sources": source_counts, "fallback": False})
            continue

        clicks = []
        source_counts = {}
        for feature, mask_dir, object_id in args.mask_source:
            source_clicks = clicks_from_mask_source(
                feature,
                mask_dir,
                object_id,
                frame_idx,
                frame_shape,
                initial_fit.H,
                samples_by_feature,
                excluded_features,
                args.max_points_per_source,
                args.max_assignment_distance_px,
                args.min_component_area,
            )
            suffix = f"#{object_id}" if object_id is not None else ""
            source_counts[f"{feature}={mask_dir}{suffix}"] = len(source_clicks)
            clicks.extend(source_clicks)
        feature_count = len({click["feature"] for click in clicks})
        if len(clicks) < args.min_points or feature_count < args.min_features:
            homographies.append(
                {
                    "frame": frame_idx,
                    "points": int(initial_fit.points),
                    "inliers": int(initial_fit.inliers),
                    "mean_error_px": float(initial_fit.mean_error_ft),
                    "matrix_image_to_world": initial_fit.H.tolist(),
                    "fallback": True,
                    "fallback_reason": "not_enough_tracked_landmark_points",
                }
            )
            assignment_metrics.append(
                {
                    "frame": frame_idx,
                    "points": len(clicks),
                    "features": {feature: sum(1 for click in clicks if click["feature"] == feature) for feature in sorted({click["feature"] for click in clicks})},
                    "sources": source_counts,
                    "fallback": True,
                }
            )
            continue

        if args.refine_mode == "image_similarity":
            H, metrics = refine_frame_image_similarity(
                frame_idx,
                clicks,
                initial_fit.H,
                samples_by_feature,
                max_nfev=args.max_nfev,
                regularization=args.regularization,
            )
        else:
            H, metrics = refine_frame(
                frame_idx,
                clicks,
                initial_fit.H,
                samples_by_feature,
                max_nfev=args.max_nfev,
                regularization=args.regularization,
            )
        homographies.append(
            {
                "frame": frame_idx,
                "points": len(clicks),
                "inliers": len(clicks),
                "mean_error_px": metrics["mean_error_px"],
                "max_error_px": metrics["max_error_px"],
                "matrix_image_to_world": H.tolist(),
                "tracked_landmark_metrics": metrics,
            }
        )
        assignment_metrics.append({"frame": frame_idx, "points": len(clicks), "features": metrics["features"], "sources": source_counts, "fallback": False})
        all_generated_clicks.extend(clicks[: min(len(clicks), 80)])

    output = {
        "schema": "floor_homography_matrix_v1",
        "world_units": "feet",
        "floor": {"length_ft": FLOOR_LENGTH_FT, "width_ft": FLOOR_WIDTH_FT, "corner_radius_ft": CORNER_RADIUS_FT},
        "fit_method": f"tracked_landmark_masks_{args.refine_mode}_refined_from_initial_dynamic_homography",
        "source_initial_calibration": args.initial_calibration,
        "source_feature_clicks": args.feature_clicks,
        "tracked_mask_sources": [
            {"feature": feature, "path": str(path), "object_id": object_id}
            for feature, path, object_id in args.mask_source
        ],
        "excluded_features": sorted(excluded_features),
        "assignment": {
            "max_points_per_source": args.max_points_per_source,
            "max_assignment_distance_px": args.max_assignment_distance_px,
            "min_component_area": args.min_component_area,
            "min_points": args.min_points,
            "min_features": args.min_features,
            "refine_mode": args.refine_mode,
            "primitive_sample_stride": args.primitive_sample_stride,
            "max_single_line_rotation_deg": args.max_single_line_rotation_deg,
            "max_single_line_translation_px": args.max_single_line_translation_px,
        },
        "assignment_metrics": assignment_metrics,
        "generated_tracked_feature_click_preview": all_generated_clicks,
        "homographies": homographies,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    usable = [item for item in homographies if not item.get("fallback")]
    print(
        json.dumps(
            {
                "output": args.output,
                "frames": len(homographies),
                "tracked_refined_frames": len(usable),
                "fallback_frames": len(homographies) - len(usable),
                "median_points": float(np.median([item["points"] for item in homographies])) if homographies else 0.0,
                "median_error_px": float(np.median([item.get("mean_error_px", 0.0) for item in usable])) if usable else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
