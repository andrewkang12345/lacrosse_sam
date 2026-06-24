from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from nll_field_geometry import (
    CENTER_X_FT,
    CENTER_Y_FT,
    CORNER_RADIUS_FT,
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    circle_points,
    goal_crease_samples,
    line_points,
    rounded_outline_samples,
)
from render_birds_eye_locations import draw_floor, world_to_canvas, write_h264
from run_vggt_birds_eye import (
    fit_plane,
    original_pixels_to_vggt,
    project_to_plane_coords,
    sample_mask_pixels,
    transform_plane_to_floor,
)


TEXT_FEATURE_COLORS_BGR = {
    "green_field": (80, 180, 100),
    "green_boundary": (40, 240, 120),
    "yellow_outline": (20, 220, 255),
    "white_lines": (245, 245, 245),
}


def load_vggt_world_points(vggt_npz: Path, vggt_repo: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys.path.insert(0, str(vggt_repo.resolve()))
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    with np.load(vggt_npz) as data:
        frame_indices = data["frame_indices"].astype(int)
        depth_map = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]
        world_points = unproject_depth_map_to_point_map(depth_map, data["extrinsic"], data["intrinsic"])
    return frame_indices, depth_conf, world_points


def load_union_mask(mask_dir: Path, frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return np.zeros((height, width), dtype=bool)
    data = np.load(path)
    masks = data["masks"].astype(bool)
    if masks.size == 0:
        return np.zeros((height, width), dtype=bool)
    mask = masks.any(axis=0)
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def hsv_refine(frame_bgr: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    if not mask.any():
        return mask
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if kind == "green":
        color = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 35)
    elif kind == "yellow":
        color = (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 44) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 80)
    elif kind == "white":
        color = (hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 145)
    else:
        return mask
    refined = mask & color
    return refined if refined.sum() >= max(50, 0.03 * mask.sum()) else mask


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    kernel = np.ones((5, 5), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & ~eroded


def sample_world_points_percentile(
    world_points: np.ndarray,
    confidence: np.ndarray,
    pixels_xy: np.ndarray,
    original_shape: tuple[int, int],
    resolution: int,
    min_conf: float,
    min_conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels_xy) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    mapped = original_pixels_to_vggt(pixels_xy, original_shape, resolution)
    xs = np.clip(np.round(mapped[:, 0]).astype(np.int32), 0, resolution - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(np.int32), 0, resolution - 1)
    pts = world_points[ys, xs].astype(np.float64)
    conf = confidence[ys, xs].astype(np.float64)
    threshold = max(float(min_conf), float(np.percentile(confidence.reshape(-1), min_conf_percentile)))
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= threshold)
    return pts[valid], conf[valid]


def white_line_model_samples() -> np.ndarray:
    parts = [
        line_points(57.5, 0.0, 57.5, FLOOR_WIDTH_FT, 240),
        line_points(CENTER_X_FT, 0.0, CENTER_X_FT, FLOOR_WIDTH_FT, 240),
        line_points(142.5, 0.0, 142.5, FLOOR_WIDTH_FT, 240),
        line_points(12.0, 0.0, 12.0, FLOOR_WIDTH_FT, 180),
        line_points(188.0, 0.0, 188.0, FLOOR_WIDTH_FT, 180),
        circle_points(CENTER_X_FT, CENTER_Y_FT, 11.0, 220),
        goal_crease_samples(arc_samples=220, chord_samples=70),
    ]
    return np.concatenate(parts, axis=0).astype(np.float64)


def similarity_floor_to_plane(params: np.ndarray, floor_xy: np.ndarray) -> np.ndarray:
    scale, theta, tx, ty, reflect = params
    centered = floor_xy.astype(np.float64) - np.asarray([CENTER_X_FT, CENTER_Y_FT], dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    reflected = centered.copy()
    reflected[:, 1] *= 1.0 if reflect >= 0.0 else -1.0
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return scale * (reflected @ rot.T) + np.asarray([tx, ty], dtype=np.float64)


def similarity_plane_to_floor_matrix(params: np.ndarray) -> np.ndarray:
    src = np.asarray(
        [[0.0, 0.0], [FLOOR_LENGTH_FT, 0.0], [0.0, FLOOR_WIDTH_FT]],
        dtype=np.float32,
    )
    dst = similarity_floor_to_plane(params, src).astype(np.float32)
    H_floor_to_plane = cv2.getAffineTransform(src, dst)
    H_floor_to_plane = np.vstack([H_floor_to_plane, np.asarray([0.0, 0.0, 1.0])])
    return np.linalg.inv(H_floor_to_plane)


def inside_rounded_rink(floor_xy: np.ndarray) -> np.ndarray:
    x = floor_xy[:, 0]
    y = floor_xy[:, 1]
    inside_bbox = (x >= 0.0) & (x <= FLOOR_LENGTH_FT) & (y >= 0.0) & (y <= FLOOR_WIDTH_FT)
    inside = inside_bbox.copy()
    r = CORNER_RADIUS_FT
    corner_specs = [
        (x < r, y < r, r, r),
        (x > FLOOR_LENGTH_FT - r, y < r, FLOOR_LENGTH_FT - r, r),
        (x > FLOOR_LENGTH_FT - r, y > FLOOR_WIDTH_FT - r, FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT - r),
        (x < r, y > FLOOR_WIDTH_FT - r, r, FLOOR_WIDTH_FT - r),
    ]
    for x_region, y_region, cx, cy in corner_specs:
        region = inside_bbox & x_region & y_region
        if region.any():
            dist = np.sqrt((x[region] - cx) ** 2 + (y[region] - cy) ** 2)
            inside[region] = dist <= r
    return inside


def rounded_rink_outside_distance(floor_xy: np.ndarray, outline_tree: cKDTree) -> np.ndarray:
    if len(floor_xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    distances, _ = outline_tree.query(floor_xy, k=1)
    distances = distances.astype(np.float64)
    distances[inside_rounded_rink(floor_xy)] = 0.0
    return distances


def initial_similarity_candidates(observed_uv: np.ndarray) -> list[np.ndarray]:
    mean = np.median(observed_uv, axis=0)
    centered = observed_uv - mean[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    base_angle = float(np.arctan2(axis[1], axis[0]))
    ext = np.percentile(centered @ vh.T, [2, 98], axis=0)
    uv_span = np.maximum(ext[1] - ext[0], 1e-3)
    scales = [
        float(uv_span[0] / FLOOR_LENGTH_FT),
        float(uv_span[1] / FLOOR_WIDTH_FT),
        float(np.mean([uv_span[0] / FLOOR_LENGTH_FT, uv_span[1] / FLOOR_WIDTH_FT])),
    ]
    candidates = []
    for scale in scales:
        if not np.isfinite(scale) or scale <= 0:
            continue
        for angle in [base_angle, base_angle + np.pi, base_angle + np.pi / 2.0, base_angle - np.pi / 2.0]:
            for reflect in [1.0, -1.0]:
                candidates.append(np.asarray([scale, angle, mean[0], mean[1], reflect], dtype=np.float64))
    return candidates


def fit_rigid_field(
    feature_uv: dict[str, np.ndarray],
    regularization: float,
    fit_green_boundary_as_outline: bool,
) -> tuple[np.ndarray, dict]:
    outline = rounded_outline_samples(samples_per_segment=180).astype(np.float64)
    outline_tree_floor = cKDTree(outline)
    white = white_line_model_samples()
    all_observed = np.concatenate([v for v in feature_uv.values() if len(v)], axis=0)
    if len(all_observed) < 80:
        raise RuntimeError("Not enough SAM3/VGGT field-mask points to fit the synthetic field.")
    anchor_sets = [
        feature_uv[name]
        for name in ["yellow_outline", "white_lines"]
        if name in feature_uv and len(feature_uv[name])
    ]
    initial_observed = np.concatenate(anchor_sets, axis=0) if anchor_sets else all_observed

    reg_points = np.concatenate([outline, white[:: max(1, len(white) // 450)]], axis=0)
    best = None

    def residual_fn(p_free: np.ndarray, reflect_sign: float, initial_model_uv: np.ndarray) -> np.ndarray:
        params = np.asarray([p_free[0], p_free[1], p_free[2], p_free[3], reflect_sign], dtype=np.float64)
        values: list[float] = []
        if "yellow_outline" in feature_uv and len(feature_uv["yellow_outline"]):
            model = similarity_floor_to_plane(params, outline)
            tree = cKDTree(model)
            distances, _ = tree.query(feature_uv["yellow_outline"], k=1)
            values.extend((distances * 1.25).tolist())
        if fit_green_boundary_as_outline and "green_boundary" in feature_uv and len(feature_uv["green_boundary"]):
            model = similarity_floor_to_plane(params, outline)
            tree = cKDTree(model)
            distances, _ = tree.query(feature_uv["green_boundary"], k=1)
            values.extend((distances * 0.45).tolist())
        if "white_lines" in feature_uv and len(feature_uv["white_lines"]):
            model = similarity_floor_to_plane(params, white)
            tree = cKDTree(model)
            distances, _ = tree.query(feature_uv["white_lines"], k=1)
            values.extend((distances * 0.85).tolist())
        if "green_field" in feature_uv and len(feature_uv["green_field"]):
            H = similarity_plane_to_floor_matrix(params)
            floor = transform_plane_to_floor(H, feature_uv["green_field"])
            outside = rounded_rink_outside_distance(floor, outline_tree_floor)
            values.extend((outside * max(abs(params[0]), 1e-6) * 0.8).tolist())
        if regularization > 0:
            model_uv = similarity_floor_to_plane(params, reg_points)
            values.extend(((model_uv - initial_model_uv) * regularization).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float64)

    for candidate in initial_similarity_candidates(initial_observed):
        reflect_sign = 1.0 if candidate[4] >= 0.0 else -1.0
        initial_model_uv = similarity_floor_to_plane(candidate, reg_points)
        result = least_squares(
            lambda p: residual_fn(p, reflect_sign, initial_model_uv),
            candidate[:4],
            loss="soft_l1",
            f_scale=0.08,
            max_nfev=1200,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        err = residual_fn(result.x, reflect_sign, initial_model_uv)
        score = float(np.median(np.abs(err))) if len(err) else float("inf")
        params = np.asarray([result.x[0], result.x[1], result.x[2], result.x[3], reflect_sign], dtype=np.float64)
        if best is None or score < best[0]:
            best = (score, params, result)
    if best is None:
        raise RuntimeError("Field similarity optimizer failed to produce a candidate.")

    score, params, result = best
    H_plane_to_floor = similarity_plane_to_floor_matrix(params)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "median_abs_residual_plane_units": float(score),
        "nfev": int(result.nfev),
        "scale_plane_units_per_ft": float(params[0]),
        "theta_rad": float(params[1]),
        "translation_uv": [float(params[2]), float(params[3])],
        "reflected_y": bool(params[4] < 0.0),
        "feature_points": {name: int(len(points)) for name, points in feature_uv.items()},
        "fit_model": "rigid_similarity_no_shape_morph_partial_green_interior",
        "green_boundary_fit_as_outline": bool(fit_green_boundary_as_outline),
    }
    return H_plane_to_floor, metrics


def downsample_feature_uv(feature_uv: dict[str, np.ndarray], max_points: int) -> dict[str, np.ndarray]:
    if max_points <= 0:
        return feature_uv
    output = {}
    for name, points in feature_uv.items():
        if len(points) <= max_points:
            output[name] = points
            continue
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        idx = rng.choice(len(points), size=max_points, replace=False)
        output[name] = points[idx]
    return output


def collect_text_feature_points(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    frame_indices, depth_conf, world_points = load_vggt_world_points(Path(args.vggt_npz), Path(args.vggt_repo))
    resolution = int(world_points.shape[1])
    features_3d: dict[str, list[np.ndarray]] = {"green_field": [], "green_boundary": [], "yellow_outline": [], "white_lines": []}
    per_frame_counts = []

    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        if frame_idx >= len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        shape = frame.shape[:2]
        green = hsv_refine(frame, load_union_mask(Path(args.green_mask_dir), frame_idx, shape), "green")
        yellow = hsv_refine(frame, load_union_mask(Path(args.yellow_mask_dir), frame_idx, shape), "yellow")
        white = hsv_refine(frame, load_union_mask(Path(args.white_mask_dir), frame_idx, shape), "white")
        masks = {
            "green_field": green,
            "green_boundary": mask_boundary(green),
            "yellow_outline": yellow,
            "white_lines": white,
        }
        counts = {"frame": int(frame_idx)}
        for name, mask in masks.items():
            pixels = sample_mask_pixels(mask, args.max_points_per_feature, seed=frame_idx * 1009 + len(name))
            pts3d, _ = sample_world_points_percentile(
                world_points[local_idx],
                depth_conf[local_idx],
                pixels,
                shape,
                resolution,
                args.min_depth_conf,
                args.min_depth_conf_percentile,
            )
            counts[name] = int(len(pts3d))
            if len(pts3d):
                features_3d[name].append(pts3d)
        per_frame_counts.append(counts)

    merged_3d = {name: np.concatenate(values, axis=0) for name, values in features_3d.items() if values}
    plane_sources = []
    for name in ["green_field", "green_boundary", "yellow_outline", "white_lines"]:
        if name in merged_3d:
            plane_sources.append(merged_3d[name])
    if not plane_sources:
        raise RuntimeError("No VGGT 3D points were sampled from the SAM3 text masks.")
    all_plane_points = np.concatenate(plane_sources, axis=0)
    center, basis_u, basis_v, inliers = fit_plane(all_plane_points)
    feature_uv = {name: project_to_plane_coords(points, center, basis_u, basis_v) for name, points in merged_3d.items()}
    original_counts = {name: int(len(points)) for name, points in feature_uv.items()}
    feature_uv = downsample_feature_uv(feature_uv, args.max_total_points_per_feature)
    metadata = {
        "frame_indices": frame_indices.tolist(),
        "vggt_resolution": resolution,
        "floor_plane": {
            "center": center.tolist(),
            "basis_u": basis_u.tolist(),
            "basis_v": basis_v.tolist(),
            "floor_points": int(len(all_plane_points)),
            "plane_inliers": int(np.asarray(inliers).sum()),
        },
        "feature_points_before_optimization_downsample": original_counts,
        "feature_points_after_optimization_downsample": {name: int(len(points)) for name, points in feature_uv.items()},
        "per_frame_counts": per_frame_counts,
    }
    return feature_uv, metadata


def render_debug_video(feature_uv: dict[str, np.ndarray], H_plane_to_floor: np.ndarray, output_video: Path, fps: float, width: int, height: int, margin: int) -> dict:
    canvas = draw_floor(width, height, margin)
    counts = {}
    for name, uv in feature_uv.items():
        floor = transform_plane_to_floor(H_plane_to_floor, uv)
        valid = np.isfinite(floor).all(axis=1)
        valid &= (floor[:, 0] >= -15.0) & (floor[:, 0] <= FLOOR_LENGTH_FT + 15.0)
        valid &= (floor[:, 1] >= -15.0) & (floor[:, 1] <= FLOOR_WIDTH_FT + 15.0)
        floor = floor[valid]
        counts[name] = int(len(floor))
        color = TEXT_FEATURE_COLORS_BGR.get(name, (255, 255, 255))
        stride = max(1, len(floor) // 9000)
        for x, y in floor[::stride]:
            px, py = world_to_canvas(float(np.clip(x, 0.0, FLOOR_LENGTH_FT)), float(np.clip(y, 0.0, FLOOR_WIDTH_FT)), width, height, margin)
            cv2.circle(canvas, (px, py), 2, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "SAM3 text masks projected through VGGT field fit", (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    write_h264([canvas for _ in range(max(1, int(round(fps))))], output_video, fps=fps)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-npz", required=True)
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--white-mask-dir", dest="white_mask_dir", required=True)
    parser.add_argument("--yellow-mask-dir", dest="yellow_mask_dir", required=True)
    parser.add_argument("--green-mask-dir", dest="green_mask_dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-depth-conf", type=float, default=1.0)
    parser.add_argument("--min-depth-conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-points-per-feature", type=int, default=1200)
    parser.add_argument("--max-total-points-per-feature", type=int, default=2500)
    parser.add_argument("--alignment-regularization", type=float, default=0.015)
    parser.add_argument(
        "--fit-green-boundary-as-outline",
        action="store_true",
        help="Treat the visible green-mask boundary as the rink outline. Off by default because broadcast occlusions create partial-field boundaries.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=54)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_uv, metadata = collect_text_feature_points(args)
    H_plane_to_floor, alignment_metrics = fit_rigid_field(
        feature_uv,
        args.alignment_regularization,
        fit_green_boundary_as_outline=args.fit_green_boundary_as_outline,
    )
    debug_video = output_dir / "sam3_text_field_masks_projected_to_synthetic_field_h264.mp4"
    projected_counts = render_debug_video(feature_uv, H_plane_to_floor, debug_video, args.fps, args.width, args.height, args.margin)
    output_json = output_dir / "field_fit_vggt_sam3_text_masks.json"
    result = {
        "schema": "vggt_sam3_text_field_fit_v1",
        "vggt_npz": args.vggt_npz,
        "frames_dir": args.frames_dir,
        "mask_dirs": {
            "white_lines": args.white_mask_dir,
            "yellow_outline": args.yellow_mask_dir,
            "green_field": args.green_mask_dir,
        },
        "vggt_resolution": metadata["vggt_resolution"],
        "floor_plane": metadata["floor_plane"],
        "feature_points_before_optimization_downsample": metadata["feature_points_before_optimization_downsample"],
        "feature_points_after_optimization_downsample": metadata["feature_points_after_optimization_downsample"],
        "plane_to_floor_homography": H_plane_to_floor.tolist(),
        "alignment_metrics": alignment_metrics,
        "projected_debug_counts": projected_counts,
        "per_frame_counts": metadata["per_frame_counts"],
    }
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_json": str(output_json), "debug_video": str(debug_video), "alignment_metrics": alignment_metrics}, indent=2))


if __name__ == "__main__":
    main()
