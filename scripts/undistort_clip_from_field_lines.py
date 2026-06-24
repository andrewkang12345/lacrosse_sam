from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar


LINE_OBJECTS = {
    "left_restraining_line": 1,
    "right_restraining_line": 2,
    "midfield_line": 3,
}


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_npz_mask(mask_dir: Path, frame_idx: int, object_id: int, shape: tuple[int, int]) -> np.ndarray:
    path = mask_dir / f"{frame_idx:08d}.npz"
    height, width = shape
    if not path.exists():
        return np.zeros((height, width), dtype=bool)
    data = np.load(path)
    ids = [int(v) for v in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    if object_id not in ids:
        return np.zeros((height, width), dtype=bool)
    mask = masks[ids.index(object_id)]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def sample_mask_points(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=max_points, replace=False)
    return pts[idx]


def collect_line_groups(
    frames_dir: Path,
    mask_dir: Path,
    frame_stride: int,
    max_points_per_line: int,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frame_*.jpg files found in {frames_dir}")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {frame_paths[0]}")
    shape = first.shape[:2]
    groups = []
    for path in frame_paths[:: max(1, frame_stride)]:
        frame_idx = int(path.stem.split("_")[-1])
        for name, object_id in LINE_OBJECTS.items():
            mask = load_npz_mask(mask_dir, frame_idx, object_id, shape)
            pts = sample_mask_points(mask, max_points_per_line, seed=frame_idx * 1009 + object_id)
            if len(pts) >= 30:
                groups.append(pts)
    if len(groups) < 4:
        raise RuntimeError(f"Not enough SAM2 line masks to estimate distortion; got {len(groups)} line groups.")
    return groups, shape


def line_rms(points: np.ndarray) -> float:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    distances = np.abs((points - center) @ normal)
    return float(np.sqrt(np.mean(np.minimum(distances, np.percentile(distances, 90)) ** 2)))


def undistort_points(points: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def estimate_k1(groups: list[np.ndarray], image_shape: tuple[int, int], focal_px: float, k1_bound: float, min_improvement_px: float = 0.03) -> tuple[float, dict]:
    height, width = image_shape
    K = np.asarray([[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def objective(k1: float) -> float:
        dist = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        values = []
        for pts in groups:
            und = undistort_points(pts, K, dist)
            values.append(line_rms(und))
        return float(np.mean(values))

    baseline = objective(0.0)
    result = minimize_scalar(objective, bounds=(-k1_bound, k1_bound), method="bounded", options={"xatol": 1e-5})
    k1 = float(result.x)
    best = float(result.fun)
    if baseline - best < min_improvement_px:
        k1 = 0.0
        best = baseline
    metrics = {
        "baseline_line_rms_px": baseline,
        "best_line_rms_px": best,
        "line_rms_improvement_px": baseline - best,
        "line_groups": len(groups),
        "focal_px": focal_px,
        "k1_bound": k1_bound,
        "optimizer_success": bool(result.success),
        "optimizer_fun": float(result.fun),
        "min_improvement_px": float(min_improvement_px),
    }
    return k1, metrics


def remap_npz_masks(input_dir: Path, output_dir: Path, map_x: np.ndarray, map_y: np.ndarray, shape: tuple[int, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.npz"):
        old.unlink()
    height, width = shape
    for path in sorted(input_dir.glob("*.npz")):
        data = np.load(path)
        ids = data["object_ids"].astype(np.int32)
        masks = data["masks"].astype(np.uint8)
        remapped = []
        for mask in masks:
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            remapped.append(cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0)
        stack = np.stack(remapped, axis=0) if remapped else np.zeros((0, height, width), dtype=bool)
        np.savez_compressed(output_dir / path.name, object_ids=ids, masks=stack.astype(np.uint8))


def remap_png_masks(input_dir: Path, output_dir: Path, map_x: np.ndarray, map_y: np.ndarray, shape: tuple[int, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()
    height, width = shape
    for path in sorted(input_dir.glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        out = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(output_dir / path.name), out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate mild radial lens undistortion from SAM2 field-line masks and remap clip assets.")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--line-mask-dir", default="outputs/sam2/floor_features/sam2_floor_feature_instance_masks_with_outline")
    parser.add_argument("--output-frames-dir", default="data/frames_10fps_undistorted")
    parser.add_argument("--calibration-json", default="outputs/vggt/undistorted/lens_undistortion.json")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-points-per-line", type=int, default=600)
    parser.add_argument("--focal-px", type=float, default=1475.0)
    parser.add_argument("--k1-bound", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--npz-mask-map", action="append", default=[], help="input_dir:output_dir pair to remap instance masks.")
    parser.add_argument("--png-mask-map", action="append", default=[], help="input_dir:output_dir pair to remap label masks.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    groups, shape = collect_line_groups(frames_dir, Path(args.line_mask_dir), args.frame_stride, args.max_points_per_line)
    height, width = shape
    k1, metrics = estimate_k1(groups, shape, args.focal_px, args.k1_bound)
    K = np.asarray([[args.focal_px, 0.0, width / 2.0], [0.0, args.focal_px, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), args.alpha, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (width, height), cv2.CV_32FC1)

    out_frames = Path(args.output_frames_dir)
    out_frames.mkdir(parents=True, exist_ok=True)
    for old in out_frames.glob("frame_*.jpg"):
        old.unlink()
    for path in sorted_frame_paths(frames_dir):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        und = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cv2.imwrite(str(out_frames / path.name), und, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    for pair in args.npz_mask_map:
        src, dst = pair.split(":", 1)
        remap_npz_masks(Path(src), Path(dst), map_x, map_y, shape)
    for pair in args.png_mask_map:
        src, dst = pair.split(":", 1)
        remap_png_masks(Path(src), Path(dst), map_x, map_y, shape)

    out_json = Path(args.calibration_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "field_line_radial_undistortion_v1",
        "frames_dir": args.frames_dir,
        "output_frames_dir": args.output_frames_dir,
        "line_mask_dir": args.line_mask_dir,
        "image_size": [width, height],
        "camera_matrix": K.tolist(),
        "new_camera_matrix": new_K.tolist(),
        "dist_coeffs": dist.tolist(),
        "roi": [int(v) for v in roi],
        "metrics": metrics,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"calibration": str(out_json), "output_frames_dir": str(out_frames), "k1": k1, **metrics}, indent=2))


if __name__ == "__main__":
    main()
