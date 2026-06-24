from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from undistort_clip_from_field_lines import (
    estimate_k1,
    remap_npz_masks,
    remap_png_masks,
    sample_mask_points,
    sorted_frame_paths,
)


def load_union_masks(path: Path, shape: tuple[int, int]) -> list[np.ndarray]:
    height, width = shape
    if not path.exists():
        return []
    data = np.load(path)
    masks = data["masks"].astype(bool)
    output = []
    for mask in masks:
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        output.append(mask)
    return output


def hsv_white_refine(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 80) & (hsv[:, :, 2] >= 140)
    refined = mask & white
    return refined if refined.sum() >= max(30, 0.02 * mask.sum()) else mask


def component_groups(mask: np.ndarray, min_area: int, max_area: int, min_aspect: float) -> list[np.ndarray]:
    if not mask.any():
        return []
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    groups = []
    for label in range(1, nlabels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(1, min(w, h))
        if aspect < min_aspect:
            continue
        ys, xs = np.where(labels == label)
        points = np.column_stack([xs, ys]).astype(np.float32)
        if len(points) >= min_area:
            groups.append(points)
    return groups


def collect_line_groups(
    frames_dir: Path,
    white_mask_dir: Path,
    frame_stride: int,
    max_points_per_component: int,
    min_component_area: int,
    max_component_area: int,
    min_component_aspect: float,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    frame_paths = sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frame_*.jpg files found in {frames_dir}")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read {frame_paths[0]}")
    shape = first.shape[:2]
    groups = []
    for local_idx, frame_path in enumerate(frame_paths[:: max(1, frame_stride)]):
        frame_idx = local_idx * max(1, frame_stride)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        masks = load_union_masks(white_mask_dir / f"{frame_idx:08d}.npz", shape)
        for mask_idx, mask in enumerate(masks):
            refined = hsv_white_refine(frame, mask)
            for comp_idx, points in enumerate(component_groups(refined, min_component_area, max_component_area, min_component_aspect)):
                sampled = points
                if len(sampled) > max_points_per_component:
                    rng = np.random.default_rng(frame_idx * 1009 + mask_idx * 97 + comp_idx)
                    sampled = points[rng.choice(len(points), size=max_points_per_component, replace=False)]
                if len(sampled) >= min_component_area:
                    groups.append(sampled.astype(np.float32))
    if len(groups) < 4:
        raise RuntimeError(f"Not enough text white-line components to estimate distortion; got {len(groups)} groups.")
    return groups, shape


def parse_map_pair(pair: str) -> tuple[Path, Path]:
    src, dst = pair.split(":", 1)
    return Path(src), Path(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate mild radial undistortion from SAM3 text white-line masks and remap clip assets.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--white-mask-dir", required=True)
    parser.add_argument("--output-frames-dir", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-points-per-component", type=int, default=500)
    parser.add_argument("--min-component-area", type=int, default=30)
    parser.add_argument("--max-component-area", type=int, default=50000)
    parser.add_argument("--min-component-aspect", type=float, default=2.0)
    parser.add_argument("--focal-px", type=float, default=1475.0)
    parser.add_argument("--k1-bound", type=float, default=0.35)
    parser.add_argument("--min-improvement-px", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--npz-mask-map", action="append", default=[])
    parser.add_argument("--png-mask-map", action="append", default=[])
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    groups, shape = collect_line_groups(
        frames_dir,
        Path(args.white_mask_dir),
        args.frame_stride,
        args.max_points_per_component,
        args.min_component_area,
        args.max_component_area,
        args.min_component_aspect,
    )
    height, width = shape
    k1, metrics = estimate_k1(groups, shape, args.focal_px, args.k1_bound, args.min_improvement_px)
    K = np.asarray([[args.focal_px, 0.0, width / 2.0], [0.0, args.focal_px, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), args.alpha, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (width, height), cv2.CV_32FC1)

    output_frames_dir = Path(args.output_frames_dir)
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    for old in output_frames_dir.glob("frame_*.jpg"):
        old.unlink()
    for frame_path in sorted_frame_paths(frames_dir):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        undistorted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cv2.imwrite(str(output_frames_dir / frame_path.name), undistorted, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    for pair in args.npz_mask_map:
        src, dst = parse_map_pair(pair)
        remap_npz_masks(src, dst, map_x, map_y, shape)
    for pair in args.png_mask_map:
        src, dst = parse_map_pair(pair)
        remap_png_masks(src, dst, map_x, map_y, shape)

    output_json = Path(args.calibration_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "text_white_line_radial_undistortion_v1",
        "frames_dir": args.frames_dir,
        "output_frames_dir": args.output_frames_dir,
        "white_mask_dir": args.white_mask_dir,
        "image_size": [width, height],
        "camera_matrix": K.tolist(),
        "new_camera_matrix": new_K.tolist(),
        "dist_coeffs": dist.tolist(),
        "roi": [int(v) for v in roi],
        "metrics": metrics,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"calibration": str(output_json), "output_frames_dir": str(output_frames_dir), "k1": k1, **metrics}, indent=2))


if __name__ == "__main__":
    main()
