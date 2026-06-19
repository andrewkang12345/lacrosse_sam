from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_birds_eye_locations import FLOOR_LENGTH_FT, FLOOR_WIDTH_FT, CORNER_RADIUS_FT, fit_homographies, nearest_fit


def load_gray(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read frame: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def feature_mask(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    mask = np.zeros_like(gray, dtype=np.uint8)
    # Avoid the fixed broadcast scoreboard and most spectator area. Keep the
    # playing surface, boards, and floor logos where camera motion is visible.
    y1 = int(round(height * 0.16))
    y2 = int(round(height * 0.95))
    mask[y1:y2, :] = 255
    return mask


def estimate_pair_homography(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, dict]:
    orb = cv2.ORB_create(nfeatures=7000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    prev_mask = feature_mask(prev_gray)
    curr_mask = feature_mask(curr_gray)
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, prev_mask)
    kp_curr, des_curr = orb.detectAndCompute(curr_gray, curr_mask)
    if des_prev is None or des_curr is None or len(kp_prev) < 12 or len(kp_curr) < 12:
        return np.eye(3, dtype=np.float64), {"ok": False, "reason": "not_enough_features", "matches": 0, "inliers": 0}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(des_curr, des_prev, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance:
            good.append(m)
    if len(good) < 12:
        return np.eye(3, dtype=np.float64), {"ok": False, "reason": "not_enough_matches", "matches": len(good), "inliers": 0}

    src = np.float32([kp_curr[m.queryIdx].pt for m in good])
    dst = np.float32([kp_prev[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995)
    if H is None or mask is None:
        return np.eye(3, dtype=np.float64), {"ok": False, "reason": "homography_failed", "matches": len(good), "inliers": 0}
    inliers = int(mask.reshape(-1).sum())
    if inliers < 16:
        return np.eye(3, dtype=np.float64), {"ok": False, "reason": "too_few_inliers", "matches": len(good), "inliers": inliers}

    H = H.astype(np.float64)
    H /= H[2, 2]
    corners = np.asarray([[[0.0, 0.0], [1279.0, 0.0], [1279.0, 719.0], [0.0, 719.0]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return np.eye(3, dtype=np.float64), {"ok": False, "reason": "nonfinite", "matches": len(good), "inliers": inliers}
    shift = np.linalg.norm(warped - corners.reshape(-1, 2), axis=1).mean()
    if shift > 180:
        return np.eye(3, dtype=np.float64), {"ok": False, "reason": "excessive_shift", "matches": len(good), "inliers": inliers, "mean_corner_shift_px": float(shift)}

    return H, {
        "ok": True,
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(good), 1)),
        "mean_corner_shift_px": float(shift),
    }


def compose_to_reference(pair_to_prev: list[np.ndarray], reference_frame: int) -> list[np.ndarray]:
    n = len(pair_to_prev)
    to_ref = [np.eye(3, dtype=np.float64) for _ in range(n)]

    # pair_to_prev[i] maps frame i -> frame i-1.
    for idx in range(reference_frame + 1, n):
        to_ref[idx] = to_ref[idx - 1] @ pair_to_prev[idx]
        to_ref[idx] /= to_ref[idx][2, 2]

    for idx in range(reference_frame - 1, -1, -1):
        try:
            curr_to_next = np.linalg.inv(pair_to_prev[idx + 1])
        except np.linalg.LinAlgError:
            curr_to_next = np.eye(3, dtype=np.float64)
        to_ref[idx] = to_ref[idx + 1] @ curr_to_next
        to_ref[idx] /= to_ref[idx][2, 2]

    return to_ref


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--base-calibration", default="outputs/floor_homography_curve_refined.json")
    parser.add_argument("--output", default="outputs/floor_homography_dynamic.json")
    parser.add_argument("--reference-frame", type=int, default=77)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames found in {args.frames_dir}")
    if not 0 <= args.reference_frame < len(frame_paths):
        raise ValueError(f"reference frame {args.reference_frame} outside 0..{len(frame_paths)-1}")

    base = json.loads(Path(args.base_calibration).read_text())
    base_fit = nearest_fit(fit_homographies(base, ransac_threshold_ft=3.0), args.reference_frame)
    base_H_ref_to_world = base_fit.H

    grays = [load_gray(path) for path in frame_paths]
    pair_to_prev = [np.eye(3, dtype=np.float64) for _ in frame_paths]
    pair_metrics = [{"ok": True, "reference": True, "matches": 0, "inliers": 0}]
    for idx in range(1, len(grays)):
        H, metrics = estimate_pair_homography(grays[idx - 1], grays[idx])
        pair_to_prev[idx] = H
        metrics = {"frame": idx, "maps": f"{idx}->{idx-1}", **metrics}
        pair_metrics.append(metrics)

    to_ref = compose_to_reference(pair_to_prev, args.reference_frame)
    homographies = []
    for idx, H_img_to_ref in enumerate(to_ref):
        H_img_to_world = base_H_ref_to_world @ H_img_to_ref
        H_img_to_world /= H_img_to_world[2, 2]
        homographies.append(
            {
                "frame": idx,
                "points": int(base_fit.points),
                "inliers": int(base_fit.inliers),
                "mean_error_px": float(base_fit.mean_error_ft),
                "matrix_image_to_world": H_img_to_world.tolist(),
            }
        )

    output = {
        "schema": "floor_homography_matrix_v1",
        "world_units": "feet",
        "floor": {"length_ft": FLOOR_LENGTH_FT, "width_ft": FLOOR_WIDTH_FT, "corner_radius_ft": CORNER_RADIUS_FT},
        "source_base_calibration": args.base_calibration,
        "source_frames_dir": args.frames_dir,
        "reference_frame": args.reference_frame,
        "fit_method": "pairwise_orb_camera_motion_composed_with_reference_floor_homography",
        "pairwise_registration": pair_metrics,
        "homographies": homographies,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    failed = [m for m in pair_metrics[1:] if not m.get("ok")]
    print(
        json.dumps(
            {
                "output": args.output,
                "frames": len(homographies),
                "reference_frame": args.reference_frame,
                "failed_pairwise_registrations": len(failed),
                "median_inliers": float(np.median([m.get("inliers", 0) for m in pair_metrics[1:]])) if len(pair_metrics) > 1 else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
