from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


FLOOR_LENGTH_FT = 200.0
FLOOR_WIDTH_FT = 85.0
CORNER_RADIUS_FT = 22.667
DEFAULT_COLORS = [
    [255, 80, 40],
    [70, 170, 255],
    [245, 210, 55],
    [180, 180, 180],
]

LANDMARK_WORLD_POINTS = [
    (100.0, 42.5),
    (100.0, 0.0),
    (100.0, 85.0),
    (57.5, 0.0),
    (57.5, 85.0),
    (142.5, 0.0),
    (142.5, 85.0),
    (12.0, 42.5),
    (188.0, 42.5),
    (12.0, 40.125),
    (12.0, 44.875),
    (188.0, 40.125),
    (188.0, 44.875),
    (21.25, 42.5),
    (12.0, 33.25),
    (12.0, 51.75),
    (178.75, 42.5),
    (188.0, 33.25),
    (188.0, 51.75),
    (42.5, 15.0),
    (42.5, 70.0),
    (157.5, 15.0),
    (157.5, 70.0),
    (22.667, 0.0),
    (177.333, 0.0),
    (22.667, 85.0),
    (177.333, 85.0),
    (0.0, 42.5),
    (200.0, 42.5),
]


@dataclass
class HomographyFit:
    frame: int
    H: np.ndarray
    points: int
    inliers: int
    mean_error_ft: float


def load_instance_mask(mask_dir: Path, frame_idx: int, object_ids: list[int], det_idx: int, height: int, width: int) -> np.ndarray | None:
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    mask_ids = [int(v) for v in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    obj_id = int(object_ids[det_idx])
    if obj_id in mask_ids:
        mask_idx = mask_ids.index(obj_id)
    else:
        mask_idx = det_idx
    if mask_idx < masks.shape[0]:
        return masks[mask_idx]
    return None


def floor_point_from_detection(
    box: list[float],
    mask: np.ndarray | None,
    frame_shape: tuple[int, int],
) -> tuple[float, float]:
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        low = ys >= np.quantile(ys, 0.90)
        if low.any():
            return float(np.median(xs[low])), float(np.max(ys[low]))
        return float(np.median(xs)), float(np.max(ys))
    height, width = frame_shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x = np.clip((x1 + x2) * 0.5, 0, width - 1)
    y = np.clip(y2, 0, height - 1)
    return float(x), float(y)


def fit_homographies(calibration: dict, ransac_threshold_ft: float) -> list[HomographyFit]:
    if calibration.get("schema") == "floor_homography_matrix_v1":
        fits = []
        for item in calibration.get("homographies", []):
            fits.append(
                HomographyFit(
                    frame=int(item["frame"]),
                    H=np.asarray(item["matrix_image_to_world"], dtype=np.float64),
                    points=int(item.get("points", 0)),
                    inliers=int(item.get("inliers", item.get("points", 0))),
                    mean_error_ft=float(item.get("mean_error_ft", item.get("mean_error_px", 0.0))),
                )
            )
        if not fits:
            raise RuntimeError("No homography matrices found in calibration.")
        return fits

    if calibration.get("schema") == "floor_unlabeled_clicks_v1" or (
        calibration.get("clicks") and "world" not in calibration["clicks"][0]
    ):
        return fit_unlabeled_homographies(calibration, ransac_threshold_ft)

    by_frame: dict[int, list[dict]] = {}
    for click in calibration.get("clicks", []):
        by_frame.setdefault(int(click["frame"]), []).append(click)

    fits: list[HomographyFit] = []
    for frame, clicks in sorted(by_frame.items()):
        if len(clicks) < 4:
            continue
        image = np.asarray([[c["image"]["x"], c["image"]["y"]] for c in clicks], dtype=np.float32)
        world = np.asarray([[c["world"]["x"], c["world"]["y"]] for c in clicks], dtype=np.float32)
        H, inlier_mask = cv2.findHomography(image, world, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold_ft)
        if H is None:
            continue
        projected = cv2.perspectiveTransform(image.reshape(1, -1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(projected - world, axis=1)
        if inlier_mask is not None:
            valid = inlier_mask.reshape(-1).astype(bool)
            mean_error = float(errors[valid].mean()) if valid.any() else float(errors.mean())
            inliers = int(valid.sum())
        else:
            mean_error = float(errors.mean())
            inliers = len(clicks)
        fits.append(HomographyFit(frame=frame, H=H, points=len(clicks), inliers=inliers, mean_error_ft=mean_error))
    if not fits:
        raise RuntimeError("No usable homography fits. Add at least 4 floor landmark clicks on one frame.")
    return fits


def dense_floor_model(step_ft: float = 4.0) -> np.ndarray:
    points: list[tuple[float, float]] = list(LANDMARK_WORLD_POINTS)
    for x in [12.0, 57.5, 100.0, 142.5, 188.0]:
        for y in np.arange(0.0, FLOOR_WIDTH_FT + 0.001, step_ft):
            points.append((x, float(y)))
    for y in [0.0, 85.0]:
        for x in np.arange(CORNER_RADIUS_FT, FLOOR_LENGTH_FT - CORNER_RADIUS_FT + 0.001, step_ft):
            points.append((float(x), y))
    for cx, cy, radius in [(100.0, 42.5, 11.0), (12.0, 42.5, 9.25), (188.0, 42.5, 9.25)]:
        for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            points.append((float(cx + radius * np.cos(angle)), float(cy + radius * np.sin(angle))))
    boundary = rounded_floor_points(samples_per_corner=28)
    points.extend((float(x), float(y)) for x, y in boundary)
    unique = sorted({(round(x, 3), round(y, 3)) for x, y in points})
    return np.asarray(unique, dtype=np.float32)


def homography_ok(H: np.ndarray) -> bool:
    if H is None or not np.isfinite(H).all():
        return False
    test = np.asarray([[[0.0, 0.0], [1279.0, 0.0], [1279.0, 719.0], [0.0, 719.0]]], dtype=np.float32)
    try:
        world = cv2.perspectiveTransform(test, H).reshape(-1, 2)
    except cv2.error:
        return False
    span = world.max(axis=0) - world.min(axis=0)
    return bool(np.isfinite(world).all() and 20.0 <= span[0] <= 500.0 and 10.0 <= span[1] <= 300.0)


def nearest_model_distances(world_points: np.ndarray, model_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = world_points[:, None, :] - model_points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    nearest = dist.argmin(axis=1)
    return dist[np.arange(len(world_points)), nearest], nearest


def score_unlabeled_homography(
    H: np.ndarray,
    image_points: np.ndarray,
    model_points: np.ndarray,
    inlier_threshold_ft: float,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2).astype(np.float32), H).reshape(-1, 2)
    in_bounds = (
        (projected[:, 0] >= -8.0)
        & (projected[:, 0] <= FLOOR_LENGTH_FT + 8.0)
        & (projected[:, 1] >= -8.0)
        & (projected[:, 1] <= FLOOR_WIDTH_FT + 8.0)
    )
    distances, nearest = nearest_model_distances(projected, model_points)
    inliers = (distances <= inlier_threshold_ft) & in_bounds
    unique_matches = len(set(int(idx) for idx in nearest[inliers].tolist()))
    if inliers.any():
        mean_error = float(distances[inliers].mean())
    else:
        mean_error = float("inf")
    return int(unique_matches), mean_error, inliers, nearest


def refine_unlabeled_homography(
    H: np.ndarray,
    image_points: np.ndarray,
    model_points: np.ndarray,
    inlier_threshold_ft: float,
) -> tuple[np.ndarray, int, float]:
    best_H = H
    best_inliers = 0
    best_error = float("inf")
    for _ in range(4):
        inlier_count, mean_error, inliers, nearest = score_unlabeled_homography(best_H, image_points, model_points, inlier_threshold_ft)
        if inlier_count < 4:
            break
        inlier_indices = np.where(inliers)[0]
        keep_indices = []
        seen: set[int] = set()
        projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2).astype(np.float32), best_H).reshape(-1, 2)
        distances = np.linalg.norm(projected - model_points[nearest], axis=1)
        for idx in sorted(inlier_indices, key=lambda i: float(distances[i])):
            match = int(nearest[idx])
            if match in seen:
                continue
            keep_indices.append(idx)
            seen.add(match)
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        snapped_world = model_points[nearest[keep_indices]]
        refined, mask = cv2.findHomography(image_points[keep_indices], snapped_world, method=cv2.RANSAC, ransacReprojThreshold=inlier_threshold_ft)
        if refined is None or not homography_ok(refined):
            break
        next_count, next_error, _, _ = score_unlabeled_homography(refined, image_points, model_points, inlier_threshold_ft)
        if next_count > best_inliers or (next_count == best_inliers and next_error < best_error):
            best_H = refined
            best_inliers = next_count
            best_error = next_error
        else:
            break
    if best_inliers == 0:
        best_inliers, best_error, _, _ = score_unlabeled_homography(best_H, image_points, model_points, inlier_threshold_ft)
    return best_H, best_inliers, best_error


def fit_one_unlabeled_frame(
    image_points: np.ndarray,
    model_points: np.ndarray,
    landmark_points: np.ndarray,
    ransac_threshold_ft: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    if len(image_points) < 4:
        raise RuntimeError("Unlabeled homography fitting needs at least 4 clicks on a frame.")
    rng = np.random.default_rng(seed)
    best_H = None
    best_inliers = -1
    best_error = float("inf")
    world_seed = np.concatenate([landmark_points, model_points[:: max(1, len(model_points) // 80)]], axis=0)

    for _ in range(iterations):
        image_idx = rng.choice(len(image_points), size=4, replace=False)
        world_idx = rng.choice(len(world_seed), size=4, replace=False)
        src = image_points[image_idx].astype(np.float32)
        dst_base = world_seed[world_idx].astype(np.float32)
        if cv2.contourArea(cv2.convexHull(src.reshape(-1, 1, 2))) < 20.0:
            continue
        if cv2.contourArea(cv2.convexHull(dst_base.reshape(-1, 1, 2))) < 1.0:
            continue
        for perm in ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)):
            dst = dst_base[list(perm)]
            H = cv2.getPerspectiveTransform(src, dst)
            if not homography_ok(H):
                continue
            inliers, error, _, _ = score_unlabeled_homography(H, image_points, model_points, ransac_threshold_ft)
            if inliers > best_inliers or (inliers == best_inliers and error < best_error):
                best_H = H
                best_inliers = inliers
                best_error = error

    if best_H is None:
        raise RuntimeError("Could not fit unlabeled homography from clicks. Add more distinct landmarks.")
    return refine_unlabeled_homography(best_H, image_points, model_points, ransac_threshold_ft)


def fit_unlabeled_homographies(calibration: dict, ransac_threshold_ft: float) -> list[HomographyFit]:
    by_frame: dict[int, list[dict]] = {}
    for click in calibration.get("clicks", []):
        by_frame.setdefault(int(click["frame"]), []).append(click)
    model_points = dense_floor_model()
    landmark_points = np.asarray(LANDMARK_WORLD_POINTS, dtype=np.float32)
    fits: list[HomographyFit] = []
    iterations = int(calibration.get("auto_fit_iterations", 40000))
    for frame, clicks in sorted(by_frame.items()):
        if len(clicks) < 4:
            continue
        image_points = np.asarray([[c["image"]["x"], c["image"]["y"]] for c in clicks], dtype=np.float32)
        H, inliers, mean_error = fit_one_unlabeled_frame(
            image_points,
            model_points,
            landmark_points,
            ransac_threshold_ft,
            iterations=iterations,
            seed=frame + 1337,
        )
        fits.append(HomographyFit(frame=frame, H=H, points=len(clicks), inliers=inliers, mean_error_ft=mean_error))
    if not fits:
        raise RuntimeError("No usable unlabeled homography fits. Add at least 4 clicks on one frame; 8+ is better.")
    return fits


def nearest_fit(fits: list[HomographyFit], frame_idx: int) -> HomographyFit:
    return min(fits, key=lambda fit: abs(fit.frame - frame_idx))


def transform_point(H: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    arr = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    out = cv2.perspectiveTransform(arr, H).reshape(2)
    return float(out[0]), float(out[1])


def clamp_world_point(x: float, y: float) -> tuple[float, float]:
    return float(np.clip(x, 0.0, FLOOR_LENGTH_FT)), float(np.clip(y, 0.0, FLOOR_WIDTH_FT))


def world_to_canvas(x: float, y: float, width: int, height: int, margin: int) -> tuple[int, int]:
    scale = min((width - 2 * margin) / FLOOR_LENGTH_FT, (height - 2 * margin) / FLOOR_WIDTH_FT)
    ox = (width - scale * FLOOR_LENGTH_FT) * 0.5
    oy = (height - scale * FLOOR_WIDTH_FT) * 0.5
    px = ox + x * scale
    py = oy + (FLOOR_WIDTH_FT - y) * scale
    return int(round(px)), int(round(py))


def rounded_floor_points(samples_per_corner: int = 24) -> np.ndarray:
    r = CORNER_RADIUS_FT
    centers = [
        (r, r, 180, 270),
        (FLOOR_LENGTH_FT - r, r, 270, 360),
        (FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT - r, 0, 90),
        (r, FLOOR_WIDTH_FT - r, 90, 180),
    ]
    pts = []
    for cx, cy, start, stop in centers:
        for angle in np.linspace(start, stop, samples_per_corner):
            rad = np.deg2rad(angle)
            pts.append((cx + r * np.cos(rad), cy + r * np.sin(rad)))
    return np.asarray(pts, dtype=np.float32)


def poly_world_to_canvas(points: np.ndarray, width: int, height: int, margin: int) -> np.ndarray:
    return np.asarray([world_to_canvas(float(x), float(y), width, height, margin) for x, y in points], dtype=np.int32)


def draw_world_line(img: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], color: tuple[int, int, int], thickness: int, margin: int) -> None:
    a = world_to_canvas(p1[0], p1[1], img.shape[1], img.shape[0], margin)
    b = world_to_canvas(p2[0], p2[1], img.shape[1], img.shape[0], margin)
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_world_circle(img: np.ndarray, center: tuple[float, float], radius_ft: float, color: tuple[int, int, int], thickness: int, margin: int) -> None:
    scale = min((img.shape[1] - 2 * margin) / FLOOR_LENGTH_FT, (img.shape[0] - 2 * margin) / FLOOR_WIDTH_FT)
    c = world_to_canvas(center[0], center[1], img.shape[1], img.shape[0], margin)
    cv2.circle(img, c, int(round(radius_ft * scale)), color, thickness, cv2.LINE_AA)


def draw_floor(width: int, height: int, margin: int) -> np.ndarray:
    img = np.full((height, width, 3), (28, 74, 54), dtype=np.uint8)
    boundary = poly_world_to_canvas(rounded_floor_points(), width, height, margin)
    cv2.fillPoly(img, [boundary], (35, 108, 78), cv2.LINE_AA)
    cv2.polylines(img, [boundary], isClosed=True, color=(0, 220, 255), thickness=4, lineType=cv2.LINE_AA)

    white = (240, 240, 235)
    red = (50, 70, 230)
    muted = (90, 145, 118)
    for x in [12.0, 57.5, 100.0, 142.5, 188.0]:
        draw_world_line(img, (x, 0.0), (x, FLOOR_WIDTH_FT), white, 2 if x != 100.0 else 3, margin)
    draw_world_circle(img, (100.0, 42.5), 11.0, white, 2, margin)
    cv2.circle(img, world_to_canvas(100.0, 42.5, width, height, margin), 4, white, -1, cv2.LINE_AA)
    for p in [(42.5, 15.0), (42.5, 70.0), (157.5, 15.0), (157.5, 70.0)]:
        cv2.circle(img, world_to_canvas(*p, width, height, margin), 4, white, -1, cv2.LINE_AA)

    for center in [(12.0, 42.5), (188.0, 42.5)]:
        draw_world_circle(img, center, 9.25, white, 2, margin)
    draw_world_line(img, (12.0, 40.125), (12.0, 44.875), red, 4, margin)
    draw_world_line(img, (188.0, 40.125), (188.0, 44.875), red, 4, margin)

    for x, label in [(12, "G"), (57.5, "R"), (100, "C"), (142.5, "R"), (188, "G")]:
        px, py = world_to_canvas(float(x), 3.0, width, height, margin)
        cv2.putText(img, label, (px - 7, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, muted, 1, cv2.LINE_AA)
    return img


def rgb_to_bgr(rgb: list[int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def write_h264(frames_bgr: list[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="error",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame in frames_bgr:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="outputs/floor_homography_clicks.json")
    parser.add_argument("--sam3-json", default="outputs/sam3_team_transreid_3clusters_detections.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--instance-mask-dir", default="outputs/sam3_text_player_instance_masks")
    parser.add_argument("--output-video", default="outputs/birds_eye_player_locations_h264.mp4")
    parser.add_argument("--output-json", default="outputs/birds_eye_player_locations.json")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=54)
    parser.add_argument("--ransac-threshold-ft", type=float, default=3.0)
    parser.add_argument("--auto-fit-iterations", type=int, default=40000)
    parser.add_argument("--trail-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(Path(args.calibration).read_text())
    calibration["auto_fit_iterations"] = args.auto_fit_iterations
    sam3 = json.loads(Path(args.sam3_json).read_text())
    fits = fit_homographies(calibration, args.ransac_threshold_ft)
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    frames_meta = sam3["frames"][: len(frame_paths)]
    mask_dir = Path(args.instance_mask_dir)

    rendered: list[np.ndarray] = []
    position_history: list[dict] = []
    tracks: dict[int, list[tuple[int, float, float, list[int]]]] = {}
    output_meta = {
        "schema": "birds_eye_player_locations_v1",
        "source_sam3_json": args.sam3_json,
        "calibration": args.calibration,
        "fps": args.fps,
        "floor": {"length_ft": FLOOR_LENGTH_FT, "width_ft": FLOOR_WIDTH_FT, "corner_radius_ft": CORNER_RADIUS_FT},
        "homographies": [
            {
                "frame": fit.frame,
                "points": fit.points,
                "inliers": fit.inliers,
                "mean_error_ft": fit.mean_error_ft,
                "matrix_image_to_world": fit.H.tolist(),
            }
            for fit in fits
        ],
        "frames": [],
    }

    for frame_idx, (frame_path, record) in enumerate(zip(frame_paths, frames_meta)):
        source = cv2.imread(str(frame_path))
        if source is None:
            continue
        height, width = source.shape[:2]
        object_ids = [int(v) for v in record.get("object_ids", [])]
        boxes = record.get("boxes", [])
        colors = record.get("team_colors", [])
        teams = record.get("teams", ["unknown"] * len(object_ids))
        fit = nearest_fit(fits, frame_idx)
        frame_positions = []

        for det_idx, (obj_id, box) in enumerate(zip(object_ids, boxes)):
            mask = load_instance_mask(mask_dir, frame_idx, object_ids, det_idx, height, width)
            image_point = floor_point_from_detection(box, mask, (height, width))
            world_x, world_y = transform_point(fit.H, image_point)
            color = colors[det_idx] if det_idx < len(colors) else DEFAULT_COLORS[obj_id % len(DEFAULT_COLORS)]
            team = teams[det_idx] if det_idx < len(teams) else "unknown"
            frame_positions.append(
                {
                    "object_id": int(obj_id),
                    "team": team,
                    "color_rgb": [int(c) for c in color],
                    "image_point": [float(image_point[0]), float(image_point[1])],
                    "world_point_ft": [float(world_x), float(world_y)],
                    "homography_frame": int(fit.frame),
                }
            )
            tracks.setdefault(obj_id, []).append((frame_idx, world_x, world_y, [int(c) for c in color]))

        canvas = draw_floor(args.width, args.height, args.margin)
        min_trail_frame = frame_idx - max(0, args.trail_frames)
        for obj_id, samples in tracks.items():
            recent = [(x, y, color) for f, x, y, color in samples if f >= min_trail_frame and f <= frame_idx]
            for (x1, y1, color1), (x2, y2, color2) in zip(recent, recent[1:]):
                if not (
                    -20.0 <= x1 <= FLOOR_LENGTH_FT + 20.0
                    and -20.0 <= y1 <= FLOOR_WIDTH_FT + 20.0
                    and -20.0 <= x2 <= FLOOR_LENGTH_FT + 20.0
                    and -20.0 <= y2 <= FLOOR_WIDTH_FT + 20.0
                ):
                    continue
                x1c, y1c = clamp_world_point(x1, y1)
                x2c, y2c = clamp_world_point(x2, y2)
                p1 = world_to_canvas(x1c, y1c, args.width, args.height, args.margin)
                p2 = world_to_canvas(x2c, y2c, args.width, args.height, args.margin)
                cv2.line(canvas, p1, p2, rgb_to_bgr(color2), 2, cv2.LINE_AA)

        for item in frame_positions:
            x, y = item["world_point_ft"]
            if not (-20.0 <= x <= FLOOR_LENGTH_FT + 20.0 and -20.0 <= y <= FLOOR_WIDTH_FT + 20.0):
                continue
            display_x, display_y = clamp_world_point(x, y)
            p = world_to_canvas(display_x, display_y, args.width, args.height, args.margin)
            color_bgr = rgb_to_bgr(item["color_rgb"])
            cv2.circle(canvas, p, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 6, color_bgr, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(item["object_id"]),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"frame {frame_idx}  homography frame {fit.frame}  points {fit.inliers}/{fit.points}  mean err {fit.mean_error_ft:.2f} ft",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        output_meta["frames"].append({"frame": frame_idx, "players": frame_positions})
        position_history.append({"frame": frame_idx, "players": frame_positions})
        rendered.append(canvas)

    write_h264(rendered, Path(args.output_video), args.fps)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output_meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output_video": args.output_video,
                "output_json": args.output_json,
                "frames": len(rendered),
                "homography_frames": [fit.frame for fit in fits],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
