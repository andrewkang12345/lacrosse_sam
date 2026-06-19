from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from render_birds_eye_locations import (
    CORNER_RADIUS_FT,
    FLOOR_LENGTH_FT,
    FLOOR_WIDTH_FT,
    fit_homographies,
    floor_point_from_detection,
    load_instance_mask,
    nearest_fit,
    rounded_floor_points,
    transform_point,
)


FEATURE_COLORS_BGR = {
    "left_restraining_line": (69, 122, 255),
    "right_restraining_line": (255, 167, 90),
    "goal_crease": (74, 211, 247),
    "midfield_line": (123, 212, 100),
    "field_outline": (255, 124, 215),
}
FEATURE_LABELS = {
    "left_restraining_line": "left R",
    "right_restraining_line": "right R",
    "goal_crease": "crease",
    "midfield_line": "mid",
    "field_outline": "outline",
}


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


def rgb_to_bgr(rgb: list[int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def world_to_image_points(H_world_to_image: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(points.astype(np.float32).reshape(1, -1, 2), H_world_to_image).reshape(-1, 2)
    return projected


def draw_projected_polyline(
    frame: np.ndarray,
    H_world_to_image: np.ndarray,
    points: list[tuple[float, float]] | np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    closed: bool = False,
) -> None:
    world = np.asarray(points, dtype=np.float32)
    image = world_to_image_points(H_world_to_image, world)
    valid = np.isfinite(image).all(axis=1)
    if valid.sum() < 2:
        return
    pts = np.round(image[valid]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], closed, color, thickness, cv2.LINE_AA)


def line_points(x1: float, y1: float, x2: float, y2: float, samples: int = 80) -> np.ndarray:
    return np.column_stack([np.linspace(x1, x2, samples), np.linspace(y1, y2, samples)]).astype(np.float32)


def circle_points(cx: float, cy: float, radius: float, samples: int = 96) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]).astype(np.float32)


def draw_projected_floor(frame: np.ndarray, H_world_to_image: np.ndarray) -> None:
    yellow = (0, 220, 255)
    white = (245, 245, 245)
    red = (40, 60, 230)
    green = (60, 180, 110)

    draw_projected_polyline(frame, H_world_to_image, rounded_floor_points(samples_per_corner=32), yellow, 3, closed=True)
    for x in [12.0, 57.5, 100.0, 142.5, 188.0]:
        draw_projected_polyline(frame, H_world_to_image, line_points(x, 0.0, x, FLOOR_WIDTH_FT), white, 2)
    for y in [0.0, FLOOR_WIDTH_FT]:
        draw_projected_polyline(frame, H_world_to_image, line_points(CORNER_RADIUS_FT, y, FLOOR_LENGTH_FT - CORNER_RADIUS_FT, y), yellow, 2)
    draw_projected_polyline(frame, H_world_to_image, circle_points(100.0, 42.5, 11.0), white, 2)
    draw_projected_polyline(frame, H_world_to_image, circle_points(12.0, 42.5, 9.25), white, 2)
    draw_projected_polyline(frame, H_world_to_image, circle_points(188.0, 42.5, 9.25), white, 2)
    draw_projected_polyline(frame, H_world_to_image, line_points(12.0, 40.125, 12.0, 44.875, 8), red, 4)
    draw_projected_polyline(frame, H_world_to_image, line_points(188.0, 40.125, 188.0, 44.875, 8), red, 4)
    for point in [(100.0, 42.5), (42.5, 15.0), (42.5, 70.0), (157.5, 15.0), (157.5, 70.0)]:
        img_pt = world_to_image_points(H_world_to_image, np.asarray([point], dtype=np.float32))[0]
        if np.isfinite(img_pt).all():
            cv2.circle(frame, tuple(np.round(img_pt).astype(int)), 4, green, -1, cv2.LINE_AA)


def draw_calibration_clicks(frame: np.ndarray, calibration: dict, fit_frame: int, H_world_to_image: np.ndarray) -> None:
    clicks = [click for click in calibration.get("clicks", []) if int(click["frame"]) == fit_frame and "world" in click]
    for idx, click in enumerate(clicks, 1):
        image = (int(round(click["image"]["x"])), int(round(click["image"]["y"])))
        world = np.asarray([[click["world"]["x"], click["world"]["y"]]], dtype=np.float32)
        reproj = world_to_image_points(H_world_to_image, world)[0]
        reproj_xy = tuple(np.round(reproj).astype(int))
        cv2.circle(frame, image, 9, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, reproj_xy, 6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, image, reproj_xy, (255, 255, 0), 1, cv2.LINE_AA)
        label = f"{idx}:{click['landmark_id']}"
        cv2.putText(frame, label, (image[0] + 10, image[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (image[0] + 10, image[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_unlabeled_clicks(frame: np.ndarray, calibration: dict, fit_frame: int) -> None:
    clicks = []
    if calibration.get("schema") == "floor_unlabeled_clicks_v1":
        clicks = calibration.get("clicks", [])
    elif calibration.get("source_unlabeled_clicks"):
        source = Path(calibration["source_unlabeled_clicks"])
        if source.exists():
            clicks = json.loads(source.read_text()).get("clicks", [])
    clicks = [click for click in clicks if int(click["frame"]) == fit_frame and "image" in click]
    for idx, click in enumerate(clicks, 1):
        image = (int(round(click["image"]["x"])), int(round(click["image"]["y"])))
        cv2.circle(frame, image, 8, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, image, 10, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"free {idx}", (image[0] + 9, image[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"free {idx}", (image[0] + 9, image[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def collect_feature_clicks(calibration: dict) -> list[dict]:
    paths = []
    if calibration.get("source_feature_clicks"):
        paths.append(Path(calibration["source_feature_clicks"]))
    if calibration.get("source_base_calibration"):
        base_path = Path(calibration["source_base_calibration"])
        if base_path.exists():
            base = json.loads(base_path.read_text())
            if base.get("source_feature_clicks"):
                paths.append(Path(base["source_feature_clicks"]))
    clicks = []
    for path in paths:
        if path.exists():
            clicks.extend(json.loads(path.read_text()).get("clicks", []))
    return clicks


def draw_feature_clicks(frame: np.ndarray, calibration: dict, fit_frame: int) -> None:
    clicks = [click for click in collect_feature_clicks(calibration) if int(click["frame"]) == fit_frame and "image" in click]
    for idx, click in enumerate(clicks, 1):
        feature = click.get("feature", "unknown")
        image = (int(round(click["image"]["x"])), int(round(click["image"]["y"])))
        color = FEATURE_COLORS_BGR.get(feature, (255, 255, 255))
        cv2.circle(frame, image, 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, image, 10, (0, 0, 0), 1, cv2.LINE_AA)
        label = f"{idx}:{FEATURE_LABELS.get(feature, feature)}"
        cv2.putText(frame, label, (image[0] + 9, image[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (image[0] + 9, image[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)


def load_tracked_landmark_mask(source: dict, frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask_dir = Path(source["path"])
    object_id = source.get("object_id")
    npz_path = mask_dir / f"{frame_idx:08d}.npz"
    png_path = mask_dir / f"{frame_idx:08d}.png"
    if npz_path.exists():
        data = np.load(npz_path)
        object_ids = [int(value) for value in data["object_ids"].tolist()]
        masks = data["masks"].astype(bool)
        if masks.size == 0:
            return np.zeros((height, width), dtype=bool)
        if object_id is not None:
            object_id = int(object_id)
            if object_id not in object_ids:
                return np.zeros((height, width), dtype=bool)
            mask = masks[object_ids.index(object_id)]
        else:
            mask = masks.any(axis=0)
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask
    if png_path.exists():
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((height, width), dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if object_id is not None:
            return mask == int(object_id)
        return mask > 0
    return np.zeros((height, width), dtype=bool)


def draw_tracked_landmark_masks(frame: np.ndarray, calibration: dict, frame_idx: int, alpha: float) -> None:
    sources = calibration.get("tracked_mask_sources", [])
    if not sources:
        return
    height, width = frame.shape[:2]
    for source in sources:
        feature = str(source.get("feature", "unknown"))
        mask = load_tracked_landmark_mask(source, frame_idx, (height, width))
        if not mask.any():
            continue
        color = np.asarray(FEATURE_COLORS_BGR.get(feature, (255, 255, 255)), dtype=np.float32)
        frame_float = frame.astype(np.float32)
        frame_float[mask] = (1.0 - alpha) * frame_float[mask] + alpha * color
        frame[:] = np.clip(frame_float, 0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, tuple(int(v) for v in color), 2, cv2.LINE_AA)
        ys, xs = np.where(mask)
        if len(xs):
            x = int(np.median(xs))
            y = int(np.percentile(ys, 10))
            label = FEATURE_LABELS.get(feature, feature)
            cv2.putText(frame, label, (x + 6, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x + 6, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_player_points(
    frame: np.ndarray,
    frame_idx: int,
    record: dict,
    mask_dir: Path,
    H_image_to_world: np.ndarray,
) -> None:
    height, width = frame.shape[:2]
    object_ids = [int(v) for v in record.get("object_ids", [])]
    boxes = record.get("boxes", [])
    colors = record.get("team_colors", [])
    teams = record.get("teams", ["unknown"] * len(object_ids))
    for det_idx, (obj_id, box) in enumerate(zip(object_ids, boxes)):
        mask = load_instance_mask(mask_dir, frame_idx, object_ids, det_idx, height, width)
        image_point = floor_point_from_detection(box, mask, (height, width))
        world_point = transform_point(H_image_to_world, image_point)
        color = colors[det_idx] if det_idx < len(colors) else [255, 255, 255]
        team = teams[det_idx] if det_idx < len(teams) else "unknown"
        p = (int(round(image_point[0])), int(round(image_point[1])))
        cv2.circle(frame, p, 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 6, rgb_to_bgr(color), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{obj_id} {team} ({world_point[0]:.0f},{world_point[1]:.0f})",
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{obj_id} {team} ({world_point[0]:.0f},{world_point[1]:.0f})",
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="outputs/floor_homography_inferred_frame77.json")
    parser.add_argument("--sam3-json", default="outputs/sam3_team_transreid_3clusters_detections.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--instance-mask-dir", default="outputs/sam3_text_player_instance_masks")
    parser.add_argument("--output-video", default="outputs/camera_floor_homography_overlay_h264.mp4")
    parser.add_argument("--draw-landmark-masks", action="store_true", help="Overlay tracked floor-landmark segmentations from calibration tracked_mask_sources.")
    parser.add_argument("--landmark-mask-alpha", type=float, default=0.35)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ransac-threshold-ft", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(Path(args.calibration).read_text())
    error_units = "px" if calibration.get("schema") == "floor_homography_matrix_v1" else "ft"
    sam3 = json.loads(Path(args.sam3_json).read_text())
    fits = fit_homographies(calibration, args.ransac_threshold_ft)
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    rendered = []
    mask_dir = Path(args.instance_mask_dir)
    frame_meta = sam3["frames"][: len(frame_paths)]
    for frame_idx, (frame_path, record) in enumerate(zip(frame_paths, frame_meta)):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        fit = nearest_fit(fits, frame_idx)
        H_world_to_image = np.linalg.inv(fit.H)
        overlay = frame.copy()
        draw_projected_floor(overlay, H_world_to_image)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)
        if args.draw_landmark_masks:
            draw_tracked_landmark_masks(frame, calibration, frame_idx, args.landmark_mask_alpha)
        draw_player_points(frame, frame_idx, record, mask_dir, fit.H)
        if frame_idx == fit.frame:
            draw_calibration_clicks(frame, calibration, fit.frame, H_world_to_image)
            draw_unlabeled_clicks(frame, calibration, fit.frame)
            draw_feature_clicks(frame, calibration, fit.frame)
        cv2.putText(
            frame,
            f"floor fit from frame {fit.frame}: points {fit.inliers}/{fit.points}, mean err {fit.mean_error_ft:.2f} {error_units}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"floor fit from frame {fit.frame}: points {fit.inliers}/{fit.points}, mean err {fit.mean_error_ft:.2f} {error_units}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(frame)
    write_h264(rendered, Path(args.output_video), args.fps)
    print(json.dumps({"output_video": args.output_video, "frames": len(rendered), "homography_frames": [fit.frame for fit in fits]}, indent=2))


if __name__ == "__main__":
    main()
