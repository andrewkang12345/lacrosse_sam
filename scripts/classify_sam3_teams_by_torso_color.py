from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


DEFAULT_TEAM_COLORS = {
    "black": [255, 80, 40],
    "white": [70, 170, 255],
    "unknown": [180, 180, 180],
}


@dataclass
class TorsoSample:
    frame_idx: int
    det_idx: int
    object_id: int
    box: list[float]
    score: float
    torso_rgb: list[float]
    torso_luma: float
    torso_pixels: int
    torso_coverage: float
    occluded: bool
    white_fraction: float
    yellow_fraction: float
    dark_fraction: float
    gray_fraction: float


def parse_color_map(value: str | None) -> dict[str, list[int]]:
    colors = dict(DEFAULT_TEAM_COLORS)
    if not value:
        return colors
    for item in value.split(";"):
        if not item.strip():
            continue
        key, raw = item.split("=", 1)
        rgb = [int(v) for v in raw.split(",")]
        if len(rgb) != 3:
            raise ValueError(f"Expected RGB triplet for {key}: {raw}")
        colors[key.strip()] = rgb
    return colors


def load_instance_mask(mask_dir: Path, frame_idx: int, object_ids: list[int], det_idx: int, height: int, width: int) -> np.ndarray:
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return np.zeros((height, width), dtype=bool)
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
    return np.zeros((height, width), dtype=bool)


def torso_mask_from_box(mask: np.ndarray, box: np.ndarray, torso_y: tuple[float, float], torso_x: tuple[float, float]) -> np.ndarray:
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    tx1 = int(round(x1 + torso_x[0] * bw))
    tx2 = int(round(x1 + torso_x[1] * bw))
    ty1 = int(round(y1 + torso_y[0] * bh))
    ty2 = int(round(y1 + torso_y[1] * bh))
    region = np.zeros_like(mask, dtype=bool)
    region[max(0, ty1) : min(height, ty2), max(0, tx1) : min(width, tx2)] = True
    return mask.astype(bool) & region


def torso_region_from_box(shape: tuple[int, int], box: np.ndarray, torso_y: tuple[float, float], torso_x: tuple[float, float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    tx1 = int(round(x1 + torso_x[0] * bw))
    tx2 = int(round(x1 + torso_x[1] * bw))
    ty1 = int(round(y1 + torso_y[0] * bh))
    ty2 = int(round(y1 + torso_y[1] * bh))
    region = np.zeros((height, width), dtype=bool)
    region[max(0, ty1) : min(height, ty2), max(0, tx1) : min(width, tx2)] = True
    return region


def box_intersection_over_area(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    return inter / area if area > 0 else 0.0


def average_torso_color(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    torso_y: tuple[float, float],
    torso_x: tuple[float, float],
    min_pixels: int,
) -> tuple[list[float], float, int]:
    height, width = frame_rgb.shape[:2]
    torso = torso_mask_from_box(mask, box, torso_y, torso_x)
    if torso.sum() < min_pixels:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        fallback = np.zeros((height, width), dtype=bool)
        fallback[y1 : y1 + max(1, (y2 - y1) // 2), x1:x2] = True
        torso = mask.astype(bool) & fallback
    if torso.sum() < min_pixels:
        torso = torso_mask_from_box(np.ones((height, width), dtype=bool), box, torso_y, torso_x)

    pixels = frame_rgb[torso]
    if pixels.size == 0:
        return [0.0, 0.0, 0.0], 0.0, 0
    avg = pixels.mean(axis=0)
    luma = float(0.2126 * avg[0] + 0.7152 * avg[1] + 0.0722 * avg[2])
    return [float(v) for v in avg], luma, int(len(pixels))


def torso_color_fractions(frame_rgb: np.ndarray, frame_hsv: np.ndarray, torso: np.ndarray) -> dict[str, float]:
    pixels = frame_rgb[torso].astype(np.float32)
    hsv = frame_hsv[torso].astype(np.float32)
    if pixels.size == 0:
        return {"white": 0.0, "yellow": 0.0, "dark": 0.0, "gray": 0.0}

    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    hue = hsv[:, 0] * 2.0
    sat = hsv[:, 1] / 255.0
    val = hsv[:, 2] / 255.0
    max_rgb = np.maximum.reduce([red, green, blue])
    min_rgb = np.minimum.reduce([red, green, blue])

    white = ((sat < 0.35) & (val > 0.45) & (red > 85) & (green > 85) & (blue > 85)) | (
        (red > 120) & (green > 120) & (blue > 120) & ((max_rgb - min_rgb) < 55)
    )
    yellow = (
        (hue >= 35)
        & (hue <= 75)
        & (sat > 0.25)
        & (val > 0.25)
        & (red > 80)
        & (green > 65)
        & (blue < 130)
    )
    dark = (val < 0.38) | ((red < 95) & (green < 95) & (blue < 95))
    gray = (sat < 0.30) & (val >= 0.30) & (val < 0.70) & (~white)
    return {
        "white": float(white.mean()),
        "yellow": float(yellow.mean()),
        "dark": float(dark.mean()),
        "gray": float(gray.mean()),
    }


def kmeans_1d(values: np.ndarray, iterations: int = 25) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 255.0, 127.5
    low, high = np.percentile(values, [25, 75])
    if abs(high - low) < 1e-6:
        return float(low), float(high), float(values.mean())
    centers = np.asarray([low, high], dtype=np.float32)
    for _ in range(iterations):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)
        for idx in range(2):
            if np.any(labels == idx):
                centers[idx] = values[labels == idx].mean()
    centers = np.sort(centers)
    threshold = float(centers.mean())
    return float(centers[0]), float(centers[1]), threshold


def collect_samples(
    frames_bgr: list[np.ndarray],
    sam3_meta: dict,
    mask_dir: Path,
    min_width: float,
    min_height: float,
    score_threshold: float,
    torso_y: tuple[float, float],
    torso_x: tuple[float, float],
    min_pixels: int,
    min_visible_coverage: float,
    overlap_threshold: float,
) -> list[TorsoSample]:
    samples: list[TorsoSample] = []
    for frame_idx, record in enumerate(sam3_meta["frames"][: len(frames_bgr)]):
        frame_rgb = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2RGB)
        frame_hsv = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2HSV)
        height, width = frame_rgb.shape[:2]
        object_ids = [int(v) for v in record.get("object_ids", [])]
        boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
        scores = np.asarray(record.get("scores", []), dtype=np.float32)
        if scores.size != len(boxes):
            scores = np.ones((len(boxes),), dtype=np.float32)
        valid_indices = []
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 - x1 >= min_width and y2 - y1 >= min_height and scores[det_idx] >= score_threshold:
                valid_indices.append(det_idx)

        for det_idx in valid_indices:
            box = boxes[det_idx]
            mask = load_instance_mask(mask_dir, frame_idx, object_ids, det_idx, height, width)
            torso_rgb, torso_luma, torso_pixels = average_torso_color(
                frame_rgb,
                mask,
                box,
                torso_y,
                torso_x,
                min_pixels,
            )
            torso_region = torso_region_from_box((height, width), box, torso_y, torso_x)
            visible_torso_mask = mask.astype(bool) & torso_region
            torso_area = int(torso_region.sum())
            torso_visible = int(visible_torso_mask.sum())
            coverage = float(torso_visible / torso_area) if torso_area else 0.0
            fractions = torso_color_fractions(frame_rgb, frame_hsv, visible_torso_mask)
            max_overlap = 0.0
            for other_idx in valid_indices:
                if other_idx == det_idx:
                    continue
                max_overlap = max(max_overlap, box_intersection_over_area(box, boxes[other_idx]))
            occluded = coverage < min_visible_coverage or max_overlap >= overlap_threshold
            samples.append(
                TorsoSample(
                    frame_idx=frame_idx,
                    det_idx=det_idx,
                    object_id=object_ids[det_idx],
                    box=[float(v) for v in box.tolist()],
                    score=float(scores[det_idx]),
                    torso_rgb=torso_rgb,
                    torso_luma=torso_luma,
                    torso_pixels=torso_pixels,
                    torso_coverage=coverage,
                    occluded=occluded,
                    white_fraction=fractions["white"],
                    yellow_fraction=fractions["yellow"],
                    dark_fraction=fractions["dark"],
                    gray_fraction=fractions["gray"],
                )
            )
    return samples


def draw_review_frame(frame_bgr: np.ndarray, record: dict, team_colors: dict[str, list[int]]) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, team, box, luma in zip(record["object_ids"], record["teams"], record["boxes"], record["torso_luma"]):
        color_rgb = team_colors.get(team, team_colors["unknown"])
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(
            out,
            f"{team}:{obj_id} torso {luma:.0f}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
    return out


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
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--sam3-json", default="outputs/sam3_text_player_instances.json")
    parser.add_argument("--instance-mask-dir", default="outputs/sam3_text_player_instance_masks")
    parser.add_argument("--output-json", default="outputs/sam3_team_torso_color_detections.json")
    parser.add_argument("--output-video", default="outputs/sam3_team_torso_color_detections_review.mp4")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--luma-threshold", type=float, default=0.0, help="Use fixed threshold if > 0; otherwise use 1D k-means.")
    parser.add_argument("--no-smooth-by-object", action="store_true", help="Classify each detection independently instead of smoothing by SAM3 object ID.")
    parser.add_argument("--torso-y", default="0.26,0.70")
    parser.add_argument("--torso-x", default="0.22,0.78")
    parser.add_argument("--min-torso-pixels", type=int, default=25)
    parser.add_argument("--min-visible-coverage", type=float, default=0.35)
    parser.add_argument("--overlap-threshold", type=float, default=0.20)
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=16.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--team-colors", default=None, help="Example: black=255,80,40;white=70,170,255")
    parser.add_argument("--classification-mode", choices=["composition", "luma"], default="composition")
    parser.add_argument("--white-fraction-threshold", type=float, default=0.25)
    parser.add_argument("--yellow-fraction-threshold", type=float, default=0.025)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def parse_pair(value: str) -> tuple[float, float]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 2 or not 0 <= parts[0] < parts[1] <= 1:
        raise ValueError(f"Expected pair in [0,1], got {value}")
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    frames_bgr = [cv2.imread(str(path)) for path in frame_paths]
    sam3_meta = json.loads(Path(args.sam3_json).read_text())
    team_colors = parse_color_map(args.team_colors)
    torso_y = parse_pair(args.torso_y)
    torso_x = parse_pair(args.torso_x)

    samples = collect_samples(
        frames_bgr,
        sam3_meta,
        Path(args.instance_mask_dir),
        args.min_width,
        args.min_height,
        args.score_threshold,
        torso_y,
        torso_x,
        args.min_torso_pixels,
        args.min_visible_coverage,
        args.overlap_threshold,
    )
    smooth_by_object = not args.no_smooth_by_object
    object_feature_medians = {}
    if smooth_by_object:
        luma_by_object: dict[int, list[float]] = {}
        fractions_by_object: dict[int, list[tuple[float, float, float, float]]] = {}
        for sample in samples:
            if not sample.occluded:
                luma_by_object.setdefault(sample.object_id, []).append(sample.torso_luma)
                fractions_by_object.setdefault(sample.object_id, []).append(
                    (sample.white_fraction, sample.yellow_fraction, sample.dark_fraction, sample.gray_fraction)
                )
        for sample in samples:
            if sample.object_id not in luma_by_object:
                luma_by_object.setdefault(sample.object_id, []).append(sample.torso_luma)
                fractions_by_object.setdefault(sample.object_id, []).append(
                    (sample.white_fraction, sample.yellow_fraction, sample.dark_fraction, sample.gray_fraction)
                )
        object_medians = {obj_id: float(statistics.median(vals)) for obj_id, vals in luma_by_object.items()}
        lumas = np.asarray(list(object_medians.values()), dtype=np.float32)
        for obj_id, vals in fractions_by_object.items():
            arr = np.asarray(vals, dtype=np.float32)
            object_feature_medians[obj_id] = {
                "white_fraction": float(np.median(arr[:, 0])),
                "yellow_fraction": float(np.median(arr[:, 1])),
                "dark_fraction": float(np.median(arr[:, 2])),
                "gray_fraction": float(np.median(arr[:, 3])),
            }
    else:
        object_medians = {}
        lumas = np.asarray([sample.torso_luma for sample in samples], dtype=np.float32)
    dark_center, light_center, learned_threshold = kmeans_1d(lumas)
    threshold = args.luma_threshold if args.luma_threshold > 0 else learned_threshold
    object_teams = {}
    for obj_id, luma in object_medians.items():
        features = object_feature_medians.get(obj_id, {})
        if args.classification_mode == "composition":
            yellow_fraction = features.get("yellow_fraction", 0.0)
            white_fraction = features.get("white_fraction", 0.0)
            team = "black" if yellow_fraction >= args.yellow_fraction_threshold or white_fraction < args.white_fraction_threshold else "white"
        else:
            team = "black" if luma < threshold else "white"
        object_teams[obj_id] = team

    samples_by_frame: dict[int, list[TorsoSample]] = {}
    for sample in samples:
        samples_by_frame.setdefault(sample.frame_idx, []).append(sample)

    metadata = {
        "model": "sam3_torso_color_team_classifier",
        "source_json": args.sam3_json,
        "fps": args.fps,
        "team_colors": team_colors,
        "torso_y": list(torso_y),
        "torso_x": list(torso_x),
        "min_visible_coverage": args.min_visible_coverage,
        "overlap_threshold": args.overlap_threshold,
        "luma_threshold": float(threshold),
        "luma_centers": {"black": dark_center, "white": light_center},
        "smooth_by_object": smooth_by_object,
        "object_median_luma": object_medians,
        "object_feature_medians": object_feature_medians,
        "object_teams": object_teams,
        "classification_mode": args.classification_mode,
        "white_fraction_threshold": args.white_fraction_threshold,
        "yellow_fraction_threshold": args.yellow_fraction_threshold,
        "frames": [],
    }
    rendered = []
    for frame_idx, frame_bgr in enumerate(frames_bgr):
        record = {
            "frame": frame_idx,
            "object_ids": [],
            "scores": [],
            "boxes": [],
            "teams": [],
            "team_colors": [],
            "torso_rgb": [],
            "torso_luma": [],
            "torso_pixels": [],
            "torso_coverage": [],
            "occluded_for_color": [],
            "white_fraction": [],
            "yellow_fraction": [],
            "dark_fraction": [],
            "gray_fraction": [],
        }
        for sample in samples_by_frame.get(frame_idx, []):
            team = object_teams.get(sample.object_id) if smooth_by_object else None
            if team is None:
                if args.classification_mode == "composition":
                    team = (
                        "black"
                        if sample.yellow_fraction >= args.yellow_fraction_threshold
                        or sample.white_fraction < args.white_fraction_threshold
                        else "white"
                    )
                else:
                    team = "black" if sample.torso_luma < threshold else "white"
            record["object_ids"].append(int(sample.object_id))
            record["scores"].append(float(sample.score))
            record["boxes"].append(sample.box)
            record["teams"].append(team)
            record["team_colors"].append(team_colors[team])
            record["torso_rgb"].append(sample.torso_rgb)
            record["torso_luma"].append(float(sample.torso_luma))
            record["torso_pixels"].append(int(sample.torso_pixels))
            record["torso_coverage"].append(float(sample.torso_coverage))
            record["occluded_for_color"].append(bool(sample.occluded))
            record["white_fraction"].append(float(sample.white_fraction))
            record["yellow_fraction"].append(float(sample.yellow_fraction))
            record["dark_fraction"].append(float(sample.dark_fraction))
            record["gray_fraction"].append(float(sample.gray_fraction))
        metadata["frames"].append(record)
        rendered.append(draw_review_frame(frame_bgr, record, team_colors))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metadata, indent=2) + "\n")
    write_h264(rendered, Path(args.output_video), args.fps)
    counts = {
        team: sum(record["teams"].count(team) for record in metadata["frames"])
        for team in ["black", "white"]
    }
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_video": args.output_video,
                "counts": counts,
                "luma_threshold": float(threshold),
                "luma_centers": metadata["luma_centers"],
                "smooth_by_object": smooth_by_object,
                "classification_mode": args.classification_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
