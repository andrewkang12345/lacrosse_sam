"""Team per SAM3 player detection from jersey brightness, with the checked review applied.

The torso (a box inside the player's mask box, --torso-y/--torso-x) is averaged in each
frame. Each SAM3 track gets the median luma of its unhidden torsos, and the track medians
are split into two teams by 1D k-means (dark: black, light: white). The review JSON
(data/team_review/) overrides tracks checked by eye: black, white, referee, ignore (off
the floor: bench, stands), or per_detection for a track that switches between players,
which is classified frame by frame.

Writes the teams per detection (`teams`, `team_colors`, per frame) and a review video
with each box labelled team:object id.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_utils import write_h264_video

TEAM_COLORS = {
    "black": [255, 80, 40],
    "white": [70, 170, 255],
    "referee": [150, 150, 150],
    "ignore": [90, 90, 90],
}


@dataclass
class TorsoSample:
    frame_idx: int
    object_id: int
    box: list[float]
    score: float
    luma: float
    hidden: bool


def load_instance_mask(mask_dir: Path, frame_idx: int, object_ids: list[int], det_idx: int, shape: tuple[int, int]) -> np.ndarray:
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    data = np.load(path)
    mask_ids = [int(v) for v in data["object_ids"].tolist()]
    masks = data["masks"].astype(bool)
    obj_id = int(object_ids[det_idx])
    mask_idx = mask_ids.index(obj_id) if obj_id in mask_ids else det_idx
    return masks[mask_idx] if mask_idx < masks.shape[0] else np.zeros(shape, dtype=bool)


def torso_region(shape: tuple[int, int], box: np.ndarray, torso_y: tuple[float, float], torso_x: tuple[float, float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    region = np.zeros((height, width), dtype=bool)
    ty1, ty2 = int(round(y1 + torso_y[0] * bh)), int(round(y1 + torso_y[1] * bh))
    tx1, tx2 = int(round(x1 + torso_x[0] * bw)), int(round(x1 + torso_x[1] * bw))
    region[max(0, ty1):min(height, ty2), max(0, tx1):min(width, tx2)] = True
    return region


def torso_luma(frame_rgb: np.ndarray, mask: np.ndarray, box: np.ndarray, torso_y, torso_x, min_pixels: int) -> float:
    """Mean luma of the torso pixels; falls back to the upper half of the mask, then to the bare torso box."""
    height, width = frame_rgb.shape[:2]
    torso = mask & torso_region((height, width), box, torso_y, torso_x)
    if torso.sum() < min_pixels:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        upper = np.zeros((height, width), dtype=bool)
        upper[y1:y1 + max(1, (y2 - y1) // 2), x1:x2] = True
        torso = mask & upper
    if torso.sum() < min_pixels:
        torso = torso_region((height, width), box, torso_y, torso_x)
    pixels = frame_rgb[torso]
    if pixels.size == 0:
        return 0.0
    avg = pixels.mean(axis=0)
    return float(0.2126 * avg[0] + 0.7152 * avg[1] + 0.0722 * avg[2])


def box_intersection_over_area(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    area = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    return inter / area if area > 0 else 0.0


def kmeans_1d(values: np.ndarray, iterations: int = 25) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 255.0, 127.5
    low, high = np.percentile(values, [25, 75])
    if abs(high - low) < 1e-6:
        return float(low), float(high), float(values.mean())
    centers = np.asarray([low, high], dtype=np.float32)
    for _ in range(iterations):
        labels = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        for idx in range(2):
            if np.any(labels == idx):
                centers[idx] = values[labels == idx].mean()
    centers = np.sort(centers)
    return float(centers[0]), float(centers[1]), float(centers.mean())


def collect_samples(frames_bgr: list[np.ndarray], sam3_meta: dict, mask_dir: Path, args: argparse.Namespace) -> list[TorsoSample]:
    samples = []
    for frame_idx, record in enumerate(sam3_meta["frames"][:len(frames_bgr)]):
        frame_rgb = cv2.cvtColor(frames_bgr[frame_idx], cv2.COLOR_BGR2RGB)
        shape = frame_rgb.shape[:2]
        object_ids = [int(v) for v in record.get("object_ids", [])]
        boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
        scores = np.asarray(record.get("scores", []), dtype=np.float32)
        if scores.size != len(boxes):
            scores = np.ones((len(boxes),), dtype=np.float32)
        valid = [i for i, (x1, y1, x2, y2) in enumerate(boxes) if x2 - x1 >= args.min_width and y2 - y1 >= args.min_height]
        for det_idx in valid:
            box = boxes[det_idx]
            mask = load_instance_mask(mask_dir, frame_idx, object_ids, det_idx, shape)
            region = torso_region(shape, box, args.torso_y, args.torso_x)
            coverage = float((mask & region).sum() / region.sum()) if region.sum() else 0.0
            overlap = max((box_intersection_over_area(box, boxes[o]) for o in valid if o != det_idx), default=0.0)
            samples.append(TorsoSample(
                frame_idx=frame_idx, object_id=object_ids[det_idx], box=[float(v) for v in box.tolist()], score=float(scores[det_idx]),
                luma=torso_luma(frame_rgb, mask, box, args.torso_y, args.torso_x, args.min_torso_pixels),
                hidden=coverage < args.min_visible_coverage or overlap >= args.overlap_threshold,
            ))
    return samples


def draw_review_frame(frame_bgr: np.ndarray, record: dict) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, team, box, luma in zip(record["object_ids"], record["teams"], record["boxes"], record["torso_luma"]):
        r, g, b = TEAM_COLORS[team]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (b, g, r), 2)
        cv2.putText(out, f"{team}:{obj_id} torso {luma:.0f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (b, g, r), 1, cv2.LINE_AA)
    return out


def parse_pair(value: str) -> tuple[float, float]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 2 or not 0 <= parts[0] < parts[1] <= 1:
        raise ValueError(f"Expected pair in [0,1], got {value}")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sam3-json", required=True, help="SAM3 player detections (boxes from rebox_sam3_json_from_instance_masks.py or link_sam3_chunk_tracks.py).")
    parser.add_argument("--instance-mask-dir", required=True)
    parser.add_argument("--review", default=None, help="Per-track corrections checked by eye, {object_id: team}; keys starting with '_' are comments.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--torso-y", type=parse_pair, default="0.26,0.70", help="Torso rows as fractions of the box height.")
    parser.add_argument("--torso-x", type=parse_pair, default="0.22,0.78", help="Torso columns as fractions of the box width.")
    parser.add_argument("--min-torso-pixels", type=int, default=25)
    parser.add_argument("--min-visible-coverage", type=float, default=0.35, help="Torsos whose box is less covered by the mask count as hidden.")
    parser.add_argument("--overlap-threshold", type=float, default=0.20, help="Torsos whose box overlaps another this much count as hidden.")
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_bgr = [cv2.imread(str(path)) for path in sorted(Path(args.frames_dir).glob("frame_*.jpg"))]
    sam3_meta = json.loads(Path(args.sam3_json).read_text())
    samples = collect_samples(frames_bgr, sam3_meta, Path(args.instance_mask_dir), args)

    luma_by_object: dict[int, list[float]] = {}
    for sample in samples:
        if not sample.hidden:
            luma_by_object.setdefault(sample.object_id, []).append(sample.luma)
    for sample in samples:
        if sample.object_id not in luma_by_object:  # tracks that are always hidden use all their torsos
            luma_by_object.setdefault(sample.object_id, []).append(sample.luma)
    object_luma = {obj_id: float(statistics.median(vals)) for obj_id, vals in luma_by_object.items()}
    dark_center, light_center, threshold = kmeans_1d(np.asarray(list(object_luma.values()), dtype=np.float32))
    object_teams = {obj_id: "black" if luma < threshold else "white" for obj_id, luma in object_luma.items()}

    raw = json.loads(Path(args.review).read_text()) if args.review else {}
    review = {int(k): str(v) for k, v in raw.items() if not str(k).startswith("_")}
    unknown = sorted(set(review.values()) - set(TEAM_COLORS) - {"per_detection"})
    if unknown:
        raise ValueError(f"Unknown team names in {args.review}: {unknown}")

    samples_by_frame: dict[int, list[TorsoSample]] = {}
    for sample in samples:
        samples_by_frame.setdefault(sample.frame_idx, []).append(sample)
    out = {
        "source_json": args.sam3_json, "fps": args.fps, "team_colors": TEAM_COLORS,
        "torso_y": list(args.torso_y), "torso_x": list(args.torso_x), "luma_threshold": threshold,
        "luma_centers": {"black": dark_center, "white": light_center}, "object_median_luma": object_luma,
        "object_teams": object_teams, "review": args.review, "review_overrides": {str(k): v for k, v in sorted(review.items())},
        "frames": [],
    }
    rendered = []
    for frame_idx, frame_bgr in enumerate(frames_bgr):
        record = {"frame": frame_idx, "object_ids": [], "scores": [], "boxes": [], "teams": [], "team_colors": [], "torso_luma": [], "hidden_for_color": []}
        for sample in samples_by_frame.get(frame_idx, []):
            reviewed = review.get(sample.object_id)
            if reviewed == "per_detection":
                team = "black" if sample.luma < threshold else "white"
            else:
                team = reviewed or object_teams[sample.object_id]
            record["object_ids"].append(sample.object_id)
            record["scores"].append(sample.score)
            record["boxes"].append(sample.box)
            record["teams"].append(team)
            record["team_colors"].append(TEAM_COLORS[team])
            record["torso_luma"].append(sample.luma)
            record["hidden_for_color"].append(sample.hidden)
        out["frames"].append(record)
        rendered.append(cv2.cvtColor(draw_review_frame(frame_bgr, record), cv2.COLOR_BGR2RGB))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2) + "\n")
    write_h264_video(rendered, Path(args.output_video), args.fps)
    counts = {team: sum(r["teams"].count(team) for r in out["frames"]) for team in TEAM_COLORS}
    print(json.dumps({"output_json": str(output_json), "output_video": args.output_video, "counts": counts,
                      "luma_threshold": threshold, "luma_centers": out["luma_centers"]}, indent=2))


if __name__ == "__main__":
    main()
