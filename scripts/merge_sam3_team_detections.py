from __future__ import annotations

import argparse
import json
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
class Detection:
    prompt_team: str
    source_id: int
    object_id: int
    frame: int
    box: np.ndarray
    score: float
    mask: np.ndarray
    avg_rgb: list[float]
    avg_luma: float
    area: int


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


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_mask(mask_dir: Path, frame_idx: int, object_ids: list[int], det_idx: int, height: int, width: int) -> np.ndarray:
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


def average_color(frame_rgb: np.ndarray, mask: np.ndarray, box: np.ndarray) -> tuple[list[float], float, int]:
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)

    usable = mask.astype(bool)
    if usable.sum() < 20 and x2 > x1 and y2 > y1:
        usable = np.zeros((height, width), dtype=bool)
        usable[y1:y2, x1:x2] = True

    pixels = frame_rgb[usable]
    if pixels.size == 0:
        return [0.0, 0.0, 0.0], 0.0, 0
    avg = pixels.mean(axis=0)
    luma = float(0.2126 * avg[0] + 0.7152 * avg[1] + 0.0722 * avg[2])
    return [float(v) for v in avg], luma, int(len(pixels))


def load_detections(
    json_path: Path,
    mask_dir: Path,
    frames: list[np.ndarray],
    prompt_team: str,
    id_offset: int,
    min_width: float,
    min_height: float,
    score_threshold: float,
) -> list[list[Detection]]:
    meta = json.loads(json_path.read_text())
    detections_by_frame: list[list[Detection]] = [[] for _ in frames]
    for frame_idx, record in enumerate(meta["frames"][: len(frames)]):
        frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        object_ids = [int(v) for v in record.get("object_ids", [])]
        boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
        scores = np.asarray(record.get("scores", []), dtype=np.float32)
        if scores.size != len(boxes):
            scores = np.ones((len(boxes),), dtype=np.float32)
        for det_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 - x1 < min_width or y2 - y1 < min_height or scores[det_idx] < score_threshold:
                continue
            mask = load_mask(mask_dir, frame_idx, object_ids, det_idx, height, width)
            avg_rgb, avg_luma, area = average_color(frame_rgb, mask, box)
            source_id = object_ids[det_idx]
            detections_by_frame[frame_idx].append(
                Detection(
                    prompt_team=prompt_team,
                    source_id=source_id,
                    object_id=id_offset + source_id,
                    frame=frame_idx,
                    box=box.astype(np.float32),
                    score=float(scores[det_idx]),
                    mask=mask,
                    avg_rgb=avg_rgb,
                    avg_luma=avg_luma,
                    area=area,
                )
            )
    return detections_by_frame


def choose_detection(cluster: list[Detection], luma_threshold: float) -> tuple[Detection, str, str]:
    prompt_teams = {det.prompt_team for det in cluster}
    best_by_score = max(cluster, key=lambda det: det.score)
    if len(prompt_teams) == 1:
        return best_by_score, best_by_score.prompt_team, "prompt"

    weighted_luma = sum(det.avg_luma * max(det.area, 1) for det in cluster) / sum(max(det.area, 1) for det in cluster)
    color_team = "black" if weighted_luma < luma_threshold else "white"
    matching = [det for det in cluster if det.prompt_team == color_team]
    if matching:
        return max(matching, key=lambda det: det.score), color_team, "tie_luma"
    return best_by_score, color_team, "tie_luma_no_matching_prompt"


def merge_frame(
    detections: list[Detection],
    luma_threshold: float,
    box_iou_threshold: float,
    mask_iou_threshold: float,
) -> list[tuple[Detection, str, str]]:
    remaining = sorted(detections, key=lambda det: det.score, reverse=True)
    merged = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        keep = []
        for det in remaining:
            overlaps = box_iou(seed.box, det.box) >= box_iou_threshold or mask_iou(seed.mask, det.mask) >= mask_iou_threshold
            if overlaps:
                cluster.append(det)
            else:
                keep.append(det)
        remaining = keep
        merged.append(choose_detection(cluster, luma_threshold))
    return merged


def draw_review_frame(frame_bgr: np.ndarray, record: dict, team_colors: dict[str, list[int]]) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, team, box, avg_luma, source in zip(
        record["object_ids"],
        record["teams"],
        record["boxes"],
        record["avg_luma"],
        record["decision_sources"],
    ):
        color_rgb = team_colors.get(team, team_colors["unknown"])
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(
            out,
            f"{team}:{obj_id} {avg_luma:.0f} {source}",
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
    parser.add_argument("--black-json", required=True)
    parser.add_argument("--black-mask-dir", required=True)
    parser.add_argument("--white-json", required=True)
    parser.add_argument("--white-mask-dir", required=True)
    parser.add_argument("--output-json", default="outputs/sam3_team_player_detections.json")
    parser.add_argument("--output-video", default="outputs/sam3_team_player_detections.mp4")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--luma-threshold", type=float, default=150.0)
    parser.add_argument("--box-iou-threshold", type=float, default=0.45)
    parser.add_argument("--mask-iou-threshold", type=float, default=0.25)
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=16.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--team-colors", default=None, help="Example: black=255,80,40;white=70,170,255")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    frames = [cv2.imread(str(path)) for path in frame_paths]
    team_colors = parse_color_map(args.team_colors)

    black = load_detections(
        Path(args.black_json),
        Path(args.black_mask_dir),
        frames,
        "black",
        1000,
        args.min_width,
        args.min_height,
        args.score_threshold,
    )
    white = load_detections(
        Path(args.white_json),
        Path(args.white_mask_dir),
        frames,
        "white",
        2000,
        args.min_width,
        args.min_height,
        args.score_threshold,
    )

    metadata = {
        "model": "sam3_team_merge",
        "texts": {
            "black": json.loads(Path(args.black_json).read_text()).get("text"),
            "white": json.loads(Path(args.white_json).read_text()).get("text"),
        },
        "fps": args.fps,
        "team_colors": team_colors,
        "luma_threshold": args.luma_threshold,
        "frames": [],
    }
    rendered = []
    for frame_idx, frame_bgr in enumerate(frames):
        merged = merge_frame(
            black[frame_idx] + white[frame_idx],
            args.luma_threshold,
            args.box_iou_threshold,
            args.mask_iou_threshold,
        )
        record = {
            "frame": frame_idx,
            "object_ids": [],
            "scores": [],
            "boxes": [],
            "teams": [],
            "team_colors": [],
            "avg_rgb": [],
            "avg_luma": [],
            "prompt_teams": [],
            "source_ids": [],
            "decision_sources": [],
        }
        for det, team, decision_source in merged:
            record["object_ids"].append(int(det.object_id))
            record["scores"].append(float(det.score))
            record["boxes"].append([float(v) for v in det.box.tolist()])
            record["teams"].append(team)
            record["team_colors"].append(team_colors.get(team, team_colors["unknown"]))
            record["avg_rgb"].append(det.avg_rgb)
            record["avg_luma"].append(float(det.avg_luma))
            record["prompt_teams"].append(det.prompt_team)
            record["source_ids"].append(int(det.source_id))
            record["decision_sources"].append(decision_source)
        metadata["frames"].append(record)
        rendered.append(draw_review_frame(frame_bgr, record, team_colors))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metadata, indent=2) + "\n")
    write_h264(rendered, Path(args.output_video), args.fps)
    counts = {
        team: sum(record["teams"].count(team) for record in metadata["frames"])
        for team in sorted(team_colors)
    }
    print(json.dumps({"output_json": str(output_json), "output_video": args.output_video, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
