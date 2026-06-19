from __future__ import annotations

import argparse
import collections.abc
import json
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


DEFAULT_TEAM_COLORS = {
    "team_a": [255, 80, 40],
    "team_b": [70, 170, 255],
    "team_c": [245, 210, 55],
    "unknown": [180, 180, 180],
}


@dataclass
class ReIDSample:
    frame_idx: int
    det_idx: int
    object_id: int
    box: list[float]
    score: float
    visible_coverage: float
    max_overlap: float
    occluded: bool
    embedding: np.ndarray | None = None
    cluster: int | None = None


def install_torch_six_compat() -> None:
    if "torch._six" in sys.modules:
        return
    module = types.ModuleType("torch._six")
    module.container_abcs = collections.abc
    sys.modules["torch._six"] = module


def add_transreid_path(transreid_root: Path) -> None:
    install_torch_six_compat()
    sys.path.insert(0, str(transreid_root.resolve()))


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


def box_intersection_over_area(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    return inter / area if area > 0 else 0.0


def clipped_box(box: np.ndarray, height: int, width: int, pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    x1 -= pad * bw
    x2 += pad * bw
    y1 -= pad * bh
    y2 += pad * bh
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(width, int(np.ceil(x2))),
        min(height, int(np.ceil(y2))),
    )


def crop_for_reid(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    input_size: tuple[int, int],
    pad: float,
    mask_background: bool,
) -> torch.Tensor:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clipped_box(box, height, width, pad)
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    if mask_background:
        crop_mask = mask[y1:y2, x1:x2]
        if crop_mask.any():
            crop = crop.copy()
            crop[~crop_mask] = 127
    input_h, input_w = input_size
    resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def collect_samples_and_crops(
    frames_bgr: list[np.ndarray],
    sam3_meta: dict,
    mask_dir: Path,
    input_size: tuple[int, int],
    min_width: float,
    min_height: float,
    score_threshold: float,
    min_visible_coverage: float,
    overlap_threshold: float,
    crop_pad: float,
    mask_background: bool,
) -> tuple[list[ReIDSample], list[torch.Tensor]]:
    samples: list[ReIDSample] = []
    crops: list[torch.Tensor] = []
    for frame_idx, record in enumerate(sam3_meta["frames"][: len(frames_bgr)]):
        frame_bgr = frames_bgr[frame_idx]
        height, width = frame_bgr.shape[:2]
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
            box_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
            coverage = float(mask.sum() / box_area)
            max_overlap = 0.0
            for other_idx in valid_indices:
                if other_idx == det_idx:
                    continue
                max_overlap = max(max_overlap, box_intersection_over_area(box, boxes[other_idx]))
            occluded = coverage < min_visible_coverage or max_overlap >= overlap_threshold
            crops.append(crop_for_reid(frame_bgr, mask, box, input_size, crop_pad, mask_background))
            samples.append(
                ReIDSample(
                    frame_idx=frame_idx,
                    det_idx=det_idx,
                    object_id=int(object_ids[det_idx]),
                    box=[float(v) for v in box.tolist()],
                    score=float(scores[det_idx]),
                    visible_coverage=coverage,
                    max_overlap=max_overlap,
                    occluded=occluded,
                )
            )
    return samples, crops


def infer_checkpoint_shape(checkpoint_path: Path) -> tuple[int, int]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    num_class = 0
    camera_num = 0
    for key, value in state.items():
        clean = key.replace("module.", "")
        if clean == "classifier.weight":
            num_class = int(value.shape[0])
        elif clean == "base.sie_embed":
            camera_num = int(value.shape[0])
    if num_class <= 0:
        raise ValueError(f"Could not infer classifier size from {checkpoint_path}")
    return num_class, camera_num


def load_transreid_model(transreid_root: Path, config_file: Path, checkpoint_path: Path, device: torch.device):
    add_transreid_path(transreid_root)
    from config import cfg
    from model import make_model

    cfg.defrost()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.freeze()

    num_class, camera_num = infer_checkpoint_shape(checkpoint_path)
    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=0)
    model.load_param(str(checkpoint_path))
    model.to(device)
    model.eval()
    return model, tuple(int(v) for v in cfg.INPUT.SIZE_TEST), camera_num


def extract_embeddings(
    model,
    crops: list[torch.Tensor],
    device: torch.device,
    camera_num: int,
    batch_size: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = torch.stack(crops[start : start + batch_size], dim=0).to(device)
            cam_label = torch.zeros((batch.shape[0],), dtype=torch.long, device=device) if camera_num > 0 else None
            out = model(batch, cam_label=cam_label, view_label=None)
            out = F.normalize(out, dim=1)
            features.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0) if features else np.zeros((0, 1), dtype=np.float32)


def cluster_object_embeddings(
    samples: list[ReIDSample],
    clusters: int,
    seed: int,
) -> tuple[dict[int, int], dict[int, list[float]], dict[int, float]]:
    by_object: dict[int, list[np.ndarray]] = {}
    all_by_object: dict[int, list[np.ndarray]] = {}
    for sample in samples:
        if sample.embedding is None:
            continue
        all_by_object.setdefault(sample.object_id, []).append(sample.embedding)
        if not sample.occluded:
            by_object.setdefault(sample.object_id, []).append(sample.embedding)
    for obj_id, embeddings in all_by_object.items():
        by_object.setdefault(obj_id, embeddings)

    object_ids = sorted(by_object)
    if len(object_ids) < 2:
        labels = {obj_id: 0 for obj_id in object_ids}
        vectors = {obj_id: by_object[obj_id][0].tolist() for obj_id in object_ids}
        confidence = {obj_id: 1.0 for obj_id in object_ids}
        return labels, vectors, confidence

    clusters = max(1, min(int(clusters), len(object_ids)))
    vectors_np = []
    vectors = {}
    for obj_id in object_ids:
        arr = np.stack(by_object[obj_id], axis=0)
        vec = arr.mean(axis=0)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        vectors_np.append(vec)
        vectors[obj_id] = [float(v) for v in vec.tolist()]
    matrix = np.stack(vectors_np, axis=0)
    kmeans = KMeans(n_clusters=clusters, n_init=50, random_state=seed)
    raw_labels = kmeans.fit_predict(matrix)

    centers = kmeans.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    # Make labels deterministic without using color: the cluster with the leftmost
    # first appearance is team_a, the other is team_b.
    first_x: dict[int, float] = {}
    for obj_id, label in zip(object_ids, raw_labels):
        xs = [s.box[0] for s in samples if s.object_id == obj_id]
        first_x.setdefault(int(label), min(xs) if xs else 1e9)
    ordered_clusters = sorted(first_x, key=lambda cluster: first_x[cluster])
    cluster_to_team = {cluster: idx for idx, cluster in enumerate(ordered_clusters)}

    labels = {}
    confidence = {}
    for obj_id, raw_label, vec in zip(object_ids, raw_labels, matrix):
        team_label = cluster_to_team[int(raw_label)]
        labels[obj_id] = int(team_label)
        sims = centers @ vec
        sorted_sims = np.sort(sims)
        confidence[obj_id] = float(sorted_sims[-1] - sorted_sims[-2]) if len(sorted_sims) > 1 else 1.0
    return labels, vectors, confidence


def draw_review_frame(frame_bgr: np.ndarray, record: dict, team_colors: dict[str, list[int]]) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, team, box, conf in zip(record["object_ids"], record["teams"], record["boxes"], record["reid_confidence"]):
        color_rgb = team_colors.get(team, team_colors["unknown"])
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(
            out,
            f"{team}:{obj_id} {conf:.2f}",
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


def write_review_sheet(frames_bgr: list[np.ndarray], output_path: Path, columns: int = 5) -> None:
    if not frames_bgr:
        return
    thumbs = [cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA) for frame in frames_bgr]
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = np.zeros((rows * 180, columns * 320, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // columns) * 180
        x = (idx % columns) * 320
        sheet[y : y + 180, x : x + 320] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transreid-root", default="third_party/TransReID")
    parser.add_argument("--config-file", default="third_party/TransReID/configs/MSMT17/vit_transreid_stride.yml")
    parser.add_argument("--checkpoint", default="checkpoints/transreid/msmt17_vit_transreid_stride.pth")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--sam3-json", default="outputs/sam3_text_player_instances.json")
    parser.add_argument("--instance-mask-dir", default="outputs/sam3_text_player_instance_masks")
    parser.add_argument("--output-json", default="outputs/sam3_team_transreid_detections.json")
    parser.add_argument("--output-video", default="outputs/sam3_team_transreid_detections_review.mp4")
    parser.add_argument("--review-sheet", default="outputs/sam3_team_transreid_detections_review_sheet.jpg")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--crop-pad", type=float, default=0.08)
    parser.add_argument("--no-mask-background", action="store_true")
    parser.add_argument("--min-visible-coverage", type=float, default=0.08)
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=16.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--team-colors", default=None, help="Example: team_a=255,80,40;team_b=70,170,255")
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copy-checkpoint-to", default="", help="Optional path for recording the exact TransReID checkpoint used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available; running TransReID on CPU.", file=sys.stderr)

    transreid_root = Path(args.transreid_root)
    checkpoint = Path(args.checkpoint)
    if args.copy_checkpoint_to:
        target = Path(args.copy_checkpoint_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.resolve() != target.resolve():
            shutil.copy2(checkpoint, target)

    model, input_size, camera_num = load_transreid_model(transreid_root, Path(args.config_file), checkpoint, device)
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    frames_bgr = [cv2.imread(str(path)) for path in frame_paths]
    sam3_meta = json.loads(Path(args.sam3_json).read_text())
    team_colors = parse_color_map(args.team_colors)

    samples, crops = collect_samples_and_crops(
        frames_bgr,
        sam3_meta,
        Path(args.instance_mask_dir),
        input_size,
        args.min_width,
        args.min_height,
        args.score_threshold,
        args.min_visible_coverage,
        args.overlap_threshold,
        args.crop_pad,
        not args.no_mask_background,
    )
    embeddings = extract_embeddings(model, crops, device, camera_num, args.batch_size)
    for sample, embedding in zip(samples, embeddings):
        sample.embedding = embedding

    object_clusters, object_embeddings, object_confidence = cluster_object_embeddings(samples, args.clusters, args.seed)
    team_names = {idx: f"team_{chr(ord('a') + idx)}" for idx in range(max(object_clusters.values(), default=0) + 1)}
    missing_colors = [name for name in team_names.values() if name not in team_colors]
    if missing_colors:
        raise ValueError(f"Missing colors for teams: {missing_colors}")
    for sample in samples:
        sample.cluster = object_clusters.get(sample.object_id, 0)

    samples_by_frame: dict[int, list[ReIDSample]] = {}
    for sample in samples:
        samples_by_frame.setdefault(sample.frame_idx, []).append(sample)

    metadata = {
        "model": "sam3_transreid_team_classifier",
        "source_json": args.sam3_json,
        "transreid_root": args.transreid_root,
        "transreid_config": args.config_file,
        "transreid_checkpoint": args.checkpoint,
        "fps": args.fps,
        "team_colors": team_colors,
        "clusters": args.clusters,
        "cluster_label_note": "team labels are TransReID appearance clusters; names are deterministic and not color-derived.",
        "mask_background": not args.no_mask_background,
        "min_visible_coverage": args.min_visible_coverage,
        "overlap_threshold": args.overlap_threshold,
        "object_teams": {str(obj_id): team_names[label] for obj_id, label in object_clusters.items()},
        "object_reid_confidence": {str(obj_id): float(conf) for obj_id, conf in object_confidence.items()},
        "object_reid_embeddings": {str(obj_id): vector for obj_id, vector in object_embeddings.items()},
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
            "reid_cluster": [],
            "reid_confidence": [],
            "visible_coverage": [],
            "max_overlap": [],
            "occluded_for_reid": [],
        }
        for sample in samples_by_frame.get(frame_idx, []):
            cluster = int(sample.cluster or 0)
            team = team_names[cluster]
            record["object_ids"].append(int(sample.object_id))
            record["scores"].append(float(sample.score))
            record["boxes"].append(sample.box)
            record["teams"].append(team)
            record["team_colors"].append(team_colors[team])
            record["reid_cluster"].append(cluster)
            record["reid_confidence"].append(float(object_confidence.get(sample.object_id, 0.0)))
            record["visible_coverage"].append(float(sample.visible_coverage))
            record["max_overlap"].append(float(sample.max_overlap))
            record["occluded_for_reid"].append(bool(sample.occluded))
        metadata["frames"].append(record)
        rendered.append(draw_review_frame(frame_bgr, record, team_colors))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metadata, indent=2) + "\n")
    write_h264(rendered, Path(args.output_video), args.fps)
    if args.review_sheet:
        stride = max(1, len(rendered) // 10)
        write_review_sheet(rendered[::stride][:10], Path(args.review_sheet))

    counts = {team: sum(record["teams"].count(team) for record in metadata["frames"]) for team in team_names.values()}
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_video": args.output_video,
                "review_sheet": args.review_sheet,
                "device": str(device),
                "detections": len(samples),
                "object_teams": metadata["object_teams"],
                "counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
