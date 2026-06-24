from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


def add_repo_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "models" / "sam_3d_body"))


def rect_masks(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        mask = np.zeros((height, width), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        masks.append(mask)
    if not masks:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.stack(masks, axis=0)


def write_h264(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="error",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for path in frame_paths:
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)


def palette_color(obj_id: int) -> list[int]:
    from utils.painter import color_list

    return color_list[(obj_id + 4) % len(color_list)]


def frame_colors(record: dict, ids: list[int]) -> list[list[int]]:
    raw_colors = record.get("team_colors", [])
    if len(raw_colors) == len(ids):
        return [[int(v) for v in color] for color in raw_colors]
    return [palette_color(obj_id) for obj_id in ids]


def save_meshes(
    outputs,
    faces,
    mesh_dir: Path,
    focal_dir: Path,
    frame_idx: int,
    ids: list[int],
    colors: list[list[int]],
) -> None:
    from sam_3d_body.visualization.renderer import Renderer

    if outputs is None:
        return
    for person_output, obj_id, rgb in zip(outputs, ids, colors):
        (mesh_dir / str(obj_id)).mkdir(parents=True, exist_ok=True)
        (focal_dir / str(obj_id)).mkdir(parents=True, exist_ok=True)
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        color = tuple(c / 255.0 for c in rgb)
        mesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"],
            person_output["pred_cam_t"],
            color,
        )
        mesh.export(mesh_dir / str(obj_id) / f"{frame_idx:08d}.ply")
        focal = {
            "focal_length": float(np.asarray(person_output["focal_length"]).reshape(-1)[0]),
            "camera": [float(x) for x in np.asarray(person_output["pred_cam_t"]).reshape(-1)],
            "color_rgb": [int(c) for c in rgb],
        }
        (focal_dir / str(obj_id) / f"{frame_idx:08d}.json").write_text(json.dumps(focal, indent=2) + "\n")


def render_mesh_overlay(img_bgr: np.ndarray, outputs, faces, ids: list[int], colors: list[list[int]]) -> np.ndarray:
    from sam_3d_body.visualization.renderer import Renderer

    if outputs is None:
        return img_bgr
    try:
        depths = np.stack([person_output["pred_cam_t"] for person_output in outputs], axis=0)[:, 2]
    except Exception:
        return img_bgr

    order = np.argsort(-depths)
    sorted_outputs = [outputs[idx] for idx in order]
    sorted_ids = [ids[idx] for idx in order]
    sorted_colors = [colors[idx] for idx in order]

    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0
    for person_output, obj_id, rgb in zip(sorted_outputs, sorted_ids, sorted_colors):
        vertices = person_output["pred_vertices"] + person_output["pred_cam_t"]
        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)
        all_colors.append(rgb)

    if not all_vertices:
        return img_bgr

    all_vertices = np.concatenate(all_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)
    fake_cam_t = (np.max(all_vertices, axis=0) + np.min(all_vertices, axis=0)) / 2
    all_vertices = all_vertices - fake_cam_t

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    renderer = Renderer(focal_length=sorted_outputs[-1]["focal_length"], faces=all_faces)
    overlay_rgb = renderer(
        all_vertices,
        fake_cam_t,
        img_rgb,
        mesh_base_color=all_colors,
        scene_bg_color=(0, 0, 0),
    )
    overlay_bgr = cv2.cvtColor((overlay_rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return overlay_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="third_party/sam-body4d")
    parser.add_argument("--config", default="third_party/sam-body4d/configs/body4d.yaml")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--sam3-json", default="outputs/sam3_text_lacrosse_player_masks.json")
    parser.add_argument("--output-dir", default="outputs/sam_body4d_sam3_boxes")
    parser.add_argument("--output-video", default="outputs/sam_body4d_sam3_boxes_h264.mp4")
    parser.add_argument("--render-mode", choices=["overlay", "white"], default="overlay")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-detections-per-frame", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    add_repo_paths(repo_root)

    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from models.sam_3d_body.tools.vis_utils import visualize_sample_together

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rendered_dir = out_dir / "rendered_frames"
    mesh_dir = out_dir / "mesh_4d_individual"
    focal_dir = out_dir / "focal_4d_individual"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    focal_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    meta = json.loads(Path(args.sam3_json).read_text())
    frame_meta = meta["frames"]
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
        frame_meta = frame_meta[: args.max_frames]

    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body.ckpt_path,
        device=str(device),
        mhr_path=cfg.sam_3d_body.mhr_path,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    rendered_paths = []
    all_ids = set()
    for frame_idx, (frame_path, record) in enumerate(zip(frame_paths, frame_meta)):
        img = cv2.imread(str(frame_path))
        height, width = img.shape[:2]
        ids = [int(v) for v in record.get("object_ids", [])]
        boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
        scores = np.asarray(record.get("scores", [1.0] * len(ids)), dtype=np.float32)
        valid = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 - x1 >= 8 and y2 - y1 >= 16 and (i >= len(scores) or scores[i] >= args.score_threshold):
                valid.append(i)
        if args.max_detections_per_frame and len(valid) > args.max_detections_per_frame:
            areas = []
            for i in valid:
                x1, y1, x2, y2 = boxes[i]
                score = float(scores[i]) if i < len(scores) else 1.0
                areas.append((float((x2 - x1) * (y2 - y1)) * max(score, 0.05), i))
            valid = [i for _, i in sorted(areas, reverse=True)[: args.max_detections_per_frame]]
            valid.sort()
        ids = [ids[i] for i in valid]
        boxes = boxes[valid] if len(valid) else np.zeros((0, 4), dtype=np.float32)
        colors = [frame_colors(record, [int(v) for v in record.get("object_ids", [])])[i] for i in valid]
        all_ids.update(ids)

        if len(ids):
            masks = rect_masks(boxes, height, width)
            with torch.inference_mode():
                outputs = estimator.process_one_image(
                    str(frame_path),
                    bboxes=boxes,
                    masks=masks,
                    use_mask=True,
                    inference_type="body",
            )
            if args.render_mode == "overlay":
                rendered = render_mesh_overlay(img, outputs, estimator.faces, ids, colors)
            else:
                rendered = visualize_sample_together(img, outputs, estimator.faces, ids)
            save_meshes(outputs, estimator.faces, mesh_dir, focal_dir, frame_idx, ids, colors)
        else:
            rendered = img if args.render_mode == "overlay" else np.ones_like(img) * 255

        out_path = rendered_dir / f"{frame_idx:08d}.jpg"
        cv2.imwrite(str(out_path), rendered.astype(np.uint8))
        rendered_paths.append(out_path)

    write_h264(rendered_paths, Path(args.output_video), fps=args.fps)
    summary = {
        "output_video": args.output_video,
        "output_dir": str(out_dir),
        "frames": len(rendered_paths),
        "render_mode": args.render_mode,
        "object_ids": sorted(all_ids),
        "mesh_files": len(list(mesh_dir.glob("*/*.ply"))),
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
