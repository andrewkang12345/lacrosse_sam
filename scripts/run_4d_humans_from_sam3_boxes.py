from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import shutil
import sys
import inspect
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def patch_legacy_smpl_pickle_compat() -> None:
    if not hasattr(inspect, "getargspec"):
        ArgSpec = collections.namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec

    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def add_repo_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))


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
            bgr = cv2.imread(str(path))
            writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def valid_sam3_detections(
    record: dict,
    min_width: float,
    min_height: float,
    score_threshold: float,
) -> tuple[list[int], np.ndarray, list[list[int]]]:
    ids = [int(v) for v in record.get("object_ids", [])]
    boxes = np.asarray(record.get("boxes", []), dtype=np.float32)
    scores = np.asarray(record.get("scores", []), dtype=np.float32)
    if boxes.size == 0:
        return [], np.zeros((0, 4), dtype=np.float32), []
    if scores.size != len(boxes):
        scores = np.ones((len(boxes),), dtype=np.float32)
    raw_colors = record.get("team_colors", [])
    has_colors = len(raw_colors) == len(boxes)

    keep = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if x2 - x1 >= min_width and y2 - y1 >= min_height and scores[idx] >= score_threshold:
            keep.append(idx)
    colors = [[int(v) for v in raw_colors[i]] if has_colors else [int(255 * c) for c in LIGHT_BLUE] for i in keep]
    return [ids[i] for i in keep], boxes[keep] if keep else np.zeros((0, 4), dtype=np.float32), colors


def render_overlay(
    img_bgr: np.ndarray,
    mesh_renderer,
    verts: list[np.ndarray],
    cams: list[np.ndarray],
    colors: list[list[int]],
    focal_length: float,
) -> np.ndarray:
    if not verts:
        return img_bgr
    import pyrender
    from hmr2.utils.renderer import create_raymond_lights

    height, width = img_bgr.shape[:2]
    offscreen = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))

    with contextlib.redirect_stdout(io.StringIO()):
        for idx, (vertices, cam_t, rgb) in enumerate(zip(verts, cams, colors)):
            color = tuple(float(c) / 255.0 for c in rgb)
            mesh = mesh_renderer.vertices_to_trimesh(vertices, cam_t.copy(), color)
            scene.add(pyrender.Mesh.from_trimesh(mesh), f"mesh_{idx}")

    camera = pyrender.IntrinsicsCamera(
        fx=focal_length,
        fy=focal_length,
        cx=width / 2.0,
        cy=height / 2.0,
        zfar=1e12,
    )
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    mesh_renderer.add_point_lighting(scene, camera_node)
    mesh_renderer.add_lighting(scene, camera_node)
    for node in create_raymond_lights():
        scene.add_node(node)

    rgba, _ = offscreen.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgba = rgba.astype(np.float32) / 255.0
    offscreen.delete()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:4]
    overlay_rgb = img_rgb * (1.0 - alpha) + rgba[:, :, :3] * alpha
    return cv2.cvtColor((overlay_rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="third_party/4D-Humans")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--sam3-json", default="outputs/sam3_text_player_masks.json")
    parser.add_argument("--output-dir", default="outputs/4d_humans_sam3_player")
    parser.add_argument("--output-video", default="outputs/4d_humans_sam3_player_h264.mp4")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-width", type=float, default=8.0)
    parser.add_argument("--min-height", type=float, default=16.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--save-mesh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    patch_legacy_smpl_pickle_compat()
    add_repo_paths(repo_root)

    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
    from hmr2.utils import recursive_to
    from hmr2.utils.renderer import Renderer, cam_crop_to_full

    download_models(CACHE_DIR_4DHUMANS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_torch_load = torch.load

    def torch_load_legacy_checkpoint(*load_args, **load_kwargs):
        if load_kwargs.get("weights_only") is None:
            load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    try:
        torch.load = torch_load_legacy_checkpoint
        model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    finally:
        torch.load = original_torch_load
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rendered_dir = output_dir / "rendered_frames"
    mesh_dir = output_dir / "meshes_obj"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    meta = json.loads(Path(args.sam3_json).read_text())
    frame_meta = meta["frames"]
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
        frame_meta = frame_meta[: args.max_frames]

    rendered_paths: list[Path] = []
    all_ids: set[int] = set()
    mesh_count = 0
    for frame_idx, (frame_path, record) in enumerate(zip(frame_paths, frame_meta)):
        img_bgr = cv2.imread(str(frame_path))
        ids, boxes, colors = valid_sam3_detections(record, args.min_width, args.min_height, args.score_threshold)
        all_ids.update(ids)

        all_verts: list[np.ndarray] = []
        all_cam_t: list[np.ndarray] = []
        all_colors: list[list[int]] = []
        per_person = []
        scaled_focal_length = float(model_cfg.EXTRA.FOCAL_LENGTH)

        if len(ids):
            dataset = ViTDetDataset(model_cfg, img_bgr, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            with contextlib.redirect_stdout(io.StringIO()):
                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    with torch.inference_mode():
                        out = model(batch)

                    pred_cam = out["pred_cam"]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    scaled_focal_length_tensor = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    scaled_focal_length = float(scaled_focal_length_tensor.detach().cpu().reshape(-1)[0])
                    pred_cam_t_full = cam_crop_to_full(
                        pred_cam,
                        box_center,
                        box_size,
                        img_size,
                        scaled_focal_length_tensor,
                    ).detach().cpu().numpy()

                    verts_batch = out["pred_vertices"].detach().cpu().numpy()
                    person_indices = batch["personid"].detach().cpu().numpy().astype(int).tolist()
                    for local_idx, person_index in enumerate(person_indices):
                        obj_id = ids[person_index]
                        verts = verts_batch[local_idx]
                        cam_t = pred_cam_t_full[local_idx]
                        all_verts.append(verts)
                        all_cam_t.append(cam_t)
                        all_colors.append(colors[person_index])
                        per_person.append({"object_id": obj_id, "box": boxes[person_index].tolist()})

                        if args.save_mesh:
                            obj_dir = mesh_dir / str(obj_id)
                            obj_dir.mkdir(parents=True, exist_ok=True)
                            color = tuple(float(c) / 255.0 for c in colors[person_index])
                            mesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), color)
                            mesh.export(obj_dir / f"{frame_idx:08d}.obj")
                            mesh_count += 1

        rendered = render_overlay(img_bgr, renderer, all_verts, all_cam_t, all_colors, scaled_focal_length)
        out_path = rendered_dir / f"{frame_idx:08d}.jpg"
        cv2.imwrite(str(out_path), rendered)
        rendered_paths.append(out_path)

    write_h264(rendered_paths, Path(args.output_video), args.fps)
    summary = {
        "output_video": args.output_video,
        "output_dir": str(output_dir),
        "frames": len(rendered_paths),
        "object_ids": sorted(all_ids),
        "mesh_files": mesh_count,
        "source_sam3_json": args.sam3_json,
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
