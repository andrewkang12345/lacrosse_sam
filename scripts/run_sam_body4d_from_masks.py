from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


def add_repo_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "models" / "sam_3d_body"))


def sorted_paths(directory: Path, pattern: str) -> list[Path]:
    return sorted(directory.glob(pattern))


def unique_object_ids(mask_paths: list[Path]) -> list[int]:
    ids: set[int] = set()
    from PIL import Image

    for path in mask_paths:
        arr = np.array(Image.open(path).convert("P"))
        ids.update(int(v) for v in np.unique(arr) if int(v) > 0)
    return sorted(ids)


def write_h264_from_jpgs(frames_dir: Path, output_path: Path, fps: float) -> None:
    import imageio.v2 as imageio

    frame_paths = sorted_paths(frames_dir, "*.jpg")
    if not frame_paths:
        raise RuntimeError(f"No rendered frames found in {frames_dir}")
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
            frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            writer.append_data(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="third_party/sam-body4d")
    parser.add_argument("--config", default="third_party/sam-body4d/configs/body4d.yaml")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--masks-dir", default="outputs/sam2_label_masks")
    parser.add_argument("--output-dir", default="outputs/sam_body4d")
    parser.add_argument("--output-video", default="outputs/sam_body4d_meshes_h264.mp4")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-fov", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    add_repo_paths(repo_root)

    from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results
    from models.sam_3d_body.tools.vis_utils import visualize_sample_together

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rendered_dir = output_dir / "rendered_frames"
    mesh_dir = output_dir / "mesh_4d_individual"
    focal_dir = output_dir / "focal_4d_individual"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    focal_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted_paths(Path(args.frames_dir), "frame_*.jpg")
    mask_paths = sorted_paths(Path(args.masks_dir), "*.png")
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
        mask_paths = mask_paths[: args.max_frames]
    if len(frame_paths) != len(mask_paths):
        raise RuntimeError(f"Frame/mask count mismatch: {len(frame_paths)} frames, {len(mask_paths)} masks")
    if not frame_paths:
        raise RuntimeError("No frames found")

    object_ids = unique_object_ids(mask_paths)
    if not object_ids:
        raise RuntimeError("No object IDs found in label masks")
    for obj_id in object_ids:
        (mesh_dir / str(obj_id)).mkdir(parents=True, exist_ok=True)
        (focal_dir / str(obj_id)).mkdir(parents=True, exist_ok=True)

    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body.ckpt_path,
        device=str(device),
        mhr_path=cfg.sam_3d_body.mhr_path,
    )

    fov_estimator = None
    if args.use_fov:
        from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name="moge2", device=device, path=cfg.sam_3d_body.fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    metadata = {
        "frames": len(frame_paths),
        "object_ids": object_ids,
        "batch_size": args.batch_size,
        "use_fov": bool(args.use_fov),
        "rendered_frames": str(rendered_dir),
        "mesh_dir": str(mesh_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    mhr_shape_scale_dict: dict = {}
    for start in range(0, len(frame_paths), args.batch_size):
        end = min(len(frame_paths), start + args.batch_size)
        batch_images = [str(p) for p in frame_paths[start:end]]
        batch_masks = [str(p) for p in mask_paths[start:end]]
        occ_dict = {obj_id: [1] * len(batch_images) for obj_id in object_ids}

        with torch.inference_mode():
            mask_outputs, id_batch, empty_frame_list = process_image_with_mask(
                estimator,
                batch_images,
                batch_masks,
                idx_path={},
                idx_dict={},
                mhr_shape_scale_dict=mhr_shape_scale_dict,
                occ_dict=occ_dict,
            )

        for local_idx, image_path in enumerate(batch_images):
            global_idx = start + local_idx
            img = cv2.imread(image_path)
            if local_idx in empty_frame_list:
                rendered = img
                outputs = None
                ids = []
            else:
                outputs = mask_outputs[local_idx]
                ids = id_batch[local_idx]
                rendered = visualize_sample_together(img, outputs, estimator.faces, ids)
                save_mesh_results(
                    outputs=outputs,
                    faces=estimator.faces,
                    save_dir=str(mesh_dir),
                    focal_dir=str(focal_dir),
                    image_path=f"{global_idx:08d}.jpg",
                    id_current=ids,
                )
            cv2.imwrite(str(rendered_dir / f"{global_idx:08d}.jpg"), rendered.astype(np.uint8))

    write_h264_from_jpgs(rendered_dir, Path(args.output_video), fps=args.fps)
    print(json.dumps({"output_video": args.output_video, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
