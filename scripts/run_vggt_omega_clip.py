"""Run VGGT-Omega on a frame directory and save cameras, depth, confidence and colors.

The saved npz is the input for `fit_field_plane_landmarks.py` and
`render_vggt_scene_3d.py`. Cameras follow the VGGT-Omega convention:
`extrinsic` is camera-from-world (OpenCV axes) and `intrinsic` is expressed in
model-image pixels (`model_hw`). Use `original_hw` to map back to frame pixels.

Clips longer than --chunk-size frames are reconstructed in overlapping chunks.
Each chunk is mapped into the first chunk's coordinates by the similarity that
best aligns the 3D points of the frames the two chunks share.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output .npz path.")
    parser.add_argument("--omega-repo", type=Path, default=Path("third_party/vggt_omega"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt_omega/vggt_omega_1b_512.pt"))
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=0, help="0 keeps every frame.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=170, help="Most frames in one forward pass (about 6.4 GB + 0.08 GB per frame).")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Frames shared by neighbouring chunks.")
    return parser.parse_args()


def chunk_ranges(frames: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if frames <= size:
        return [(0, frames)]
    ranges, start = [], 0
    while True:
        stop = min(start + size, frames)
        ranges.append((max(0, stop - size), stop))
        if stop == frames:
            return ranges
        start = stop - overlap


def world_points(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    z = depth[rows, cols]
    camera = np.column_stack([(cols - intrinsic[0, 2]) / intrinsic[0, 0] * z, (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z, z])
    return (camera - extrinsic[:3, 3]) @ extrinsic[:3, :3]


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale, rotation and translation with target ~ scale * rotation @ source + translation."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mean_s, target - mean_t
    U, D, Vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    rotation = U @ sign @ Vt
    scale = float(np.trace(np.diag(D) @ sign) / max((a**2).sum() / len(a), 1e-12))
    return scale, rotation, mean_t - scale * rotation @ mean_s


def align_chunk(reference: dict, chunk: dict, frames: range, offset_reference: int, offset_chunk: int) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Similarity from a chunk's coordinates to the reference's, from the frames both reconstruct."""
    src, dst = [], []
    for f in frames:
        i, j = f - offset_reference, f - offset_chunk
        rows, cols = np.mgrid[4 : reference["depth"].shape[1] : 8, 4 : reference["depth"].shape[2] : 8]
        rows, cols = rows.ravel(), cols.ravel()
        conf_i, conf_j = reference["depth_conf"][i][rows, cols], chunk["depth_conf"][j][rows, cols]
        keep = (conf_i >= np.median(conf_i)) & (conf_j >= np.median(conf_j)) & (reference["depth"][i][rows, cols] > 0) & (chunk["depth"][j][rows, cols] > 0)
        dst.append(world_points(reference["depth"][i], reference["intrinsic"][i], reference["extrinsic"][i], rows[keep], cols[keep]))
        src.append(world_points(chunk["depth"][j], chunk["intrinsic"][j], chunk["extrinsic"][j], rows[keep], cols[keep]))
    src, dst = np.concatenate(src), np.concatenate(dst)
    keep = np.ones(len(src), bool)
    for _ in range(6):  # trimmed least squares: moving players and bad depth drop out
        scale, rotation, translation = umeyama(src[keep], dst[keep])
        error = np.linalg.norm(scale * src @ rotation.T + translation - dst, axis=1)
        keep = error <= np.percentile(error, 80)
    spread = float(np.median(np.linalg.norm(dst - np.median(dst, axis=0), axis=1)))
    return scale, rotation, translation, {"scale": scale, "points": int(keep.sum()), "median_error_over_scene": float(np.median(error) / max(spread, 1e-9))}


def apply_similarity(chunk: dict, scale: float, rotation: np.ndarray, translation: np.ndarray) -> None:
    R = chunk["extrinsic"][:, :3, :3] @ rotation.T
    centres = scale * (-np.einsum("sji,sj->si", chunk["extrinsic"][:, :3, :3], chunk["extrinsic"][:, :3, 3])) @ rotation.T + translation
    chunk["extrinsic"] = np.concatenate([R, -np.einsum("sij,sj->si", R, centres)[..., None]], axis=2).astype(np.float32)
    chunk["depth"] = chunk["depth"] * float(scale)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.omega_repo.resolve()))
    from PIL import Image
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    frame_paths = sorted(args.frames_dir.glob("frame_*.jpg"))[:: max(1, args.frame_stride)]
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.jpg files in {args.frames_dir}")
    with Image.open(frame_paths[0]) as first:
        original_w, original_h = first.size

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    ranges = chunk_ranges(len(frame_paths), args.chunk_size, args.chunk_overlap)
    merged: dict[str, np.ndarray] = {}
    previous, previous_start, alignments = None, 0, []
    start = time.time()
    for chunk_index, (first, stop) in enumerate(ranges):
        images = load_and_preprocess_images([str(p) for p in frame_paths[first:stop]], image_resolution=args.image_resolution).to("cuda")
        with torch.inference_mode():
            predictions = model(images)
            extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
        torch.cuda.synchronize()
        model_h, model_w = (int(v) for v in predictions["images"].shape[-2:])
        depth = predictions["depth"][0].float().cpu().numpy()
        if depth.ndim == 4:
            depth = depth[..., 0]
        chunk = {
            "extrinsic": extrinsic[0].float().cpu().numpy(),
            "intrinsic": intrinsic[0].float().cpu().numpy(),
            "depth": depth.astype(np.float32),
            "depth_conf": predictions["depth_conf"][0].float().cpu().numpy().astype(np.float32),
            "colors": (predictions["images"][0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8),
        }
        del predictions, images, extrinsic, intrinsic
        torch.cuda.empty_cache()
        if not merged:
            merged = {key: np.zeros((len(frame_paths), *value.shape[1:]), value.dtype) for key, value in chunk.items()}
        keep_from = first
        if previous is not None:
            shared = range(first, previous_start + len(previous["depth"]))
            scale, rotation, translation, info = align_chunk(previous, chunk, shared, previous_start, first)
            apply_similarity(chunk, scale, rotation, translation)
            alignments.append({"chunk": chunk_index, "first_frame": first, **info})
            keep_from = shared.start + len(shared) // 2  # each shared frame comes from the chunk it is more central in
        for key, value in chunk.items():
            merged[key][keep_from:stop] = value[keep_from - first :]
        previous, previous_start = chunk, first
        print(json.dumps({"chunk": chunk_index + 1, "of": len(ranges), "frames": [first, stop], "seconds": round(time.time() - start, 1)}), flush=True)
    elapsed = time.time() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        frame_names=np.asarray([p.name for p in frame_paths]),
        model_hw=np.asarray([model_h, model_w], dtype=np.int32),
        original_hw=np.asarray([original_h, original_w], dtype=np.int32),
        **merged,
    )
    summary = {
        "output": str(args.output),
        "frames": len(frame_paths),
        "model_hw": [model_h, model_w],
        "original_hw": [original_h, original_w],
        "chunks": [list(r) for r in ranges],
        "chunk_alignment": alignments,
        "inference_seconds": round(elapsed, 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
