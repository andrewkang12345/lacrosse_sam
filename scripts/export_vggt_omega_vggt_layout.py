"""Export VGGT-Omega predictions and the planar field fit in the 518x518 VGGT layout.

The SAM-Body4D mesh scripts (render_sam_body4d_on_field.py, view_vggt_reconstruction.py)
read, from --output-dir:

  vggt_predictions_compact.npz  frame_indices, extrinsic, intrinsic, depth_map,
                                depth_conf on a 518x518 square, letterboxed image
  field_fit_vggt_layout.json    floor_plane {center, basis_u, basis_v},
                                plane_to_floor_homography, vggt_resolution, and the
                                path of the field_plane_fit.json it came from

This script resamples VGGT-Omega depth/confidence into that square layout (z-depth is
unchanged by resampling, intrinsics are rescaled) and writes the rigid plane->field
similarity from fit_field_plane_landmarks.py as the plane_to_floor_homography. The
meshes are therefore placed with the rigid planar fit, not the per-frame refined
cameras of field_homographies.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VGGT_LAYOUT_RESOLUTION = 518


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="npz from run_vggt_omega_clip.py")
    parser.add_argument("--field-fit", required=True, help="field_plane_fit.json from fit_field_plane_landmarks.py")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.predictions) as data:
        pred = {key: data[key] for key in data.files}
    fit = json.loads(Path(args.field_fit).read_text())
    model_h, model_w = (int(v) for v in pred["model_hw"])
    orig_h, orig_w = (int(v) for v in pred["original_hw"])
    res = VGGT_LAYOUT_RESOLUTION
    side = max(orig_h, orig_w)
    scale = res / side
    left, top = (side - orig_w) / 2.0, (side - orig_h) / 2.0

    # square layout pixel -> original pixel -> VGGT-Omega model pixel
    us, vs = np.meshgrid(np.arange(res) + 0.5, np.arange(res) + 0.5)
    ox, oy = us / scale - left, vs / scale - top
    map_x = (ox * model_w / orig_w - 0.5).astype(np.float32)
    map_y = (oy * model_h / orig_h - 0.5).astype(np.float32)
    outside = (ox < 0) | (ox >= orig_w) | (oy < 0) | (oy >= orig_h)

    S = pred["depth"].shape[0]
    depth = np.zeros((S, res, res, 1), dtype=np.float32)
    conf = np.zeros((S, res, res), dtype=np.float32)
    for i in range(S):
        d = cv2.remap(pred["depth"][i], map_x, map_y, cv2.INTER_NEAREST)
        c = cv2.remap(pred["depth_conf"][i], map_x, map_y, cv2.INTER_NEAREST)
        d[outside], c[outside] = 0.0, 0.0
        depth[i, :, :, 0], conf[i] = d, c

    K = pred["intrinsic"].astype(np.float64).copy()
    K[:, 0, :] *= orig_w / model_w  # model -> original pixels
    K[:, 1, :] *= orig_h / model_h
    K[:, 0, 2] += left  # original -> letterboxed square
    K[:, 1, 2] += top
    K[:, :2, :] *= scale

    all_frames = sorted(p.name for p in Path(args.frames_dir).glob("frame_*.jpg"))
    frame_indices = np.asarray([all_frames.index(str(n)) for n in pred["frame_names"]], dtype=np.int32)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "vggt_predictions_compact.npz",
        frame_indices=frame_indices,
        extrinsic=pred["extrinsic"].astype(np.float32),
        intrinsic=K.astype(np.float32),
        depth_map=depth,
        depth_conf=conf,
    )

    sim = fit["plane_to_field_similarity"]
    k, theta = float(np.exp(sim["log_k"])), sim["theta_rad"]
    H = np.asarray([[k * np.cos(theta), -k * np.sin(theta), sim["tx_ft"]], [k * np.sin(theta), k * np.cos(theta), sim["ty_ft"]], [0.0, 0.0, 1.0]])
    plane = fit["plane"]
    layout_fit = {
        "schema": "vggt_sam3_text_field_fit_v1",
        "source": "export_vggt_omega_vggt_layout.py",
        "vggt_npz": str(out_dir / "vggt_predictions_compact.npz"),
        "field_plane_fit": args.field_fit,
        "frames_dir": args.frames_dir,
        "vggt_resolution": res,
        "floor_plane": {"center": plane["center"], "basis_u": plane["e1"], "basis_v": plane["e2"], "normal_up": plane["normal"]},
        "plane_to_floor_homography": H.tolist(),
        "alignment_metrics": {"fit_model": "vggt_omega_plane_rigid_similarity", **fit["stage1_planar_fit"]},
    }
    (out_dir / "field_fit_vggt_layout.json").write_text(json.dumps(layout_fit, indent=2) + "\n")
    print(json.dumps({"npz": str(out_dir / "vggt_predictions_compact.npz"), "fit_json": str(out_dir / "field_fit_vggt_layout.json"), "frames": int(S)}, indent=2))


if __name__ == "__main__":
    main()
