from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = REPO_ROOT / "third_party" / "VGGT"
sys.path.insert(0, str(VGGT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from refit_vggt_field_with_sam2_faceoff_masks import (  # noqa: E402
    apply_homography,
    collect_faceoff_points,
    load_npz_mask,
    pixel_to_plane_uv,
)
from render_birds_eye_locations import dense_floor_model  # noqa: E402
from run_vggt_birds_eye import transform_plane_to_floor  # noqa: E402


GOAL_CENTERS = np.asarray([[12.0, 42.5], [188.0, 42.5]], dtype=np.float64)


def similarity_handedness(H: np.ndarray) -> int:
    H = np.asarray(H, dtype=np.float64)
    H = H / H[2, 2]
    return -1 if float(np.linalg.det(H[:2, :2])) < 0 else 1


def params_from_similarity_h(H: np.ndarray, handedness: int) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    H = H / H[2, 2]
    if handedness < 0:
        a = 0.5 * (H[0, 0] - H[1, 1])
        b = 0.5 * (H[0, 1] + H[1, 0])
    else:
        a = 0.5 * (H[0, 0] + H[1, 1])
        b = 0.5 * (H[1, 0] - H[0, 1])
    return np.asarray([a, b, H[0, 2], H[1, 2]], dtype=np.float64)


def similarity_h(params: np.ndarray, handedness: int) -> np.ndarray:
    a, b, tx, ty = [float(v) for v in params]
    if handedness < 0:
        return np.asarray([[a, b, tx], [b, -a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_frame(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def player_foot_pixel(mask: np.ndarray) -> np.ndarray | None:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    if len(xs) < 8:
        return None
    cutoff = np.quantile(ys, 0.9)
    keep = ys >= cutoff
    if keep.sum() < 3:
        keep = np.ones_like(ys, dtype=bool)
    return np.asarray([float(np.median(xs[keep])), float(np.median(ys[keep]))], dtype=np.float64)


def collect_player_foot_uv(
    frame_indices: np.ndarray,
    frames_dir: Path,
    player_mask_dir: Path,
    sam3_json: Path,
    world_points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> dict[int, list[dict]]:
    sam3 = json.loads(sam3_json.read_text())
    by_frame = {int(frame["frame"]): frame for frame in sam3.get("frames", [])}
    output: dict[int, list[dict]] = {}
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        frame = load_frame(frames_dir / f"frame_{frame_idx:05d}.jpg")
        if frame is None:
            continue
        shape = frame.shape[:2]
        record = by_frame.get(int(frame_idx), {})
        for object_id in [int(value) for value in record.get("object_ids", [])]:
            mask = load_npz_mask(player_mask_dir, int(frame_idx), object_id, shape)
            pixel = player_foot_pixel(mask)
            if pixel is None:
                continue
            uv = pixel_to_plane_uv(
                pixel,
                shape,
                world_points.shape[1],
                intrinsic[local_idx],
                extrinsic[local_idx],
                center,
                basis_u,
                basis_v,
            )
            if uv is None or not np.isfinite(uv).all():
                continue
            output.setdefault(object_id, []).append(
                {
                    "frame": int(frame_idx),
                    "object_id": int(object_id),
                    "pixel": pixel.tolist(),
                    "plane_uv": uv.tolist(),
                }
            )
    return output


def choose_goalie_track(
    player_uv_by_object: dict[int, list[dict]],
    H_plane_to_floor: np.ndarray,
    min_frames: int,
) -> tuple[int, np.ndarray, np.ndarray, dict]:
    candidates = []
    for object_id, records in player_uv_by_object.items():
        if len(records) < min_frames:
            continue
        uv = np.asarray([item["plane_uv"] for item in records], dtype=np.float64)
        floor = transform_plane_to_floor(H_plane_to_floor, uv)
        valid = np.isfinite(floor).all(axis=1)
        if valid.sum() < min_frames:
            continue
        floor = floor[valid]
        distances = np.linalg.norm(floor[:, None, :] - GOAL_CENTERS[None, :, :], axis=2)
        nearest_goal_idx = int(np.argmin(np.median(distances, axis=0)))
        goal_dist = distances[:, nearest_goal_idx]
        candidates.append(
            {
                "object_id": int(object_id),
                "records": [records[i] for i in np.nonzero(valid)[0]],
                "uv": uv[valid],
                "floor": floor,
                "goal_index": nearest_goal_idx,
                "goal_center": GOAL_CENTERS[nearest_goal_idx],
                "mean_goal_distance_ft": float(np.mean(goal_dist)),
                "median_goal_distance_ft": float(np.median(goal_dist)),
                "min_goal_distance_ft": float(np.min(goal_dist)),
                "frames": int(valid.sum()),
                "mean_floor_xy_ft": floor.mean(axis=0),
            }
        )
    if not candidates:
        raise RuntimeError(f"No player track had at least {min_frames} usable foot points.")
    candidates.sort(key=lambda item: (item["median_goal_distance_ft"], -item["frames"]))
    chosen = candidates[0]
    meta = {
        "object_id": chosen["object_id"],
        "frames": chosen["frames"],
        "goal_index": chosen["goal_index"],
        "goal_center": chosen["goal_center"].tolist(),
        "mean_goal_distance_ft": chosen["mean_goal_distance_ft"],
        "median_goal_distance_ft": chosen["median_goal_distance_ft"],
        "min_goal_distance_ft": chosen["min_goal_distance_ft"],
        "mean_floor_xy_ft_before": chosen["mean_floor_xy_ft"].tolist(),
        "ranked_candidates": [
            {
                "object_id": item["object_id"],
                "frames": item["frames"],
                "goal_center": item["goal_center"].tolist(),
                "median_goal_distance_ft": item["median_goal_distance_ft"],
                "mean_floor_xy_ft": item["mean_floor_xy_ft"].tolist(),
            }
            for item in candidates[:8]
        ],
    }
    return chosen["object_id"], chosen["uv"], chosen["goal_center"], meta


def fit_similarity_with_goalie(
    old_H: np.ndarray,
    faceoff_uv: np.ndarray,
    faceoff_targets: np.ndarray,
    goalie_uv: np.ndarray,
    goalie_goal: np.ndarray,
    goalie_radius_ft: float,
    faceoff_weight: float,
    goalie_weight: float,
    prior_weight: float,
) -> tuple[np.ndarray, dict]:
    handedness = similarity_handedness(old_H)
    p0 = params_from_similarity_h(old_H, handedness)
    dense_floor = dense_floor_model(step_ft=4.0).astype(np.float64)
    old_floor_to_plane = np.linalg.inv(old_H)
    prior_uv = apply_homography(old_floor_to_plane, dense_floor)
    valid_prior = np.isfinite(prior_uv).all(axis=1)
    prior_uv = prior_uv[valid_prior]
    prior_old = transform_plane_to_floor(old_H, prior_uv)

    def residuals(params: np.ndarray) -> np.ndarray:
        H = similarity_h(params, handedness)
        values: list[float] = []
        if len(faceoff_uv):
            pred = transform_plane_to_floor(H, faceoff_uv)
            valid = np.isfinite(pred).all(axis=1)
            if valid.any():
                values.extend(((pred[valid] - faceoff_targets[valid]) * faceoff_weight).reshape(-1).tolist())
        if len(goalie_uv):
            pred = transform_plane_to_floor(H, goalie_uv)
            valid = np.isfinite(pred).all(axis=1)
            if valid.any():
                vec = pred[valid] - goalie_goal[None, :]
                dist = np.linalg.norm(vec, axis=1)
                unit = vec / np.maximum(dist[:, None], 1e-6)
                outside = np.maximum(0.0, dist - goalie_radius_ft)
                values.extend((unit * outside[:, None] * goalie_weight).reshape(-1).tolist())
        if prior_weight > 0:
            pred_prior = transform_plane_to_floor(H, prior_uv)
            valid = np.isfinite(pred_prior).all(axis=1) & np.isfinite(prior_old).all(axis=1)
            if valid.any():
                values.extend(((pred_prior[valid] - prior_old[valid]) * prior_weight).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residuals,
        p0,
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=600,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    H = similarity_h(result.x, handedness)
    old_faceoff_error = np.linalg.norm(transform_plane_to_floor(old_H, faceoff_uv) - faceoff_targets, axis=1) if len(faceoff_uv) else np.zeros(0)
    new_faceoff_error = np.linalg.norm(transform_plane_to_floor(H, faceoff_uv) - faceoff_targets, axis=1) if len(faceoff_uv) else np.zeros(0)
    old_goalie = transform_plane_to_floor(old_H, goalie_uv) if len(goalie_uv) else np.zeros((0, 2))
    new_goalie = transform_plane_to_floor(H, goalie_uv) if len(goalie_uv) else np.zeros((0, 2))
    old_goalie_dist = np.linalg.norm(old_goalie - goalie_goal[None, :], axis=1) if len(goalie_uv) else np.zeros(0)
    new_goalie_dist = np.linalg.norm(new_goalie - goalie_goal[None, :], axis=1) if len(goalie_uv) else np.zeros(0)
    prior_drift = np.linalg.norm(transform_plane_to_floor(H, prior_uv) - prior_old, axis=1)
    metrics = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "transform_model": "goalie_anchor_similarity",
        "transform_matrix_type": "affine_3x3_similarity",
        "similarity_handedness": int(handedness),
        "faceoff_constraints": int(len(faceoff_uv)),
        "goalie_constraints": int(len(goalie_uv)),
        "old_mean_faceoff_error_ft": float(np.mean(old_faceoff_error)) if len(old_faceoff_error) else None,
        "new_mean_faceoff_error_ft": float(np.mean(new_faceoff_error)) if len(new_faceoff_error) else None,
        "old_max_faceoff_error_ft": float(np.max(old_faceoff_error)) if len(old_faceoff_error) else None,
        "new_max_faceoff_error_ft": float(np.max(new_faceoff_error)) if len(new_faceoff_error) else None,
        "old_mean_goalie_goal_distance_ft": float(np.mean(old_goalie_dist)) if len(old_goalie_dist) else None,
        "new_mean_goalie_goal_distance_ft": float(np.mean(new_goalie_dist)) if len(new_goalie_dist) else None,
        "old_median_goalie_goal_distance_ft": float(np.median(old_goalie_dist)) if len(old_goalie_dist) else None,
        "new_median_goalie_goal_distance_ft": float(np.median(new_goalie_dist)) if len(new_goalie_dist) else None,
        "old_max_goalie_goal_distance_ft": float(np.max(old_goalie_dist)) if len(old_goalie_dist) else None,
        "new_max_goalie_goal_distance_ft": float(np.max(new_goalie_dist)) if len(new_goalie_dist) else None,
        "goalie_radius_ft": float(goalie_radius_ft),
        "faceoff_weight": float(faceoff_weight),
        "goalie_weight": float(goalie_weight),
        "prior_weight": float(prior_weight),
        "mean_prior_drift_ft": float(np.mean(prior_drift)),
        "max_prior_drift_ft": float(np.max(prior_drift)),
    }
    return H, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit rigid VGGT field alignment with a soft goalkeeper-near-goal anchor.")
    parser.add_argument("--vggt-npz", default="outputs/vggt/birds_eye_undistorted/vggt_predictions_compact.npz")
    parser.add_argument("--base-fit-json", default="outputs/vggt/birds_eye_undistorted/birds_eye_player_locations_vggt_nll_crease_two_point_similarity.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps_undistorted")
    parser.add_argument("--faceoff-mask-dir", default="outputs/sam2/floor_features/sam2_faceoff_corner_instance_masks_undistorted")
    parser.add_argument("--player-mask-dir", default="outputs/sam3/text/sam3_text_player_instance_masks_undistorted")
    parser.add_argument("--sam3-json", default="outputs/sam3/team_classification/sam3_team_transreid_3clusters_detections.json")
    parser.add_argument("--output", default="outputs/vggt/birds_eye_undistorted/birds_eye_player_locations_vggt_nll_crease_goalie_anchor_similarity.json")
    parser.add_argument("--min-goalie-frames", type=int, default=8)
    parser.add_argument("--goalie-radius-ft", type=float, default=12.0)
    parser.add_argument("--faceoff-weight", type=float, default=1.0)
    parser.add_argument("--goalie-weight", type=float, default=0.35)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument("--min-depth-conf", type=float, default=3.0)
    parser.add_argument("--max-faceoff-points", type=int, default=160)
    parser.add_argument("--min-faceoff-area", type=int, default=8)
    parser.add_argument("--max-faceoff-area", type=int, default=10000)
    args = parser.parse_args()

    base = json.loads(Path(args.base_fit_json).read_text())
    plane = base["floor_plane"]
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    old_H = np.asarray(base["plane_to_floor_homography"], dtype=np.float64)

    with np.load(args.vggt_npz) as data:
        frame_indices = data["frame_indices"].astype(int) if "frame_indices" in data else np.arange(data["depth_map"].shape[0])
        extrinsic = data["extrinsic"].astype(np.float32)
        intrinsic = data["intrinsic"].astype(np.float32)
        depth = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    faceoff_uv, faceoff_targets, faceoff_metrics = collect_faceoff_points(
        frame_indices,
        Path(args.frames_dir),
        Path(args.faceoff_mask_dir),
        world_points,
        depth_conf,
        intrinsic,
        extrinsic,
        center,
        basis_u,
        basis_v,
        args.min_depth_conf,
        args.max_faceoff_points,
        args.min_faceoff_area,
        args.max_faceoff_area,
    )
    player_uv = collect_player_foot_uv(
        frame_indices,
        Path(args.frames_dir),
        Path(args.player_mask_dir),
        Path(args.sam3_json),
        world_points,
        intrinsic,
        extrinsic,
        center,
        basis_u,
        basis_v,
    )
    goalie_object_id, goalie_uv, goalie_goal, goalie_meta = choose_goalie_track(player_uv, old_H, args.min_goalie_frames)
    H, metrics = fit_similarity_with_goalie(
        old_H,
        faceoff_uv,
        faceoff_targets,
        goalie_uv,
        goalie_goal,
        args.goalie_radius_ft,
        args.faceoff_weight,
        args.goalie_weight,
        args.prior_weight,
    )

    output = dict(base)
    output["plane_to_floor_homography"] = H.tolist()
    output["alignment_metrics"] = dict(base.get("alignment_metrics", {}))
    output["alignment_metrics"].update(metrics)
    output["goalie_anchor_refinement"] = {
        "schema": "vggt_goalie_anchor_similarity_v1",
        "source_base_fit_json": args.base_fit_json,
        "faceoff_mask_dir": args.faceoff_mask_dir,
        "player_mask_dir": args.player_mask_dir,
        "sam3_json": args.sam3_json,
        "goalie_object_id": int(goalie_object_id),
        "goalie_meta": goalie_meta,
        "metrics": metrics,
        "faceoff_point_metrics": faceoff_metrics,
    }
    output["field_geometry_model"] = dict(output.get("field_geometry_model", {}))
    output["field_geometry_model"]["transform_constraint"] = "goalie_anchor_similarity"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(out_path), "goalie_object_id": int(goalie_object_id), **metrics}, indent=2))


if __name__ == "__main__":
    main()
