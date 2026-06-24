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
from nll_field_geometry import CENTER_X_FT, CENTER_Y_FT, LEFT_GOAL_X_FT, RIGHT_GOAL_X_FT  # noqa: E402
from render_birds_eye_locations import dense_floor_model  # noqa: E402
from run_vggt_birds_eye import original_pixels_to_vggt, project_to_plane_coords, transform_plane_to_floor  # noqa: E402


GOAL_CENTERS = {
    "left_goal": np.asarray([LEFT_GOAL_X_FT, CENTER_Y_FT], dtype=np.float64),
    "right_goal": np.asarray([RIGHT_GOAL_X_FT, CENTER_Y_FT], dtype=np.float64),
}
CENTER_SPOT = np.asarray([CENTER_X_FT, CENTER_Y_FT], dtype=np.float64)


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    out = homog @ H.T
    denom = np.where(np.abs(out[:, 2]) < 1e-9, np.nan, out[:, 2])
    return out[:, :2] / denom[:, None]


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


def load_mask(mask_dir: Path, frame_idx: int, object_id: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return np.zeros((height, width), dtype=bool)
    data = np.load(path)
    ids = data["object_ids"].astype(int).tolist()
    masks = data["masks"].astype(bool)
    if object_id not in ids:
        return np.zeros((height, width), dtype=bool)
    mask = masks[ids.index(object_id)]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def player_bottom_pixels(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    ys, xs = np.where(mask)
    if len(xs) < 8:
        return np.zeros((0, 2), dtype=np.float32)
    cutoff = np.quantile(ys, 0.88)
    keep = ys >= cutoff
    points = np.column_stack([xs[keep], ys[keep]]).astype(np.float32)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def pixels_to_plane_uv(
    pixels_xy: np.ndarray,
    image_shape: tuple[int, int],
    world_points_frame: np.ndarray,
    depth_conf_frame: np.ndarray,
    min_conf: float,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    resolution = world_points_frame.shape[0]
    mapped = original_pixels_to_vggt(pixels_xy, image_shape, resolution)
    xs = np.clip(np.round(mapped[:, 0]).astype(np.int32), 0, resolution - 1)
    ys = np.clip(np.round(mapped[:, 1]).astype(np.int32), 0, resolution - 1)
    points = world_points_frame[ys, xs].astype(np.float64)
    conf = depth_conf_frame[ys, xs].astype(np.float64)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)
    if not valid.any():
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    uv = project_to_plane_coords(points[valid], center, basis_u, basis_v)
    return uv, conf[valid]


def collect_player_uv(
    frame_indices: np.ndarray,
    frames_dir: Path,
    player_mask_dir: Path,
    sam3_json: Path,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    min_conf: float,
    max_points: int,
) -> dict[int, list[dict]]:
    sam3 = json.loads(sam3_json.read_text())
    by_frame = {int(frame["frame"]): frame for frame in sam3.get("frames", [])}
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    output: dict[int, list[dict]] = {}
    for local_idx, frame_idx in enumerate(frame_indices.tolist()):
        if frame_idx >= len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_idx]), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        shape = frame.shape[:2]
        record = by_frame.get(int(frame_idx), {})
        for object_id in [int(value) for value in record.get("object_ids", [])]:
            mask = load_mask(player_mask_dir, int(frame_idx), object_id, shape)
            pixels = player_bottom_pixels(mask, max_points, seed=frame_idx * 1009 + object_id)
            uv, conf = pixels_to_plane_uv(
                pixels,
                shape,
                world_points[local_idx],
                depth_conf[local_idx],
                min_conf,
                center,
                basis_u,
                basis_v,
            )
            if len(uv) == 0:
                continue
            median_uv = np.median(uv, axis=0)
            if not np.isfinite(median_uv).all():
                continue
            output.setdefault(object_id, []).append(
                {
                    "frame": int(frame_idx),
                    "object_id": int(object_id),
                    "plane_uv": median_uv.tolist(),
                    "samples": int(len(uv)),
                    "median_conf": float(np.median(conf)),
                }
            )
    return output


def choose_huddle_anchor(player_uv: dict[int, list[dict]], H: np.ndarray, search_frames: int, max_pair_sep_ft: float) -> tuple[np.ndarray, dict]:
    by_frame: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for object_id, records in player_uv.items():
        for record in records:
            frame = int(record["frame"])
            if frame >= search_frames:
                continue
            uv = np.asarray(record["plane_uv"], dtype=np.float64)
            floor = transform_plane_to_floor(H, uv.reshape(1, 2))[0]
            if np.isfinite(floor).all():
                by_frame.setdefault(frame, []).append((int(object_id), uv, floor))
    candidates = []
    for frame, items in by_frame.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                id1, uv1, floor1 = items[i]
                id2, uv2, floor2 = items[j]
                sep = float(np.linalg.norm(floor1 - floor2))
                midpoint_floor = 0.5 * (floor1 + floor2)
                center_dist = float(np.linalg.norm(midpoint_floor - CENTER_SPOT))
                if sep <= max_pair_sep_ft:
                    candidates.append((center_dist + 0.15 * sep + 0.05 * frame, frame, id1, id2, uv1, uv2, sep, midpoint_floor))
    if not candidates:
        raise RuntimeError("Could not identify an early close huddle pair for the center anchor.")
    candidates.sort(key=lambda item: item[0])
    _, frame, id1, id2, uv1, uv2, sep, midpoint_floor = candidates[0]
    huddle_uv = 0.5 * (uv1 + uv2)
    meta = {
        "frame": int(frame),
        "object_ids": [int(id1), int(id2)],
        "pair_separation_ft_before": float(sep),
        "midpoint_floor_xy_before": midpoint_floor.tolist(),
        "center_distance_ft_before": float(np.linalg.norm(midpoint_floor - CENTER_SPOT)),
    }
    return huddle_uv.reshape(1, 2), meta


def choose_goalie_anchors(
    player_uv: dict[int, list[dict]],
    H: np.ndarray,
    min_frames: int,
    explicit_goalie_ids: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict]:
    candidates_by_goal: dict[str, list[dict]] = {name: [] for name in GOAL_CENTERS}
    for object_id, records in player_uv.items():
        if len(records) < min_frames:
            continue
        uv = np.asarray([record["plane_uv"] for record in records], dtype=np.float64)
        floor = transform_plane_to_floor(H, uv)
        valid = np.isfinite(floor).all(axis=1)
        if valid.sum() < min_frames:
            continue
        uv = uv[valid]
        floor = floor[valid]
        for goal_name, goal_xy in GOAL_CENTERS.items():
            dist = np.linalg.norm(floor - goal_xy[None, :], axis=1)
            candidates_by_goal[goal_name].append(
                {
                    "object_id": int(object_id),
                    "uv": uv,
                    "frames": int(len(uv)),
                    "median_goal_distance_ft": float(np.median(dist)),
                    "mean_goal_distance_ft": float(np.mean(dist)),
                    "mean_floor_xy_before": floor.mean(axis=0).tolist(),
                }
            )
    anchors = {}
    meta = {}
    for goal_name, candidates in candidates_by_goal.items():
        candidates.sort(key=lambda item: (item["median_goal_distance_ft"], -item["frames"]))
        if goal_name in explicit_goalie_ids:
            chosen_candidates = [item for item in candidates if item["object_id"] == explicit_goalie_ids[goal_name]]
            if not chosen_candidates:
                raise RuntimeError(f"Explicit {goal_name} goalie ID {explicit_goalie_ids[goal_name]} was not a usable candidate.")
            candidates = chosen_candidates + [item for item in candidates if item["object_id"] != explicit_goalie_ids[goal_name]]
        if not candidates:
            continue
        chosen = candidates[0]
        anchors[goal_name] = chosen["uv"]
        meta[goal_name] = {
            key: value
            for key, value in chosen.items()
            if key != "uv"
        }
        meta[goal_name]["ranked_candidates"] = [
            {key: value for key, value in item.items() if key != "uv"}
            for item in candidates[:8]
        ]
    return anchors, meta


def refit_similarity(
    old_H: np.ndarray,
    huddle_uv: np.ndarray,
    goalie_uv_by_goal: dict[str, np.ndarray],
    huddle_weight: float,
    goalie_weight: float,
    goalie_radius_ft: float,
    prior_weight: float,
) -> tuple[np.ndarray, dict]:
    handedness = similarity_handedness(old_H)
    p0 = params_from_similarity_h(old_H, handedness)
    dense_floor = dense_floor_model(step_ft=4.0).astype(np.float64)
    prior_uv = apply_homography(np.linalg.inv(old_H), dense_floor)
    prior_old = transform_plane_to_floor(old_H, prior_uv)

    def residuals(params: np.ndarray) -> np.ndarray:
        H = similarity_h(params, handedness)
        values: list[float] = []
        if len(huddle_uv):
            pred = transform_plane_to_floor(H, huddle_uv)
            valid = np.isfinite(pred).all(axis=1)
            if valid.any():
                values.extend(((pred[valid] - CENTER_SPOT[None, :]) * huddle_weight).reshape(-1).tolist())
        for goal_name, uv in goalie_uv_by_goal.items():
            if len(uv) == 0:
                continue
            pred = transform_plane_to_floor(H, uv)
            valid = np.isfinite(pred).all(axis=1)
            if not valid.any():
                continue
            goal = GOAL_CENTERS[goal_name]
            vec = pred[valid] - goal[None, :]
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
        max_nfev=800,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    H = similarity_h(result.x, handedness)
    metrics: dict[str, object] = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "transform_model": "huddle_center_and_goalie_goal_similarity",
        "similarity_handedness": int(handedness),
        "huddle_constraints": int(len(huddle_uv)),
        "huddle_weight": float(huddle_weight),
        "goalie_weight": float(goalie_weight),
        "goalie_radius_ft": float(goalie_radius_ft),
        "prior_weight": float(prior_weight),
    }
    if len(huddle_uv):
        old_huddle = transform_plane_to_floor(old_H, huddle_uv)
        new_huddle = transform_plane_to_floor(H, huddle_uv)
        metrics["old_huddle_center_distance_ft"] = float(np.linalg.norm(old_huddle[0] - CENTER_SPOT))
        metrics["new_huddle_center_distance_ft"] = float(np.linalg.norm(new_huddle[0] - CENTER_SPOT))
    for goal_name, uv in goalie_uv_by_goal.items():
        old_goalie = transform_plane_to_floor(old_H, uv)
        new_goalie = transform_plane_to_floor(H, uv)
        goal = GOAL_CENTERS[goal_name]
        old_dist = np.linalg.norm(old_goalie - goal[None, :], axis=1)
        new_dist = np.linalg.norm(new_goalie - goal[None, :], axis=1)
        metrics[f"{goal_name}_constraints"] = int(len(uv))
        metrics[f"old_{goal_name}_median_distance_ft"] = float(np.median(old_dist)) if len(old_dist) else None
        metrics[f"new_{goal_name}_median_distance_ft"] = float(np.median(new_dist)) if len(new_dist) else None
        metrics[f"old_{goal_name}_mean_distance_ft"] = float(np.mean(old_dist)) if len(old_dist) else None
        metrics[f"new_{goal_name}_mean_distance_ft"] = float(np.mean(new_dist)) if len(new_dist) else None
    prior_drift = np.linalg.norm(transform_plane_to_floor(H, prior_uv) - prior_old, axis=1)
    metrics["mean_prior_drift_ft"] = float(np.mean(prior_drift))
    metrics["max_prior_drift_ft"] = float(np.max(prior_drift))
    return H, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit a VGGT field similarity using early huddle center and goalie goal anchors.")
    parser.add_argument("--vggt-npz", required=True)
    parser.add_argument("--base-fit-json", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--player-mask-dir", required=True)
    parser.add_argument("--sam3-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-depth-conf", type=float, default=1.0)
    parser.add_argument("--max-player-points", type=int, default=80)
    parser.add_argument("--huddle-search-frames", type=int, default=24)
    parser.add_argument("--huddle-max-pair-sep-ft", type=float, default=22.0)
    parser.add_argument("--huddle-weight", type=float, default=1.0)
    parser.add_argument("--min-goalie-frames", type=int, default=4)
    parser.add_argument("--left-goalie-object-id", type=int, default=-1)
    parser.add_argument("--right-goalie-object-id", type=int, default=-1)
    parser.add_argument("--goalie-radius-ft", type=float, default=2.5)
    parser.add_argument("--goalie-weight", type=float, default=0.75)
    parser.add_argument("--prior-weight", type=float, default=0.018)
    args = parser.parse_args()

    base = json.loads(Path(args.base_fit_json).read_text())
    plane = base["floor_plane"]
    center = np.asarray(plane["center"], dtype=np.float64)
    basis_u = np.asarray(plane["basis_u"], dtype=np.float64)
    basis_v = np.asarray(plane["basis_v"], dtype=np.float64)
    old_H = np.asarray(base["plane_to_floor_homography"], dtype=np.float64)

    with np.load(args.vggt_npz) as data:
        frame_indices = data["frame_indices"].astype(int)
        depth = data["depth_map"].astype(np.float32)
        depth_conf = data["depth_conf"].astype(np.float32)
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]
        extrinsic = data["extrinsic"].astype(np.float32)
        intrinsic = data["intrinsic"].astype(np.float32)
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    player_uv = collect_player_uv(
        frame_indices,
        Path(args.frames_dir),
        Path(args.player_mask_dir),
        Path(args.sam3_json),
        world_points,
        depth_conf,
        center,
        basis_u,
        basis_v,
        args.min_depth_conf,
        args.max_player_points,
    )
    huddle_uv, huddle_meta = choose_huddle_anchor(player_uv, old_H, args.huddle_search_frames, args.huddle_max_pair_sep_ft)
    explicit_goalies = {}
    if args.left_goalie_object_id >= 0:
        explicit_goalies["left_goal"] = int(args.left_goalie_object_id)
    if args.right_goalie_object_id >= 0:
        explicit_goalies["right_goal"] = int(args.right_goalie_object_id)
    goalie_uv_by_goal, goalie_meta = choose_goalie_anchors(player_uv, old_H, args.min_goalie_frames, explicit_goalies)
    H, metrics = refit_similarity(
        old_H,
        huddle_uv,
        goalie_uv_by_goal,
        args.huddle_weight,
        args.goalie_weight,
        args.goalie_radius_ft,
        args.prior_weight,
    )

    output = dict(base)
    output["plane_to_floor_homography"] = H.tolist()
    output["alignment_metrics"] = dict(base.get("alignment_metrics", {}))
    output["alignment_metrics"].update(metrics)
    output["player_anchor_refinement"] = {
        "schema": "vggt_huddle_goalie_anchor_similarity_v1",
        "source_base_fit_json": args.base_fit_json,
        "player_mask_dir": args.player_mask_dir,
        "sam3_json": args.sam3_json,
        "huddle_anchor": huddle_meta,
        "goalie_anchors": goalie_meta,
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "huddle_anchor": huddle_meta, "goalie_anchors": goalie_meta, **metrics}, indent=2))


if __name__ == "__main__":
    main()
