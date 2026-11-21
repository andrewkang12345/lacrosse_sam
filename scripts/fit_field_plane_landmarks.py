"""Fit the NLL field to a VGGT-Omega reconstruction through its floor plane.

Pipeline (see docs/field_plane_fit.md for the reasoning):

1. Floor plane: RANSAC + least squares on confident VGGT-Omega depth points that
   land on green turf (players and boards are excluded by colour).
2. Landmark lifting: each landmark mask pixel is lifted by intersecting its
   camera ray with that single plane. Per-pixel depth is not used, so depth
   noise and per-frame depth-scale drift do not smear the landmarks.
3. Planar field fit: a proper similarity (scale, rotation, translation, no
   mirror) maps plane coordinates to field feet. It is initialised by a global
   distance-transform search and refined with a robust loss in plane units.
4. Camera refinement: VGGT-Omega's per-frame cameras are re-expressed in field
   feet, then each frame gets a small pan/tilt/roll/zoom correction so the
   landmarks reproject onto the known planar field. Corrections are
   regularised towards VGGT-Omega and smoothed between neighbouring frames.

Outputs (in --output-dir):
  field_plane_fit.json            plane, similarity, per-frame field cameras, metrics
  field_homographies.json         per-frame image->field homographies
                                  (schema floor_homography_matrix_v1, usable by
                                  render_birds_eye_locations.py and
                                  render_camera_homography_overlay.py)
  field_reprojection_h264.mp4     camera frames with the fitted field drawn on top
  landmarks_on_field.jpg          lifted landmark pixels drawn on the field diagram
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import field_model as fm
from field_model import FieldDims
from video_utils import write_h264_video

# ---------------------------------------------------------------------------
# Field model (feet; x along the length, y across the 85 ft width, z up).
# Geometry lives in field_model.py; this section only maps landmark names to
# how their masks become observations.
# ---------------------------------------------------------------------------

# How each landmark's mask is turned into observation pixels.
#   pixels       every mask pixel lies on the landmark (painted lines, board-base pixels)
#   boundary     the filled mask's outline lies on the landmark (filled crease)
#   bottom_edge  lowest mask pixel per column, where the boards meet the floor
#   top_edge     highest floor-mask pixel per column (far boards / floor junction)
#   centroid     one point per frame (faceoff dots)
DEFAULT_MODE = {
    **{name: "pixels" for name in fm.LINE_NAMES},
    "white_lines": "pixels",
    "left_crease_lines": "pixels",
    "right_crease_lines": "pixels",
    "left_goal_line": "pixels",
    "right_goal_line": "pixels",
    "board_base": "pixels",
    "goal_crease": "boundary",
    "left_goal_crease": "boundary",
    "right_goal_crease": "boundary",
    "field_outline": "bottom_edge",
    "far_boards_edge": "top_edge",
    **{name: "centroid" for name in fm.DOT_NAMES},
}
CLASS_WEIGHT = {"centroid": 2.0, "pixels": 1.0, "boundary": 0.7, "bottom_edge": 0.6, "top_edge": 0.6}

DRAW_COLORS_BGR = {
    "lines": (255, 255, 255),
    "crease": (60, 60, 255),
    "outline": (0, 220, 255),
    "dots": (255, 200, 0),
}


# ---------------------------------------------------------------------------
# Cameras and plane geometry
# ---------------------------------------------------------------------------


@dataclass
class Cameras:
    """Per-frame pinhole cameras in original-image pixels. R maps world->camera."""

    K: np.ndarray  # (S,3,3)
    R: np.ndarray  # (S,3,3)
    C: np.ndarray  # (S,3) camera centres in world coordinates

    def rays(self, i: int, pixels: np.ndarray) -> np.ndarray:
        K = self.K[i]
        d = np.column_stack([(pixels[:, 0] - K[0, 2]) / K[0, 0], (pixels[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(pixels))])
        return d @ self.R[i]  # rows: R^T d

    def project(self, i: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = (points - self.C[i]) @ self.R[i].T
        z = cam[:, 2]
        px = cam @ self.K[i].T
        with np.errstate(divide="ignore", invalid="ignore"):
            px = px[:, :2] / px[:, 2:3]
        return px, z


def load_predictions(path: Path) -> dict:
    with np.load(path) as data:
        pred = {key: data[key] for key in data.files}
    model_h, model_w = (int(v) for v in pred["model_hw"])
    orig_h, orig_w = (int(v) for v in pred["original_hw"])
    K = pred["intrinsic"].astype(np.float64).copy()
    K[:, 0, :] *= orig_w / model_w
    K[:, 1, :] *= orig_h / model_h
    E = pred["extrinsic"].astype(np.float64)
    R = E[:, :, :3]
    C = -np.einsum("sji,sj->si", R, E[:, :, 3])
    pred["cameras"] = Cameras(K=K, R=R.copy(), C=C)
    pred["scale_xy"] = (model_w / orig_w, model_h / orig_h)
    return pred


def green_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 35)


def floor_depth_points(pred: dict, max_points_per_frame: int, conf_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    """World points from VGGT-Omega depth on confident green-turf pixels."""
    cams: Cameras = pred["cameras"]
    depth, conf, colors = pred["depth"], pred["depth_conf"], pred["colors"]
    sx, sy = pred["scale_xy"]
    rng = np.random.default_rng(0)
    points, frame_ids = [], []
    for i in range(depth.shape[0]):
        mask = green_mask(colors[i]) & (conf[i] >= np.percentile(conf[i], conf_percentile)) & (depth[i] > 0)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_frame:
            keep = rng.choice(len(xs), max_points_per_frame, replace=False)
            ys, xs = ys[keep], xs[keep]
        px = np.column_stack([(xs + 0.5) / sx, (ys + 0.5) / sy])
        world = cams.C[i] + cams.rays(i, px) * depth[i][ys, xs][:, None]
        points.append(world)
        frame_ids.append(np.full(len(world), i))
    return np.concatenate(points), np.concatenate(frame_ids)


@dataclass
class Plane:
    center: np.ndarray
    normal: np.ndarray  # points towards the cameras ("up")
    e1: np.ndarray
    e2: np.ndarray  # e1 x e2 = normal, so (e1, e2, normal) is right-handed

    def to_uv(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.center
        return np.column_stack([rel @ self.e1, rel @ self.e2])

    def from_uv(self, uv: np.ndarray, height: np.ndarray | float = 0.0) -> np.ndarray:
        return self.center + uv[:, :1] * self.e1 + uv[:, 1:2] * self.e2 + np.asarray(height).reshape(-1, 1) * self.normal


def fit_floor_plane(points: np.ndarray, camera_centers: np.ndarray, iterations: int = 600) -> tuple[Plane, dict]:
    rng = np.random.default_rng(1)
    scene_scale = float(np.median(np.linalg.norm(points - np.median(camera_centers, axis=0), axis=1)))
    threshold = 0.01 * scene_scale
    best_count, best_inliers = -1, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        inliers = np.abs((points - sample[0]) @ (normal / norm)) < threshold
        if inliers.sum() > best_count:
            best_count, best_inliers = int(inliers.sum()), inliers
    for _ in range(3):  # least-squares polish on the consensus set
        inlier_pts = points[best_inliers]
        center = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - center, full_matrices=False)
        normal = vh[-1]
        best_inliers = np.abs((points - center) @ normal) < threshold
    if float(np.median((camera_centers - center) @ normal)) < 0:
        normal = -normal
    e1 = vh[0] - (vh[0] @ normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    residual = (points[best_inliers] - center) @ normal
    plane = Plane(center=center, normal=normal, e1=e1, e2=e2)
    heights = (camera_centers - center) @ normal
    stats = {
        "points": int(len(points)),
        "inlier_fraction": float(best_inliers.mean()),
        "inlier_rms_over_scene_scale": float(np.sqrt(np.mean(residual**2)) / scene_scale),
        "camera_height_median_units": float(np.median(heights)),
        "camera_height_range_units": [float(heights.min()), float(heights.max())],
    }
    return plane, stats


def ray_plane_uv(cams: Cameras, i: int, pixels: np.ndarray, plane: Plane) -> np.ndarray:
    rays = cams.rays(i, pixels)
    denom = rays @ plane.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = ((plane.center - cams.C[i]) @ plane.normal) / denom
    world = cams.C[i] + lam[:, None] * rays
    uv = plane.to_uv(world)
    uv[~(lam > 0)] = np.nan
    return uv


# ---------------------------------------------------------------------------
# Landmark observations
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    name: str
    mode: str
    frame: int  # local frame index into the predictions
    pixels: np.ndarray  # (N,2) original-image pixels


def parse_landmark_spec(spec: str) -> tuple[str, Path, int | None]:
    name, _, source = spec.partition("=")
    if name not in DEFAULT_MODE:
        raise ValueError(f"Unknown landmark '{name}'. Known: {sorted(DEFAULT_MODE)}")
    path, _, object_id = source.partition("#")
    return name, Path(path), int(object_id) if object_id else None


def load_mask(mask_dir: Path, frame_number: int, object_id: int | None, shape: tuple[int, int]) -> np.ndarray | None:
    path = mask_dir / f"{frame_number:08d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        ids = [int(v) for v in data["object_ids"].tolist()]
        masks = data["masks"]
        if object_id is None:
            mask = masks.astype(bool).any(axis=0) if len(ids) else None
        elif object_id in ids:
            mask = masks[ids.index(object_id)].astype(bool)
        else:
            mask = None
    if mask is None or not mask.any():
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def mask_to_pixels(mask: np.ndarray, mode: str, frame_bgr: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    if mode == "centroid":
        ys, xs = np.nonzero(mask)
        return np.asarray([[np.median(xs), np.median(ys)]], dtype=np.float64)
    if mode == "pixels":
        ys, xs = np.nonzero(mask)
    elif mode == "boundary":
        m8 = mask.astype(np.uint8)
        edge = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))).astype(bool)
        edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = False
        ys, xs = np.nonzero(edge)
    elif mode in ("bottom_edge", "top_edge"):
        turf = green_mask(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cols = np.nonzero(mask.any(axis=0))[0]
        rows = np.arange(h)[:, None]
        if mode == "bottom_edge":
            edge_rows = np.where(mask[:, cols], rows, -1).max(axis=0)
            probe = np.clip(edge_rows + 4, 0, h - 1)
        else:
            edge_rows = np.where(mask[:, cols], rows, h).min(axis=0)
            probe = np.clip(edge_rows + 4, 0, h - 1)
        # keep columns where the edge really is a floor junction: turf just below it,
        # not at the image border (where the mask is merely cut off by the frame).
        keep = (edge_rows > 2) & (edge_rows < h - 3) & turf[probe, cols]
        if mode == "top_edge":
            above = np.clip(edge_rows - 4, 0, h - 1)
            keep &= ~turf[above, cols]
        xs, ys = cols[keep], edge_rows[keep]
    else:
        raise ValueError(mode)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(xs) > max_points:
        keep = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[keep], ys[keep]
    return np.column_stack([xs, ys]).astype(np.float64) + 0.5


def white_paint_pixels(frame_bgr: np.ndarray, exclude: np.ndarray | None, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Thin white paint on the turf, found by colour. Used when SAM3/SAM2 line
    masks miss most of the visible lines; players are removed with their masks."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    turf = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 35)).astype(np.uint8)
    floor = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)) > 0
    white = (hsv[..., 1] <= 60) & (hsv[..., 2] >= 150)
    paint = white & floor & (cv2.dilate(turf, np.ones((7, 7), np.uint8)) > 0)
    thick = cv2.morphologyEx(paint.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    paint &= ~(cv2.dilate(thick, np.ones((9, 9), np.uint8)) > 0)
    if exclude is not None:
        paint &= ~(cv2.dilate(exclude.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    # keep long, thin components (lines, crease arcs); drop text, logos, stick blobs
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paint.astype(np.uint8), connectivity=8)
    length = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (length >= 50.0) & (stats[:, cv2.CC_STAT_AREA] / np.maximum(length, 1.0) <= 10.0)
    keep[0] = False
    paint = keep[labels]
    ys, xs = np.nonzero(paint)
    if len(xs) > max_points:
        keep = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[keep], ys[keep]
    return np.column_stack([xs, ys]).astype(np.float64) + 0.5


def collect_observations(args: argparse.Namespace, frame_names: list[str], shape: tuple[int, int]) -> list[Observation]:
    specs = [parse_landmark_spec(spec) for spec in args.landmark]
    rng = np.random.default_rng(2)
    frames_dir = Path(args.frames_dir)
    all_frames = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    observations = []
    for local_idx, name in enumerate(frame_names):
        # Mask files are numbered by the frame's position in the sorted frame directory.
        frame_number = all_frames.index(name)
        frame_bgr = None
        for landmark, mask_dir, object_id in specs:
            mask = load_mask(mask_dir, frame_number, object_id, shape)
            if mask is None:
                continue
            mode = DEFAULT_MODE[landmark]
            if mode in ("bottom_edge", "top_edge") and frame_bgr is None:
                frame_bgr = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
            pixels = mask_to_pixels(mask, mode, frame_bgr, args.max_points_per_mask, rng)
            if mode == "centroid" and not (args.min_dot_area <= mask.sum() <= args.max_dot_area):
                continue
            if len(pixels):
                observations.append(Observation(landmark, mode, local_idx, pixels))
        if getattr(args, "detect_white_paint", False):
            frame_bgr = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR) if frame_bgr is None else frame_bgr
            players = load_mask(Path(args.player_mask_dir), frame_number, None, shape) if args.player_mask_dir else None
            pixels = white_paint_pixels(frame_bgr, players, args.max_points_per_mask, rng)
            if len(pixels) >= 20:
                observations.append(Observation("white_lines", "pixels", local_idx, pixels))
    return observations


# ---------------------------------------------------------------------------
# Stage 1: planar similarity fit (plane uv -> field feet)
# ---------------------------------------------------------------------------


def similarity_apply(params: np.ndarray, uv: np.ndarray) -> np.ndarray:
    log_k, theta, tx, ty = params
    c, s = np.cos(theta), np.sin(theta)
    k = np.exp(log_k)
    return np.column_stack([k * (c * uv[:, 0] - s * uv[:, 1]) + tx, k * (s * uv[:, 0] + c * uv[:, 1]) + ty])


class FieldModel:
    """Distances (ft) from floor points to each landmark of a parametric field."""

    def __init__(self, dims: FieldDims, names: set[str]):
        self.dims, self.names = dims, set(names)
        # raster distance maps (1 ft grid) for the brute-force initial search
        self.res = 1.0
        self.x0, self.y0 = -40.0, -40.0
        gx = np.arange(self.x0, 240.0 + 1e-6, self.res)
        gy = np.arange(self.y0, dims.width + 40.0 + 1e-6, self.res)
        grid = np.stack(np.meshgrid(gx, gy, indexing="xy"), axis=-1)
        self.shape = grid.shape[:2]
        self.rasters = {name: fm.distance(name, grid, dims) for name in self.names}

    def query(self, name: str, xy: np.ndarray, dims: FieldDims | None = None) -> np.ndarray:
        return fm.distance(name, xy, dims or self.dims)

    def point(self, name: str, dims: FieldDims | None = None) -> np.ndarray:
        return fm.dot_xy(name, dims or self.dims)

    def raster_lookup(self, name: str, xy: np.ndarray, cap: float) -> np.ndarray:
        ix = np.round((xy[..., 0] - self.x0) / self.res).astype(np.int64)
        iy = np.round((xy[..., 1] - self.y0) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.shape[1]) & (iy >= 0) & (iy < self.shape[0])
        out = np.full(ix.shape, cap, dtype=np.float64)
        out[inside] = np.minimum(self.rasters[name][iy[inside], ix[inside]], cap)
        return out


@dataclass
class LiftedSet:
    name: str
    mode: str
    uv: np.ndarray
    weight: float


def lift_observations(observations: list[Observation], cams: Cameras, plane: Plane, max_points_per_class: int) -> list[LiftedSet]:
    grouped: dict[str, list[np.ndarray]] = {}
    modes = {}
    for obs in observations:
        uv = ray_plane_uv(cams, obs.frame, obs.pixels, plane)
        grouped.setdefault(obs.name, []).append(uv[np.isfinite(uv).all(axis=1)])
        modes[obs.name] = obs.mode
    rng = np.random.default_rng(3)
    lifted = []
    for name, parts in grouped.items():
        uv = np.concatenate(parts)
        if len(uv) > max_points_per_class:
            uv = uv[rng.choice(len(uv), max_points_per_class, replace=False)]
        # every class contributes like ~400 points times its mode weight
        weight = CLASS_WEIGHT[modes[name]] * np.sqrt(400.0 / max(len(uv), 1))
        lifted.append(LiftedSet(name, modes[name], uv, weight))
    return lifted


def stage1_residuals(params: np.ndarray, lifted: list[LiftedSet], dist: FieldModel, k_ref: float) -> np.ndarray:
    # Errors are measured in plane units (observation space) and expressed in
    # "reference feet" (k_ref) so the robust scale has an intuitive meaning.
    to_ref = k_ref / np.exp(params[0])
    out = []
    for item in lifted:
        xy = similarity_apply(params, item.uv)
        if item.mode == "centroid":
            out.append(((xy - dist.point(item.name)) * (item.weight * to_ref)).ravel())
        else:
            out.append(dist.query(item.name, xy) * (item.weight * to_ref))
    return np.concatenate(out)


def stage1_grid_search(lifted: list[LiftedSet], dist: FieldModel, k_range: tuple[float, float], top: int) -> list[np.ndarray]:
    """Brute-force search over rotation, scale and translation with raster lookups."""
    rng = np.random.default_rng(4)
    samples = []
    for item in lifted:
        n = min(len(item.uv), 150)
        idx = rng.choice(len(item.uv), n, replace=False)
        samples.append((item, item.uv[idx]))
    uv_all = np.concatenate([s for _, s in samples])
    uv_mean = uv_all.mean(axis=0)
    cap = 15.0
    targets = np.stack(
        np.meshgrid(np.arange(-20.0, 220.1, 5.0), np.arange(-20.0, dist.dims.width + 20.1, 5.0), indexing="xy"), -1
    ).reshape(-1, 2)
    results = []
    for log_k in np.linspace(np.log(k_range[0]), np.log(k_range[1]), 9):
        for theta in np.deg2rad(np.arange(0.0, 360.0, 4.0)):
            base_params = np.asarray([log_k, theta, 0.0, 0.0])
            cost = np.zeros(len(targets))
            centre = similarity_apply(base_params, uv_mean[None])[0]
            for item, uv in samples:
                xy = similarity_apply(base_params, uv) - centre  # centroid at origin
                pts = xy[None, :, :] + targets[:, None, :]
                d = dist.raster_lookup(item.name, pts, cap)
                cost += CLASS_WEIGHT[item.mode] * d.mean(axis=1)
            best = np.argsort(cost)[:3]
            for b in best:
                t = targets[b] - centre
                results.append((float(cost[b]), np.asarray([log_k, theta, t[0], t[1]])))
    results.sort(key=lambda r: r[0])
    chosen: list[np.ndarray] = []
    for _, params in results:
        if all(abs(params[0] - c[0]) > 0.15 or abs(np.angle(np.exp(1j * (params[1] - c[1])))) > np.deg2rad(10) or np.hypot(*(params[2:] - c[2:])) > 12 for c in chosen):
            chosen.append(params)
        if len(chosen) >= top:
            break
    return chosen


def point_landmark_initialisation(lifted: list[LiftedSet]) -> list[np.ndarray]:
    """Closed-form similarity from two or more faceoff-dot observations."""
    pts = [(np.median(item.uv, axis=0), fm.dot_xy(item.name, FieldDims())) for item in lifted if item.mode == "centroid" and len(item.uv) >= 3]
    if len(pts) < 2:
        return []
    src = np.asarray([p[0] for p in pts])
    dst = np.asarray([p[1] for p in pts])
    src_c, dst_c = src - src.mean(0), dst - dst.mean(0)
    zs = src_c[:, 0] + 1j * src_c[:, 1]
    zd = dst_c[:, 0] + 1j * dst_c[:, 1]
    a = (np.conj(zs) @ zd) / (np.conj(zs) @ zs)  # proper similarity only (no mirror)
    k, theta = abs(a), np.angle(a)
    t = dst.mean(0) - similarity_apply(np.asarray([np.log(k), theta, 0.0, 0.0]), src.mean(0)[None])[0]
    return [np.asarray([np.log(k), theta, t[0], t[1]])]


def fit_stage1(lifted: list[LiftedSet], dist: FieldModel, camera_height_units: float, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    k_range = (args.camera_height_min_ft / camera_height_units, args.camera_height_max_ft / camera_height_units)
    candidates = point_landmark_initialisation(lifted) + stage1_grid_search(lifted, dist, k_range, top=args.search_candidates)
    k_ref = float(np.sqrt(k_range[0] * k_range[1]))
    best = None
    for init in candidates:
        result = least_squares(
            stage1_residuals, init, args=(lifted, dist, k_ref), loss="soft_l1", f_scale=args.robust_scale_ft, max_nfev=200, x_scale=[0.1, 0.05, 5.0, 5.0]
        )
        cost = float(result.cost)
        if best is None or cost < best[0]:
            best = (cost, result.x)
    params = best[1]
    k = float(np.exp(params[0]))
    per_class = {}
    for item in lifted:
        xy = similarity_apply(params, item.uv)
        if item.mode == "centroid":
            d = np.linalg.norm(xy - dist.point(item.name), axis=1)
        else:
            d = dist.query(item.name, xy)
        per_class[item.name] = {"observations": int(len(d)), "median_ft": float(np.median(d)), "p90_ft": float(np.percentile(d, 90))}
    info = {
        "ft_per_plane_unit": k,
        "rotation_deg": float(np.rad2deg(params[1]) % 360.0),
        "translation_ft": [float(params[2]), float(params[3])],
        "camera_height_ft": camera_height_units * k,
        "candidates_tried": len(candidates),
        "robust_cost": best[0],
        "residual_ft_by_landmark": per_class,
    }
    return params, info


# ---------------------------------------------------------------------------
# Field-space cameras and stage 2 refinement
# ---------------------------------------------------------------------------


def cameras_in_field(cams: Cameras, plane: Plane, params: np.ndarray) -> Cameras:
    """Express VGGT cameras in field feet (x, y on the floor, z up)."""
    log_k, theta, tx, ty = params
    k = float(np.exp(log_k))
    c, s = np.cos(theta), np.sin(theta)
    Q = np.stack([c * plane.e1 - s * plane.e2, s * plane.e1 + c * plane.e2, plane.normal])  # world vec -> field axes
    C = k * (cams.C - plane.center) @ Q.T + np.asarray([tx, ty, 0.0])
    R = cams.R @ Q.T
    return Cameras(K=cams.K.copy(), R=R, C=C)


def field_ray_floor(cams: Cameras, i: int, pixels: np.ndarray) -> np.ndarray:
    rays = cams.rays(i, pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -cams.C[i, 2] / rays[:, 2]
    xy = cams.C[i, :2] + lam[:, None] * rays[:, :2]
    xy[~(lam > 0)] = np.nan
    return xy


def aim_cameras(base: Cameras, center: np.ndarray) -> Cameras:
    """Move every camera to `center`, re-aiming it at the floor point its optical
    axis hit before and rescaling focal length so that point keeps its image scale.
    This turns VGGT's zoom-as-dolly motion back into a fixed pan/tilt/zoom camera."""
    S = len(base.R)
    K = base.K.copy()
    R = np.empty_like(base.R)
    for i in range(S):
        axis = base.R[i, 2]
        lam = -base.C[i, 2] / axis[2] if axis[2] < -1e-6 else np.nan
        target = base.C[i] + (lam if np.isfinite(lam) and lam > 0 else 300.0) * axis
        z = target - center
        dist_new = np.linalg.norm(z)
        z /= dist_new
        x = base.R[i, 0] - (base.R[i, 0] @ z) * z
        x /= np.linalg.norm(x)
        R[i] = np.stack([x, np.cross(z, x), z])
        scale = dist_new / np.linalg.norm(target - base.C[i])
        K[i, 0, 0] *= scale
        K[i, 1, 1] *= scale
    return Cameras(K=K, R=R, C=np.repeat(center[None], S, axis=0))


def corrected_camera(base: Cameras, i: int, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = Rotation.from_rotvec(delta[:3]).as_matrix() @ base.R[i]
    K = base.K[i].copy()
    K[0, 0] *= np.exp(delta[3])
    K[1, 1] *= np.exp(delta[3])
    return K, R


@dataclass
class FrameObs:
    name: str
    mode: str
    pixels: np.ndarray
    weight: float


STAGE2_MODE_WEIGHT = {"centroid": 1.0, "pixels": 1.0, "boundary": 0.8, "bottom_edge": 0.8, "top_edge": 0.8}


def build_frame_observations(observations: list[Observation], max_per_obs: int, points_per_obs: float) -> dict[int, list[FrameObs]]:
    """Every (frame, landmark) observation counts like `points_per_obs` pixels,
    so dense line masks do not drown out single-point landmarks."""
    rng = np.random.default_rng(5)
    per_frame: dict[int, list[FrameObs]] = {}
    for obs in observations:
        px = obs.pixels
        if len(px) > max_per_obs:
            px = px[rng.choice(len(px), max_per_obs, replace=False)]
        weight = STAGE2_MODE_WEIGHT[obs.mode] * np.sqrt(points_per_obs / len(px))
        per_frame.setdefault(obs.frame, []).append(FrameObs(obs.name, obs.mode, px, weight))
    return per_frame


def smooth_pairs(cams: Cameras, max_rotation_deg: float) -> list[tuple[int, int]]:
    """Neighbouring frames with small relative rotation (i.e. no broadcast cut)."""
    pairs = []
    for i in range(len(cams.R) - 1):
        rel = Rotation.from_matrix(cams.R[i + 1] @ cams.R[i].T).magnitude()
        if np.rad2deg(rel) <= max_rotation_deg:
            pairs.append((i, i + 1))
    return pairs


def _observation_residuals(cams: Cameras, i: int, fo: FrameObs, dims: FieldDims) -> np.ndarray:
    """Floor distance of each observation pixel to its landmark, divided by the local
    floor-feet-per-pixel of the camera, i.e. approximately image pixels."""
    xy = field_ray_floor(cams, i, fo.pixels)
    dx = field_ray_floor(cams, i, fo.pixels + [1.0, 0.0]) - xy
    dy = field_ray_floor(cams, i, fo.pixels + [0.0, 1.0]) - xy
    ft_per_px = 0.5 * (np.linalg.norm(dx, axis=1) + np.linalg.norm(dy, axis=1))
    bad = ~(np.isfinite(xy).all(axis=1) & np.isfinite(ft_per_px) & (ft_per_px > 0))
    xy[bad] = 1e4
    ft_per_px[bad] = 1.0
    if fo.mode == "centroid":
        r = ((xy - fm.dot_xy(fo.name, dims)) / ft_per_px[:, None]).ravel()
    else:
        r = fm.distance(fo.name, xy, dims) / ft_per_px
    return np.clip(r, -400.0, 400.0) * fo.weight


def dims_bounds(names: tuple[str, ...], dims: FieldDims) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for n in names:
        v = getattr(dims, n)
        a, b = {"corner_radius": (8.0, 35.0), "crease_radius": (6.0, 12.5), "crease_back": (1.0, 8.5)}.get(n, (0.75 * v, 1.25 * v))
        lo.append(a)
        hi.append(b)
    return np.asarray(lo), np.asarray(hi)


class Stage2Problem:
    """Shared camera centre + per-frame pan/tilt/roll/zoom corrections, and
    optionally the arena's field dimensions (shared by all frames).

    Parameter vector: [camera centre (3), field dimensions (m), per-frame deltas (4 S)].
    Residuals are landmark distances on the floor divided by the local
    floor-feet-per-pixel of the current camera, i.e. approximately image pixels.
    """

    def __init__(self, rigid: Cameras, per_frame: dict[int, list[FrameObs]], dist: FieldModel, args: argparse.Namespace, dim_names: tuple[str, ...]):
        self.rigid, self.per_frame, self.dist, self.args = rigid, per_frame, dist, args
        self.dim_names = dim_names
        self.m = len(dim_names)
        self.dims0 = dist.dims.vector(dim_names)
        self.off = 3 + self.m
        self.S = len(rigid.R)
        self.pairs = smooth_pairs(rigid, args.cut_rotation_deg)
        self.blocks = []
        rows = 0
        for i in range(self.S):
            start = rows
            for fo in per_frame.get(i, []):
                rows += len(fo.pixels) * (2 if fo.mode == "centroid" else 1)
            self.blocks.append((start, rows))
        self.prior_start = rows
        rows += 4 * self.S
        self.smooth_start = rows
        rows += 4 * len(self.pairs)
        self.dims_start = rows
        rows += self.m
        self.n_rows = rows

    def initial(self, centre: np.ndarray) -> np.ndarray:
        x = np.zeros(self.off + 4 * self.S)
        x[:3] = centre
        x[3 : self.off] = self.dims0
        return x

    def dims(self, x: np.ndarray) -> FieldDims:
        return self.dist.dims.with_vector(self.dim_names, x[3 : self.off])

    def cameras(self, x: np.ndarray) -> Cameras:
        base = aim_cameras(self.rigid, x[:3])
        delta = x[self.off :].reshape(self.S, 4)
        K, R = base.K.copy(), base.R.copy()
        for i in range(self.S):
            K[i], R[i] = corrected_camera(base, i, delta[i])
        return Cameras(K=K, R=R, C=base.C)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        cams = self.cameras(x)
        dims = self.dims(x)
        delta = x[self.off :].reshape(self.S, 4)
        out = np.zeros(self.n_rows)
        for i in range(self.S):
            row = self.blocks[i][0]
            for fo in self.per_frame.get(i, []):
                r = _observation_residuals(cams, i, fo, dims)
                out[row : row + len(r)] = r
                row += len(r)
        focal = cams.K[:, 0, 0]
        half_width = cams.K[:, 0, 2]
        motion = np.column_stack([delta[:, :3] * focal[:, None], delta[:, 3] * half_width])
        out[self.prior_start : self.smooth_start] = (motion * self.args.prior_weight).ravel()
        if self.pairs:
            a = np.asarray([p[0] for p in self.pairs])
            b = np.asarray([p[1] for p in self.pairs])
            out[self.smooth_start : self.dims_start] = ((motion[b] - motion[a]) * self.args.smooth_weight).ravel()
        if self.m:
            out[self.dims_start :] = (x[3 : self.off] - self.dims0) / self.args.dims_prior_sigma_ft * self.args.robust_scale_px
        return out

    def sparsity(self) -> lil_matrix:
        S, off = self.S, self.off
        sp = lil_matrix((self.n_rows, off + 4 * S), dtype=np.int8)
        for i, (start, stop) in enumerate(self.blocks):
            if stop > start:
                sp[start:stop, :off] = 1  # centre and field dimensions touch every observation
                sp[start:stop, off + 4 * i : off + 4 * i + 4] = 1
        for i in range(S):
            for j in range(4):
                sp[self.prior_start + 4 * i + j, :3] = 1
                sp[self.prior_start + 4 * i + j, off + 4 * i + j] = 1
        for p, (a, b) in enumerate(self.pairs):
            for j in range(4):
                sp[self.smooth_start + 4 * p + j, :3] = 1
                sp[self.smooth_start + 4 * p + j, off + 4 * a + j] = 1
                sp[self.smooth_start + 4 * p + j, off + 4 * b + j] = 1
        for j in range(self.m):
            sp[self.dims_start + j, 3 + j] = 1
        return sp


def _initial_centres(problem: Stage2Problem, rigid: Cameras, args: argparse.Namespace) -> tuple[list[tuple[float, np.ndarray]], np.ndarray, np.ndarray]:
    """Candidate shared centres: VGGT's median centre and points pulled towards the
    floor it looks at (VGGT tends to over-estimate focal length and distance for
    telephoto broadcast footage)."""
    centre0 = np.median(rigid.C, axis=0)
    look_at = []
    for i in range(problem.S):
        axis = rigid.R[i, 2]
        if axis[2] < -1e-6:
            look_at.append(rigid.C[i] - rigid.C[i, 2] / axis[2] * axis)
    look_at = np.median(np.asarray(look_at), axis=0) if look_at else np.asarray([100.0, 42.5, 0.0])
    scan = []
    for alpha in (1.0, 0.8, 0.65, 0.5, 0.4, 0.3):
        x = problem.initial(look_at + alpha * (centre0 - look_at))
        r = problem.residuals(x)
        scan.append((float(np.sum(np.log1p((r / args.robust_scale_px) ** 2))), x))
    scan.sort(key=lambda item: item[0])
    return scan, centre0, look_at


def stage2_refine(
    rigid: Cameras,
    per_frame: dict[int, list[FrameObs]],
    dist: FieldModel,
    args: argparse.Namespace,
    dim_names: tuple[str, ...],
    warm_start: np.ndarray | None = None,
) -> tuple[Cameras, FieldDims, dict, np.ndarray]:
    problem = Stage2Problem(rigid, per_frame, dist, args, dim_names)
    S = problem.S
    scan, centre0, look_at = _initial_centres(problem, rigid, args)
    starts = [warm_start] if warm_start is not None else [x for _, x in scan[:2]]
    # per-frame corrections are bounded: at most 20 deg of rotation and 2x zoom
    d_lo, d_hi = dims_bounds(dim_names, dist.dims)
    lower = np.concatenate([np.full(3, -np.inf), d_lo, np.tile([-0.35, -0.35, -0.35, -0.7], S)])
    upper = np.concatenate([np.full(3, np.inf), d_hi, np.tile([0.35, 0.35, 0.35, 0.7], S)])
    sparsity = problem.sparsity()
    best = None
    for x0 in starts:
        x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
        result = least_squares(
            problem.residuals, x0, jac_sparsity=sparsity, bounds=(lower, upper), loss="soft_l1", f_scale=args.robust_scale_px, max_nfev=args.refine_iterations, x_scale="jac", diff_step=1e-4
        )
        if best is None or result.cost < best.cost:
            best = result
    cams = problem.cameras(best.x)
    dims = problem.dims(best.x)
    delta = best.x[problem.off :].reshape(S, 4)
    info = {
        "camera_model": "shared_centre_pan_tilt_roll_zoom",
        "camera_centre_ft": best.x[:3].tolist(),
        "vggt_median_camera_centre_ft": centre0.tolist(),
        "camera_distance_to_look_at_ft": {"vggt": float(np.linalg.norm(centre0 - look_at)), "refined": float(np.linalg.norm(best.x[:3] - look_at))},
        "horizontal_fov_deg": {
            "vggt_median": float(np.median(np.rad2deg(2 * np.arctan(rigid.K[:, 0, 2] / rigid.K[:, 0, 0])))),
            "refined_median": float(np.median(np.rad2deg(2 * np.arctan(cams.K[:, 0, 2] / cams.K[:, 0, 0])))),
        },
        "estimated_field_dims": list(dim_names),
        "frames_with_landmarks": int(sum(1 for i in range(S) if per_frame.get(i))),
        "smoothness_pairs": len(problem.pairs),
        "median_residual_rotation_deg": float(np.median(np.rad2deg(np.linalg.norm(delta[:, :3], axis=1)))),
        "median_abs_residual_zoom_pct": float(np.median(np.abs(np.expm1(delta[:, 3]))) * 100.0),
        "nfev": int(best.nfev),
    }
    return cams, dims, info, best.x


def refit_own_camera(start: Cameras, i: int, items: list[FrameObs], dims: FieldDims, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one frame with its own camera centre (a broadcast cut to another camera)."""

    def residuals(x: np.ndarray) -> np.ndarray:
        K, R = corrected_camera(start, i, x[3:])
        cam = Cameras(K=K[None], R=R[None], C=(start.C[i] + x[:3])[None])
        out = [_observation_residuals(cam, 0, fo, dims) for fo in items]
        out.append(np.concatenate([x[3:6] * start.K[i, 0, 0], [x[6] * start.K[i, 0, 2]]]) * args.prior_weight)
        return np.concatenate(out)

    result = least_squares(residuals, np.zeros(7), loss="soft_l1", f_scale=args.robust_scale_px, max_nfev=200, x_scale=[5.0, 5.0, 5.0, 1e-3, 1e-3, 1e-3, 1e-3], diff_step=1e-4)
    K, R = corrected_camera(start, i, result.x[3:])
    return K, R, start.C[i] + result.x[:3]


def off_camera_frames(cams: Cameras, observations: list[Observation], shape: tuple[int, int], threshold_px: float, dims: FieldDims) -> dict[int, float]:
    errors = reprojection_errors([field_to_image_homography(cams, i) for i in range(len(cams.R))], observations, shape, dims)["by_frame"]
    return {i: err for i, err in errors.items() if err > threshold_px}


def fit_own_cameras(
    final: Cameras,
    rigid: Cameras,
    frames: dict[int, float],
    observations: list[Observation],
    per_frame: dict[int, list[FrameObs]],
    dims: FieldDims,
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[Cameras, dict[int, str]]:
    """Frames the shared broadcast camera cannot explain (cuts to another camera)
    get their own camera; frames that still do not fit are flagged unreliable."""
    K, R, C = final.K.copy(), final.R.copy(), final.C.copy()
    status = {}
    for i, err in frames.items():
        frame_obs = [Observation(o.name, o.mode, 0, o.pixels) for o in observations if o.frame == i]
        candidates = []
        for start in (rigid, final):
            Ki, Ri, Ci = refit_own_camera(start, i, per_frame.get(i, []), dims, args)
            trial = Cameras(K=Ki[None], R=Ri[None], C=Ci[None])
            candidates.append((reprojection_errors([field_to_image_homography(trial, 0)], frame_obs, shape, dims)["median_px"], Ki, Ri, Ci))
        best = min(candidates, key=lambda c: c[0])
        if best[0] < min(err, args.outlier_frame_px):
            K[i], R[i], C[i] = best[1], best[2], best[3]
            status[i] = "own_camera"
        else:
            status[i] = "unreliable"
    return Cameras(K=K, R=R, C=C), status


# ---------------------------------------------------------------------------
# Evaluation and outputs
# ---------------------------------------------------------------------------


def field_to_image_homography(cams: Cameras, i: int) -> np.ndarray:
    R, K, C = cams.R[i], cams.K[i], cams.C[i]
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def project_field(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p[:, :2] / p[:, 2:3]
    out[p[:, 2] <= 0] = np.nan
    return out


def reprojection_errors(homographies: list[np.ndarray], observations: list[Observation], shape: tuple[int, int], dims: FieldDims) -> dict:
    """Image-space distance (px) from each landmark pixel to its projected model curve."""
    curves: dict[str, np.ndarray] = {}
    h, w = shape
    by_class: dict[str, list[np.ndarray]] = {}
    by_frame: dict[int, list[np.ndarray]] = {}
    for obs in observations:
        if obs.name not in curves:
            curves[obs.name] = fm.curve(obs.name, dims, 6.0)
        model = project_field(homographies[obs.frame], curves[obs.name])
        ok = np.isfinite(model).all(axis=1) & (np.abs(model[:, 0]) < 4 * w) & (np.abs(model[:, 1]) < 4 * h)
        if not ok.any():
            d = np.full(len(obs.pixels), 4.0 * w)
        else:
            d = cKDTree(model[ok]).query(obs.pixels)[0]
        if obs.mode == "centroid":
            d = d[:1]
        by_class.setdefault(obs.name, []).append(d)
        by_frame.setdefault(obs.frame, []).append(d)
    summary = {}
    for name, parts in by_class.items():
        d = np.concatenate(parts)
        summary[name] = {"median_px": float(np.median(d)), "p90_px": float(np.percentile(d, 90)), "count": int(len(d))}
    frame_medians = {int(i): float(np.median(np.concatenate(parts))) for i, parts in by_frame.items()}
    all_d = np.concatenate([np.concatenate(p) for p in by_class.values()])
    return {
        "median_px": float(np.median(all_d)),
        "p90_px": float(np.percentile(all_d, 90)),
        "median_of_frame_medians_px": float(np.median(list(frame_medians.values()))),
        "worst_frame_median_px": float(max(frame_medians.values())),
        "by_landmark": summary,
        "by_frame": frame_medians,
    }


def _circle_fit(xy: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    keep = np.ones(len(xy), dtype=bool)
    for _ in range(5):  # trimmed algebraic circle fit
        sol, *_ = np.linalg.lstsq(A[keep], (xy[keep] ** 2).sum(axis=1), rcond=None)
        centre = sol[:2]
        radius = float(np.sqrt(sol[2] + centre @ centre))
        err = np.abs(np.linalg.norm(xy - centre, axis=1) - radius)
        keep = err <= np.percentile(err, 70)
    return centre, radius


def landmark_positions(cams: Cameras, observations: list[Observation], dims: FieldDims) -> dict:
    """Where each landmark lands on the floor under the given cameras, compared
    with the field model. Large systematic offsets point at model geometry
    errors rather than camera errors."""
    lifted: dict[str, list[np.ndarray]] = {}
    for obs in observations:
        xy = field_ray_floor(cams, obs.frame, obs.pixels)
        lifted.setdefault(obs.name, []).append(xy[np.isfinite(xy).all(axis=1)])
    report = {}
    for name, parts in lifted.items():
        xy = np.concatenate(parts)
        if len(xy) == 0:
            continue
        d = fm.distance(name, xy, dims) if DEFAULT_MODE[name] != "centroid" else np.linalg.norm(xy - fm.dot_xy(name, dims), axis=1)
        entry = {"median_distance_to_model_ft": float(np.median(d)), "p90_distance_to_model_ft": float(np.percentile(d, 90))}
        if name in fm.LINE_NAMES:
            entry.update({"model_x_ft": fm.line_x(name, dims), "median_x_ft": float(np.median(xy[:, 0])), "iqr_x_ft": float(np.subtract(*np.percentile(xy[:, 0], [75, 25])))})
        elif name in fm.DOT_NAMES:
            entry.update({"model_xy_ft": fm.dot_xy(name, dims).tolist(), "median_xy_ft": np.median(xy, axis=0).tolist(), "std_xy_ft": xy.std(axis=0).tolist()})
        elif "crease" in name and len(xy) >= 10:
            side = "left" if name.startswith("left") else "right" if name.startswith("right") else None
            near = xy if side is None else xy[fm.distance(f"{side}_goal_crease", xy, dims) <= fm.distance(f"{side}_goal_line", xy, dims)]
            if len(near) >= 10:
                centre, radius = _circle_fit(near)
                entry.update({"circle_centre_ft": centre.tolist(), "circle_radius_ft": radius})
        report[name] = entry
    return report


def draw_field_on_frame(frame: np.ndarray, H: np.ndarray, dims: FieldDims, thickness: int = 2) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]
    for kind, poly in fm.drawing(dims):
        px = project_field(H, poly)
        if kind == "dots":
            p = px[0]
            if np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(out, tuple(np.round(p).astype(int)), 6, DRAW_COLORS_BGR["dots"], thickness, cv2.LINE_AA)
            continue
        ok = np.isfinite(px).all(axis=1) & (np.abs(px[:, 0]) < 3 * w) & (np.abs(px[:, 1]) < 3 * h)
        for seg in np.split(px, np.nonzero(~ok)[0]):
            seg = seg[np.isfinite(seg).all(axis=1)]
            if len(seg) >= 2:
                cv2.polylines(out, [np.round(seg).astype(np.int32)], False, DRAW_COLORS_BGR[kind], thickness, cv2.LINE_AA)
    return out


def render_reprojection_video(
    frames_dir: Path,
    frame_names: list[str],
    before: list[np.ndarray],
    after: list[np.ndarray],
    errors_before: dict,
    errors_after: dict,
    dims_before: FieldDims,
    dims_after: FieldDims,
    path: Path,
    fps: float,
) -> None:
    frames = []
    for i, name in enumerate(frame_names):
        img = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        left = draw_field_on_frame(img, before[i], dims_before)
        right = draw_field_on_frame(img, after[i], dims_after)
        for panel, label, errs in ((left, "planar fit (rigid)", errors_before), (right, "planar fit + camera refinement", errors_after)):
            err = errs["by_frame"].get(i)
            text = f"{label}   frame {i}" + (f"   median {err:.1f}px" if err is not None else "")
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(panel, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        combo = np.vstack([cv2.resize(left, (960, 540)), cv2.resize(right, (960, 540))])
        frames.append(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB))
    write_h264_video(frames, path, fps)


def field_canvas(dims: FieldDims, width: int = 1280, height: int = 640, margin: int = 50):
    """Top-down drawing of the (fitted) field and a field-ft -> canvas-px mapper."""
    x0, x1 = 100.0 - dims.half_length, 100.0 + dims.half_length
    scale = min((width - 2 * margin) / (x1 - x0), (height - 2 * margin) / dims.width)

    def to_px(xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return np.column_stack([margin + (xy[:, 0] - x0) * scale, height - margin - xy[:, 1] * scale])

    canvas = np.full((height, width, 3), (28, 74, 54), np.uint8)
    outline = np.round(to_px(fm.outline_polyline(dims))).astype(np.int32)
    cv2.fillPoly(canvas, [outline], (35, 108, 78), cv2.LINE_AA)
    colors = {"outline": (0, 220, 255), "lines": (240, 240, 235), "crease": (80, 80, 255), "dots": (240, 240, 235)}
    for kind, poly in fm.drawing(dims):
        px = np.round(to_px(poly)).astype(np.int32)
        if kind == "dots":
            cv2.circle(canvas, tuple(px[0]), 4, colors[kind], -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [px], False, colors[kind], 2, cv2.LINE_AA)
    return canvas, to_px


def render_landmarks_on_field(lifted: list[LiftedSet], params: np.ndarray, path: Path, dims: FieldDims) -> None:
    canvas, to_px = field_canvas(dims)
    palette = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0), (180, 120, 255), (255, 160, 60)]
    for k, item in enumerate(lifted):
        xy = similarity_apply(params, item.uv)
        xy = xy[(xy[:, 0] > -20) & (xy[:, 0] < 220) & (xy[:, 1] > -20) & (xy[:, 1] < dims.width + 20)]
        color = palette[k % len(palette)]
        for x, y in np.round(to_px(xy[:: max(1, len(xy) // 2500)])).astype(int):
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, item.name, (20, 30 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="npz written by run_vggt_omega_clip.py")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--landmark", action="append", default=[], help=f"NAME=MASK_DIR[#OBJECT_ID]; NAME in {sorted(DEFAULT_MODE)}")
    parser.add_argument("--paint-masks", default=None, help="Output directory of build_field_paint_masks.py; adds all of its labeled landmarks.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detect-white-paint", action="store_true", help="Add colour-detected white paint on the turf as unlabeled white_lines.")
    parser.add_argument("--player-mask-dir", default=None, help="Player instance masks to exclude from --detect-white-paint.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--plane-points-per-frame", type=int, default=4000)
    parser.add_argument("--plane-conf-percentile", type=float, default=30.0)
    parser.add_argument("--max-points-per-mask", type=int, default=250)
    parser.add_argument("--max-points-per-class", type=int, default=6000)
    parser.add_argument("--min-dot-area", type=int, default=8)
    parser.add_argument("--max-dot-area", type=int, default=4000)
    parser.add_argument("--camera-height-min-ft", type=float, default=20.0, help="Prior range for the broadcast camera height; sets the scale search.")
    parser.add_argument("--camera-height-max-ft", type=float, default=120.0)
    parser.add_argument("--search-candidates", type=int, default=12)
    parser.add_argument("--robust-scale-ft", type=float, default=1.5)
    parser.add_argument("--no-camera-refinement", action="store_true")
    parser.add_argument("--robust-scale-px", type=float, default=3.0)
    parser.add_argument("--points-per-observation", type=float, default=40.0)
    parser.add_argument("--prior-weight", type=float, default=0.03)
    parser.add_argument("--smooth-weight", type=float, default=0.3)
    parser.add_argument("--cut-rotation-deg", type=float, default=4.0, help="Neighbouring frames rotating more than this are treated as a cut (no smoothing).")
    parser.add_argument("--refine-iterations", type=int, default=60)
    parser.add_argument("--outlier-frame-px", type=float, default=15.0, help="Frames above this median error get their own camera (broadcast cuts).")
    parser.add_argument(
        "--estimate-dims",
        default="observed",
        help=f"Field dimensions to estimate with the cameras: 'observed' (the arena-specific {list(FieldDims.ESTIMABLE)} that the landmarks constrain), 'none', or a comma list that may also include rulebook dimensions {list(FieldDims.DIAGNOSTIC)} as a check. The 85 ft width is the fixed scale.",
    )
    parser.add_argument("--dims-prior-sigma-ft", type=float, default=5.0, help="Soft prior pulling estimated dimensions towards the NLL defaults.")
    parser.add_argument("--centre-circle-radius-ft", type=float, default=0.0, help="Painted centre faceoff circle radius (0 = none; neither arena here has one).")
    args = parser.parse_args()
    if args.paint_masks:
        root = Path(args.paint_masks)
        ids = json.loads((root / "labels.json").read_text())["object_ids"]
        args.landmark += [f"{name}={root / 'masks'}#{oid}" for name, oid in ids.items()]
    if not args.landmark:
        parser.error("give --landmark and/or --paint-masks")
    return args


# Which observed landmarks constrain which field dimensions.
_CREASES = ("left_crease_lines", "right_crease_lines", "left_goal_crease", "right_goal_crease", "goal_crease", "left_goal_line", "right_goal_line")
_BOARDS = ("board_base", "field_outline", "far_boards_edge")
DIM_EVIDENCE = {
    "half_length": _CREASES + _BOARDS,  # goals sit 12' from the end boards
    "corner_radius": _BOARDS,
    "restraining_offset": ("left_restraining_line", "right_restraining_line"),
    "goal_from_end": _CREASES,
    "crease_radius": _CREASES,
    "crease_back": _CREASES,
    "dot_offset_x": fm.DOT_NAMES,
    "dot_offset_y": fm.DOT_NAMES,
}


def dims_to_estimate(spec: str, observed: set[str]) -> tuple[str, ...]:
    if spec == "none":
        return ()
    if spec == "observed":
        return tuple(n for n in FieldDims.ESTIMABLE if observed & set(DIM_EVIDENCE[n]))
    names = tuple(n.strip() for n in spec.split(",") if n.strip())
    unknown = [n for n in names if n not in FieldDims.ESTIMABLE + FieldDims.DIAGNOSTIC]
    if unknown:
        raise ValueError(f"Cannot estimate {unknown}; choose from {FieldDims.ESTIMABLE + FieldDims.DIAGNOSTIC}")
    return names


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(Path(args.predictions))
    cams: Cameras = pred["cameras"]
    frame_names = [str(n) for n in pred["frame_names"]]
    shape = tuple(int(v) for v in pred["original_hw"])
    default_dims = FieldDims(centre_circle_radius=args.centre_circle_radius_ft)

    floor_pts, _ = floor_depth_points(pred, args.plane_points_per_frame, args.plane_conf_percentile)
    plane, plane_stats = fit_floor_plane(floor_pts, cams.C)
    observations = collect_observations(args, frame_names, shape)
    if not observations:
        raise RuntimeError("No landmark observations found; check --landmark / --paint-masks paths.")
    lifted = lift_observations(observations, cams, plane, args.max_points_per_class)
    dist = FieldModel(default_dims, {item.name for item in lifted})

    params, stage1 = fit_stage1(lifted, dist, plane_stats["camera_height_median_units"], args)
    rigid_cams = cameras_in_field(cams, plane, params)
    rigid_H = [field_to_image_homography(rigid_cams, i) for i in range(len(frame_names))]
    rigid_errors = reprojection_errors(rigid_H, observations, shape, default_dims)

    dims = default_dims
    if args.no_camera_refinement:
        final_cams, stage2 = rigid_cams, {"skipped": True}
    else:
        dim_names = dims_to_estimate(args.estimate_dims, {o.name for o in observations})
        per_frame = build_frame_observations(observations, args.max_points_per_mask, args.points_per_observation)
        final_cams, dims, stage2, x = stage2_refine(rigid_cams, per_frame, dist, args, dim_names)
        off = off_camera_frames(final_cams, observations, shape, args.outlier_frame_px, dims)
        if off:  # refit the shared camera without the frames it cannot explain
            shared_frames = {i: v for i, v in per_frame.items() if i not in off}
            final_cams, dims, stage2, x = stage2_refine(rigid_cams, shared_frames, dist, args, dim_names, warm_start=x)
            off = off_camera_frames(final_cams, observations, shape, args.outlier_frame_px, dims)
        final_cams, own_status = fit_own_cameras(final_cams, rigid_cams, off, observations, per_frame, dims, shape, args)
        frame_status = {i: ("shared_camera" if per_frame.get(i) else "no_landmarks") for i in range(len(frame_names))}
        frame_status.update(own_status)
        stage2["frame_status_counts"] = {k: list(frame_status.values()).count(k) for k in sorted(set(frame_status.values()))}
        stage2["frame_status"] = {frame_names[i]: v for i, v in frame_status.items()}
    final_H = [field_to_image_homography(final_cams, i) for i in range(len(frame_names))]
    final_errors = reprojection_errors(final_H, observations, shape, dims)

    positions = {"rigid": landmark_positions(rigid_cams, observations, default_dims), "refined": landmark_positions(final_cams, observations, dims)}
    frames_dir = Path(args.frames_dir)
    render_landmarks_on_field(lifted, params, out_dir / "landmarks_on_field.jpg", default_dims)
    render_reprojection_video(frames_dir, frame_names, rigid_H, final_H, rigid_errors, final_errors, default_dims, dims, out_dir / "field_reprojection_h264.mp4", args.fps)

    all_frames = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    # Unreliable frames are left out so consumers fall back to the nearest good frame.
    homographies = []
    for i, name in enumerate(frame_names):
        status = stage2.get("frame_status", {}).get(name, "rigid")
        if status == "unreliable":
            continue
        H_img_to_field = np.linalg.inv(final_H[i])
        homographies.append(
            {
                "frame": all_frames.index(name),
                "frame_name": name,
                "matrix_image_to_world": (H_img_to_field / H_img_to_field[2, 2]).tolist(),
                "mean_error_px": float(final_errors["by_frame"].get(i, np.nan)),
                "status": status,
            }
        )
    (out_dir / "field_homographies.json").write_text(
        json.dumps(
            {"schema": "floor_homography_matrix_v1", "world_units": "ft", "source": "fit_field_plane_landmarks.py", "field_dims": dims.to_dict(), "homographies": homographies}, indent=1
        )
        + "\n"
    )
    dims_report = {
        name: {"default_ft": getattr(default_dims, name), "fitted_ft": getattr(dims, name), "estimated": name in stage2.get("estimated_field_dims", [])}
        for name in FieldDims.ESTIMABLE + FieldDims.DIAGNOSTIC
    }
    result = {
        "schema": "field_plane_fit_v1",
        "predictions": args.predictions,
        "frames_dir": args.frames_dir,
        "landmarks": args.landmark,
        "frame_names": frame_names,
        "field_dims": dims.to_dict(),
        "field_dims_report": dims_report,
        "plane": {"center": plane.center.tolist(), "normal": plane.normal.tolist(), "e1": plane.e1.tolist(), "e2": plane.e2.tolist(), **plane_stats},
        "plane_to_field_similarity": {"log_k": float(params[0]), "theta_rad": float(params[1]), "tx_ft": float(params[2]), "ty_ft": float(params[3])},
        "stage1_planar_fit": stage1,
        "stage2_camera_refinement": stage2,
        "landmark_positions_ft": positions,
        "reprojection_rigid": rigid_errors,
        "reprojection_refined": final_errors,
        "cameras_field_ft": {
            "note": "x_img ~ K R (X - C), X in field feet with z up; K in original-image pixels.",
            "rigid": {"K": rigid_cams.K.tolist(), "R": rigid_cams.R.tolist(), "C": rigid_cams.C.tolist()},
            "refined": {"K": final_cams.K.tolist(), "R": final_cams.R.tolist(), "C": final_cams.C.tolist()},
        },
    }
    (out_dir / "field_plane_fit.json").write_text(json.dumps(result, indent=1) + "\n")
    summary = {
        "output_dir": str(out_dir),
        "observations": len(observations),
        "plane": {k: plane_stats[k] for k in ("inlier_fraction", "inlier_rms_over_scene_scale")},
        "stage1": {k: stage1[k] for k in ("ft_per_plane_unit", "camera_height_ft", "rotation_deg")},
        "stage1_residual_ft": {k: round(v["median_ft"], 2) for k, v in stage1["residual_ft_by_landmark"].items()},
        "reprojection_rigid_median_px": round(rigid_errors["median_px"], 2),
        "reprojection_refined_median_px": round(final_errors["median_px"], 2),
        "reprojection_rigid_by_landmark": {k: round(v["median_px"], 1) for k, v in rigid_errors["by_landmark"].items()},
        "reprojection_refined_by_landmark": {k: round(v["median_px"], 1) for k, v in final_errors["by_landmark"].items()},
        "field_dims": {k: round(v["fitted_ft"], 2) for k, v in dims_report.items() if v["estimated"]},
        "stage2": {k: v for k, v in stage2.items() if k != "frame_status"},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
