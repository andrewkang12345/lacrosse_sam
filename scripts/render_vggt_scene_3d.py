"""Render the VGGT-Omega reconstruction in 3D, placed on the fitted NLL field.

Two H.264 videos are written to --output-dir:

  vggt_scene_orbit_h264.mp4     accumulated point cloud of the clip (frames the
                                shared broadcast camera explains). The
                                camera starts at the recovered broadcast camera,
                                orbits the field, rises to a top-down view and
                                holds it.
  vggt_scene_timeline_h264.mp4  one output frame per input frame: the input frame
                                with the fitted field (top left), that frame's
                                VGGT-Omega points in an oblique 3D view (top
                                right) and from directly above (bottom).

Point placement (--geometry):

  refined (default)  every pixel is placed along the ray of the *refined* camera
                     of its frame (the camera that puts the painted lines on the
                     field), at the height above that frame's floor that
                     VGGT-Omega gives it (heights under 1 ft snap to the floor).
                     Floor pixels land on z = 0, so the floor paint of every
                     frame lands on the fitted lines; players and boards keep
                     VGGT-Omega's heights. Points too high for that (crowd above
                     the camera) keep VGGT's distance, rescaled per frame.
  raw                VGGT-Omega depth unprojected with VGGT-Omega's own cameras and
                     mapped to field feet by the rigid plane + similarity. This
                     shows the reconstruction as VGGT-Omega made it, including its
                     telephoto distortion and frame-to-frame drift.

The field is drawn with the dimensions fitted for this arena (field_dims in the
fit JSON). Red frustums are VGGT-Omega's own cameras, the yellow frustum path is
the refined broadcast camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import field_model as fm
from fit_field_plane_landmarks import Cameras, draw_field_on_frame, field_to_image_homography
from video_utils import write_h264_video

FIELD_CENTER = np.asarray([100.0, fm.FieldDims().cy, 0.0])
FIELD_LENGTH_FT, FIELD_WIDTH_FT = 200.0, fm.FieldDims().width
BACKGROUND = (18, 20, 24)
CROP_MIN = np.asarray([-45.0, -60.0, -6.0])
CROP_MAX = np.asarray([FIELD_LENGTH_FT + 45.0, FIELD_WIDTH_FT + 60.0, 70.0])


# ---------------------------------------------------------------------------
# Scene data
# ---------------------------------------------------------------------------


def world_to_field_transform(fit: dict) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    plane = fit["plane"]
    sim = fit["plane_to_field_similarity"]
    e1, e2, n = (np.asarray(plane[key]) for key in ("e1", "e2", "normal"))
    c, s = np.cos(sim["theta_rad"]), np.sin(sim["theta_rad"])
    Q = np.stack([c * e1 - s * e2, s * e1 + c * e2, n])
    return float(np.exp(sim["log_k"])), Q, np.asarray(plane["center"]), np.asarray([sim["tx_ft"], sim["ty_ft"], 0.0])


def frame_points(
    pred: dict, i: int, k: float, Q: np.ndarray, center: np.ndarray, shift: np.ndarray, conf_percentile: float, stride: int, refined: Cameras | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Field-feet points and RGB colours of frame i (see --geometry in the module doc)."""
    depth, conf, colors = pred["depth"][i], pred["depth_conf"][i], pred["colors"][i]
    E, K = pred["extrinsic"][i].astype(np.float64), pred["intrinsic"][i].astype(np.float64)
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    ys, xs = ys.ravel(), xs.ravel()
    d = depth[ys, xs]
    keep = (conf[ys, xs] >= np.percentile(conf, conf_percentile)) & (d > 0)
    ys, xs, d = ys[keep], xs[keep], d[keep]
    cam = np.column_stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * d, (ys + 0.5 - K[1, 2]) / K[1, 1] * d, d])
    world = (cam - E[:, 3]) @ E[:, :3]
    field = k * (world - center) @ Q.T + shift
    if refined is not None:
        field[:, 2] = height_above_frame_floor(field, colors[ys, xs])
        field = place_on_refined_rays(field, xs, ys, depth.shape, pred["original_hw"], refined, i, k * (-E[:, :3].T @ E[:, 3] - center) @ Q.T + shift)
    inside = np.all((field >= CROP_MIN) & (field <= CROP_MAX), axis=1) & np.isfinite(field).all(axis=1)
    return field[inside], colors[ys[inside], xs[inside]]


def height_above_frame_floor(field: np.ndarray, rgb: np.ndarray, snap_ft: float = 1.0) -> np.ndarray:
    """Height of each point above *this frame's* floor.

    VGGT-Omega's floor offset and tilt drift from frame to frame, so heights above
    the clip-wide plane are not zero on the floor. A plane is fitted to this frame's
    turf points (green, trimmed least squares) and heights are taken from it. Heights
    below snap_ft are set to zero: paint and turf lie on the floor."""
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.int32)
    turf = (hsv[:, 0] >= 38) & (hsv[:, 0] <= 95) & (hsv[:, 1] >= 45) & (hsv[:, 2] >= 35)
    pts = field[turf]
    if len(pts) < 200:
        h = field[:, 2]
    else:
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        keep = np.ones(len(pts), bool)
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(A[keep], pts[keep, 2], rcond=None)
            res = np.abs(A @ coef - pts[:, 2])
            keep = res <= max(np.percentile(res, 70), 1e-3)
        h = field[:, 2] - (field[:, 0] * coef[0] + field[:, 1] * coef[1] + coef[2])
    h = np.maximum(h, 0.0)
    h[h < snap_ft] = 0.0
    return h


def place_on_refined_rays(
    raw: np.ndarray, xs: np.ndarray, ys: np.ndarray, model_hw: tuple[int, int], original_hw: np.ndarray, cams: Cameras, i: int, raw_centre: np.ndarray
) -> np.ndarray:
    """Move each point onto the refined camera's ray through its pixel, at the height
    above the floor VGGT-Omega gives it."""
    oh, ow = (int(v) for v in original_hw)
    px = np.column_stack([(xs + 0.5) * ow / model_hw[1], (ys + 0.5) * oh / model_hw[0]])
    rays = cams.rays(i, px)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    C = cams.C[i]
    height = np.maximum(raw[:, 2], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (height - C[2]) / rays[:, 2]
    by_height = (rays[:, 2] < -1e-3) & (height < C[2] - 3.0) & (lam > 0)
    # the rest (crowd at or above camera height): VGGT's distance, rescaled to the refined camera
    raw_dist = np.linalg.norm(raw - raw_centre, axis=1)
    scale = np.median(lam[by_height] / raw_dist[by_height]) if by_height.sum() >= 50 else 1.0
    lam = np.where(by_height, lam, raw_dist * scale)
    return C + lam[:, None] * rays


def field_cameras(fit: dict, key: str) -> Cameras:
    cams = fit["cameras_field_ft"][key]
    return Cameras(K=np.asarray(cams["K"]), R=np.asarray(cams["R"]), C=np.asarray(cams["C"]))


# ---------------------------------------------------------------------------
# Software renderer (z-buffered point splats + anti-aliased overlays)
# ---------------------------------------------------------------------------


def look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray = np.asarray([0.0, 0.0, 1.0])) -> np.ndarray:
    z = target - eye
    z /= np.linalg.norm(z)
    if abs(z @ up_hint) > 0.999:
        up_hint = np.asarray([0.0, 1.0, 0.0])
    x = np.cross(z, up_hint)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # world -> camera (OpenCV: x right, y down, z forward)


class View:
    def __init__(self, eye: np.ndarray, R: np.ndarray, focal: float, width: int, height: int):
        self.eye, self.R, self.f, self.w, self.h = eye, R, focal, width, height

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = (pts - self.eye) @ self.R.T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.column_stack([self.f * cam[:, 0] / z + self.w / 2.0, self.f * cam[:, 1] / z + self.h / 2.0])
        return uv, z


def render_points(view: View, pts: np.ndarray, colors: np.ndarray, point_px: int) -> np.ndarray:
    img = np.empty((view.h, view.w, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    uv, z = view.project(pts)
    ok = (z > 1.0) & np.isfinite(uv).all(axis=1)
    uv, z, col = uv[ok], z[ok], colors[ok]
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    # each point is splatted as a point_px x point_px square
    offsets = [(0, 0)] if point_px <= 1 else [(dx, dy) for dx in range(point_px) for dy in range(point_px)]
    all_idx, all_z, all_c = [], [], []
    for dx, dy in offsets:
        uu, vv = u + dx, v + dy
        inside = (uu >= 0) & (uu < view.w) & (vv >= 0) & (vv < view.h)
        all_idx.append(vv[inside] * view.w + uu[inside])
        all_z.append(z[inside])
        all_c.append(col[inside])
    if not all_idx:
        return img
    idx, zz, cc = np.concatenate(all_idx), np.concatenate(all_z), np.concatenate(all_c)
    zbuf = np.full(view.w * view.h, np.inf)
    np.minimum.at(zbuf, idx, zz)  # z-buffer: nearest depth per pixel
    nearest = zz <= zbuf[idx]
    img.reshape(-1, 3)[idx[nearest]] = cc[nearest][:, ::-1]  # RGB -> BGR
    return img


def draw_polyline_3d(img: np.ndarray, view: View, pts: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    uv, z = view.project(pts)
    ok = (z > 1.0) & np.isfinite(uv).all(axis=1)
    for seg in np.split(np.arange(len(pts)), np.nonzero(~ok)[0]):
        seg = seg[ok[seg]]
        if len(seg) >= 2:
            cv2.polylines(img, [np.round(uv[seg] * 4).astype(np.int32)], False, color, thickness, cv2.LINE_AA, shift=2)


FIELD_COLORS = {"lines": (235, 235, 235), "crease": (80, 80, 255), "outline": (0, 210, 255)}


FIELD_COLORS = {**FIELD_COLORS, "dots": (235, 235, 235)}


def draw_field(img: np.ndarray, view: View, dims: fm.FieldDims, alpha: float = 0.55) -> np.ndarray:
    overlay = img.copy()
    for kind, poly in fm.drawing(dims):
        if kind == "dots":
            uv, z = view.project(np.column_stack([poly, np.zeros(len(poly))]))
            if z[0] > 1.0 and np.isfinite(uv).all():
                cv2.circle(overlay, tuple(np.round(uv[0]).astype(int)), 3, FIELD_COLORS[kind], 1, cv2.LINE_AA)
            continue
        draw_polyline_3d(overlay, view, np.column_stack([poly, np.zeros(len(poly))]), FIELD_COLORS[kind], 1)
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)


def fit_dims(fit: dict) -> fm.FieldDims:
    return fm.FieldDims(**fit["field_dims"]) if "field_dims" in fit else fm.FieldDims()


def frustum_lines(cams: Cameras, i: int, size_ft: float) -> list[np.ndarray]:
    K, R, C = cams.K[i], cams.R[i], cams.C[i]
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    corners = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rays = np.column_stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)]) @ R
    tips = C + size_ft * rays / np.linalg.norm(rays, axis=1, keepdims=True)
    lines = [np.stack([C, tip]) for tip in tips]
    lines.append(np.vstack([tips, tips[:1]]))
    return lines


def draw_cameras(img: np.ndarray, view: View, cams: Cameras, color: tuple[int, int, int], every: int, size_ft: float) -> None:
    for i in range(0, len(cams.R), every):
        if np.linalg.norm(cams.C[i] - view.eye) < max(80.0, 3.0 * size_ft):  # skip frustums at the viewpoint
            continue
        for line in frustum_lines(cams, i, size_ft):
            draw_polyline_3d(img, view, line, color, 1)
    # camera path, broken at cuts (large jumps between consecutive centres)
    jumps = np.linalg.norm(np.diff(cams.C, axis=0), axis=1) > 30.0
    for seg in np.split(np.arange(len(cams.C)), np.nonzero(jumps)[0] + 1):
        if len(seg) >= 2:
            draw_polyline_3d(img, view, cams.C[seg], color, 1)


def label(img: np.ndarray, text: str, origin: tuple[int, int] = (14, 28), scale: float = 0.7) -> None:
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Camera paths
# ---------------------------------------------------------------------------


def orbit_eye(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    a, e = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    return FIELD_CENTER + distance * np.asarray([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def orbit_path(broadcast_eye: np.ndarray, fps: float, width: int) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """(eye, rotation, focal, caption) per output frame."""
    focal = 0.9 * width
    rel = broadcast_eye - FIELD_CENTER
    az0 = float(np.rad2deg(np.arctan2(rel[1], rel[0])))
    el0 = float(np.rad2deg(np.arcsin(rel[2] / np.linalg.norm(rel))))
    dist0 = float(np.linalg.norm(rel))
    orbit_el, orbit_dist = 30.0, 360.0
    frames = []

    def add(n: int, fn) -> None:
        for j in range(n):
            frames.append(fn(j / max(1, n - 1)))

    def at(az: float, el: float, dist: float, caption: str, up=np.asarray([0.0, 0.0, 1.0])):
        eye = orbit_eye(az, el, dist)
        return eye, look_at(eye, FIELD_CENTER, up), focal, caption

    add(int(1.5 * fps), lambda t: at(az0, el0, dist0, "VGGT-Omega point cloud from the broadcast camera position"))
    add(int(2.0 * fps), lambda t: at(az0, el0 + (orbit_el - el0) * smoothstep(t), dist0 + (orbit_dist - dist0) * smoothstep(t), "moving out"))
    add(int(9.0 * fps), lambda t: at(az0 + 360.0 * smoothstep(t), orbit_el, orbit_dist, "orbit"))
    # rise to top-down; keep the field's long axis horizontal on screen
    top_up = np.asarray([0.0, 1.0, 0.0])

    def rise(t: float):
        s = smoothstep(t)
        el = orbit_el + (89.9 - orbit_el) * s
        dist = orbit_dist + (270.0 - orbit_dist) * s
        az = az0 + (-90.0 - az0) * s if abs(az0 + 90.0) < 180.0 else az0
        eye = orbit_eye(az, el, dist)
        return eye, look_at(eye, FIELD_CENTER, np.asarray([0.0, 0.0, 1.0]) * (1 - s) + top_up * s), focal, "rising to top-down"

    add(int(3.0 * fps), rise)
    eye_top = FIELD_CENTER + np.asarray([0.0, 0.0, 270.0])
    add(int(4.0 * fps), lambda t: (eye_top, look_at(eye_top, FIELD_CENTER, top_up), focal, "top-down"))
    return frames


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def refined_or_none(fit: dict, args: argparse.Namespace) -> Cameras | None:
    return field_cameras(fit, "refined") if args.geometry == "refined" else None


def render_orbit(pred: dict, fit: dict, out: Path, args: argparse.Namespace) -> None:
    k, Q, center, shift = world_to_field_transform(fit)
    dims = fit_dims(fit)
    S = pred["depth"].shape[0]
    rng = np.random.default_rng(0)
    pts, cols = [], []
    status = fit.get("stage2_camera_refinement", {}).get("frame_status", {})
    for i in range(S):
        # the accumulated cloud uses frames the shared broadcast camera explains; cuts,
        # close-ups with their own camera and frames without landmarks are left out
        if args.geometry == "refined" and status.get(fit["frame_names"][i], "shared_camera") != "shared_camera":
            continue
        p, c = frame_points(pred, i, k, Q, center, shift, args.conf_percentile, args.stride, refined_or_none(fit, args))
        if len(p) > args.points_per_frame:
            keep = rng.choice(len(p), args.points_per_frame, replace=False)
            p, c = p[keep], c[keep]
        pts.append(p)
        cols.append(c)
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    vggt_cams = field_cameras(fit, "rigid")
    refined = field_cameras(fit, "refined")
    broadcast_eye = np.median(refined.C, axis=0)
    frames = []
    for eye, R, focal, caption in orbit_path(broadcast_eye, args.video_fps, args.width):
        view = View(eye, R, focal, args.width, args.height)
        img = render_points(view, pts, cols, args.point_px)
        img = draw_field(img, view, dims)
        draw_cameras(img, view, vggt_cams, (60, 60, 230), max(1, S // 12), 18.0)
        draw_cameras(img, view, refined, (0, 215, 255), max(1, S // 12), 18.0)
        label(img, caption)
        label(img, "red: VGGT-Omega cameras   yellow: refined broadcast camera   lines: fitted NLL field", (14, args.height - 16), 0.55)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    write_h264_video(frames, out, args.video_fps)


def camera_look_at_points(cams: Cameras, smooth: int) -> np.ndarray:
    """Floor point each camera's optical axis hits, smoothed over neighbouring frames."""
    pts = []
    for i in range(len(cams.R)):
        axis = cams.R[i, 2]
        lam = -cams.C[i, 2] / axis[2] if axis[2] < -1e-6 else np.nan
        pts.append(cams.C[i] + lam * axis if np.isfinite(lam) and lam > 0 else FIELD_CENTER)
    pts = np.clip(np.asarray(pts), [-20.0, -20.0, 0.0], [FIELD_LENGTH_FT + 20.0, FIELD_WIDTH_FT + 20.0, 0.0])
    kernel = np.ones(smooth) / smooth
    padded = np.pad(pts, ((smooth // 2, smooth - 1 - smooth // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(3)], axis=1)


def render_timeline(pred: dict, fit: dict, frames_dir: Path, out: Path, args: argparse.Namespace) -> None:
    k, Q, center, shift = world_to_field_transform(fit)
    dims = fit_dims(fit)
    refined = field_cameras(fit, "refined")
    names = fit["frame_names"]
    half_w, half_h = args.width // 2, args.height // 2
    targets = camera_look_at_points(refined, smooth=max(1, int(round(args.timeline_fps))))
    top_eye = FIELD_CENTER + np.asarray([0.0, 0.0, 400.0])
    top_focal = 400.0 * min(args.width / (FIELD_LENGTH_FT + 30.0), half_h / (FIELD_WIDTH_FT + 20.0))
    top = View(top_eye, look_at(top_eye, FIELD_CENTER, np.asarray([0.0, 1.0, 0.0])), top_focal, args.width, half_h)
    frames = []
    for i, name in enumerate(names):
        frame = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        frame_status = fit.get("stage2_camera_refinement", {}).get("frame_status", {}).get(name, "")
        if frame_status == "closeup":  # another camera: the field is not placed
            panel = cv2.resize(frame, (half_w, half_h))
            label(panel, f"input frame {i}: close-up, not placed on the field")
            empty = np.full((half_h, half_w, 3), 18, np.uint8)
            frames.append(cv2.cvtColor(np.vstack([np.hstack([panel, empty]), np.full((half_h, args.width, 3), 18, np.uint8)]), cv2.COLOR_BGR2RGB))
            continue
        p, c = frame_points(pred, i, k, Q, center, shift, args.conf_percentile, 1, refined_or_none(fit, args))
        cam_panel = cv2.resize(draw_field_on_frame(frame, field_to_image_homography(refined, i), dims), (half_w, half_h))
        label(cam_panel, f"input frame {i} + fitted field" + (f"  [{frame_status}]" if frame_status and frame_status != "shared_camera" else ""))
        # oblique view from the opposite side of the broadcast camera, following the play
        side = np.sign(np.median(refined.C[:, 1]) - FIELD_CENTER[1]) or -1.0
        eye = targets[i] + np.asarray([-60.0, -side * 120.0, 85.0])
        oblique = View(eye, look_at(eye, targets[i]), 0.85 * half_w, half_w, half_h)
        obl = draw_field(render_points(oblique, p, c, 2), oblique, dims)
        label(obl, f"VGGT-Omega points, this frame (3D, {args.geometry} geometry)")
        tp = draw_field(render_points(top, p, c, 1), top, dims, 0.7)
        draw_cameras(tp, top, Cameras(K=refined.K[i : i + 1], R=refined.R[i : i + 1], C=refined.C[i : i + 1]), (0, 215, 255), 1, 25.0)
        label(tp, "same points from directly above (top-down)")
        canvas = np.vstack([np.hstack([cam_panel, obl]), tp])
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    write_h264_video(frames, out, args.timeline_fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--field-fit", required=True, help="field_plane_fit.json from fit_field_plane_landmarks.py")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--timeline-fps", type=float, default=10.0)
    parser.add_argument("--conf-percentile", type=float, default=35.0)
    parser.add_argument("--stride", type=int, default=2, help="Pixel stride when accumulating the orbit point cloud.")
    parser.add_argument("--points-per-frame", type=int, default=25000)
    parser.add_argument("--point-px", type=int, default=2)
    parser.add_argument("--geometry", choices=["refined", "raw"], default="refined", help="Place points along the refined cameras' rays (default) or show VGGT-Omega's raw geometry.")
    parser.add_argument("--skip-orbit", action="store_true")
    parser.add_argument("--skip-timeline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.predictions) as data:
        pred = {key: data[key] for key in data.files}
    fit = json.loads(Path(args.field_fit).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_orbit:
        render_orbit(pred, fit, out_dir / "vggt_scene_orbit_h264.mp4", args)
    if not args.skip_timeline:
        render_timeline(pred, fit, Path(args.frames_dir), out_dir / "vggt_scene_timeline_h264.mp4", args)
    print(json.dumps({"output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
