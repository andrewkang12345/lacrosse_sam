"""Labeled field-landmark masks from painted lines, with a visual review step.

Text- and click-prompted SAM2/SAM3 masks are a poor landmark source for the
field fit: on the Week 1 clip the "yellow outline" prompt also segments the
centre logo and some white lines, the "white lines" prompt misses most visible
lines, and neither says which line a pixel belongs to. This script detects the
floor paint by colour and labels every piece explicitly:

1. Floor: green turf (and purple end-zone turf) by colour.
2. Paint: thin, bright, low-saturation structures on the floor (white top-hat),
   players removed with their SAM3 masks. Each connected piece is a candidate
   with a stable id (sorted by size).
3. Board base: the first turf row from the top in every column, i.e. where the
   far boards meet the floor (skipped where a player stands in front of it,
   smoothed across columns). The near boards are not used: the camera looks
   over them, so the visible turf ends at their top edge, about 4 ft up, and
   logos or floor ads reaching the image bottom look like the same edge.
4. Labels: every candidate gets the landmark whose projected curve it lies on
   under an existing fit (--fit), if that is unambiguous. A review file
   (--review) overrides labels per frame and candidate id; the review sheets
   written to <output>/review/ show the ids and labels so a person (or a
   vision model) can check every frame and write those overrides.
   Straight-line labels are trimmed to the pixels on one straight line.

Close-up shots (a player at least --closeup-height of the image tall, sustained for
--closeup-min-frames) come from other cameras. They get no landmarks and are listed
under `closeup_frames` in labels.json, so the field fit leaves them out.

Writes <output>/masks/NNNNNNNN.npz (object_ids, masks; numbered by position in
the sorted frames directory like the SAM outputs) and <output>/labels.json,
which fit_field_plane_landmarks.py reads through --paint-masks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

import field_model as fm

OBJECT_IDS = {"midfield_line": 1, "left_restraining_line": 2, "right_restraining_line": 3, "left_crease_lines": 4, "right_crease_lines": 5, "board_base": 6}
PAINT_LABELS = ("midfield_line", "left_restraining_line", "right_restraining_line", "left_crease_lines", "right_crease_lines")
STRAIGHT = {"midfield_line", "left_restraining_line", "right_restraining_line"}
COLORS_BGR = {
    "midfield_line": (40, 40, 255),
    "left_restraining_line": (255, 120, 0),
    "right_restraining_line": (0, 230, 0),
    "left_crease_lines": (255, 0, 255),
    "right_crease_lines": (0, 150, 255),
    "unlabeled": (170, 170, 170),
    "reject": (70, 70, 70),
    "board_base": (0, 255, 255),
}
SHORT = {"midfield_line": "mid", "left_restraining_line": "Lres", "right_restraining_line": "Rres", "left_crease_lines": "Lcr", "right_crease_lines": "Rcr", "unlabeled": "?", "reject": "x"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


FLOOR_COLOURS_HSV = {  # OpenCV HSV ranges: (h_min, h_max, s_min, v_min)
    "green": (38, 95, 45, 35),
    "purple": (105, 165, 20, 50),  # San Diego end zones; overlaps blue jerseys, so opt-in
}


def _main_components(mask: np.ndarray) -> np.ndarray:
    """The playing floor only: turf-coloured seats or banners elsewhere form separate,
    smaller blobs. Components are taken after bridging painted lines, which split the turf."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # paint between turf colours
    bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    if count <= 1:
        return mask > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.concatenate([[False], areas >= 0.2 * areas.max()])
    return keep[labels] & (mask > 0)


def floor_mask(bgr: np.ndarray, colours: tuple[str, ...] = ("green",)) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., k].astype(np.int32) for k in range(3))

    def colour(name: str) -> np.ndarray:
        h0, h1, s0, v0 = FLOOR_COLOURS_HSV[name]
        return cv2.morphologyEx(((H >= h0) & (H <= h1) & (S >= s0) & (V >= v0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    floor = _main_components(colour("green"))
    extras = [c for c in colours if c != "green"]
    if extras:
        # other turf colours only where a region of that colour touches the green floor
        # directly (purple end zones do; purple seats behind the boards do not)
        near_floor = cv2.dilate(floor.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        grown = floor.copy()
        for name in extras:
            extra = colour(name)
            _, labels, _, _ = cv2.connectedComponentsWithStats(extra, connectivity=8)
            touching = np.unique(labels[near_floor & (extra > 0)])
            grown |= np.isin(labels, touching[touching > 0])
        floor = _main_components(grown)
    return floor


def field_region(floor: np.ndarray) -> np.ndarray:
    """The floor plus everything enclosed by it (players, crease, logos)."""
    closed = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    h, w = closed.shape
    flood = closed.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 8):  # flood the background from the image border
        for y in (0, h - 1):
            if flood[y, x] == 0:
                cv2.floodFill(flood, pad, (x, y), 2)
    for y in range(0, h, 8):
        for x in (0, w - 1):
            if flood[y, x] == 0:
                cv2.floodFill(flood, pad, (x, y), 2)
    return flood != 2


def paint_components(bgr: np.ndarray, turf: np.ndarray, players: np.ndarray | None, max_components: int) -> list[np.ndarray]:
    """Thin white paint pieces on the floor, largest first, as (N,2) pixel arrays."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]
    tophat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    paint = (tophat >= 35) & (hsv[..., 1] <= 90) & (V >= 120)
    paint &= field_region(turf)
    if players is not None:
        paint &= ~(cv2.dilate(players.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
    paint = cv2.morphologyEx(paint.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
    pieces = []
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        if np.hypot(w, h) < 30:
            continue
        sub = labels[y : y + h, x : x + w] == k
        skeleton_len = max(int(skeletonize(sub).sum()), 1)
        if skeleton_len < 30 or area / skeleton_len > 25.0:  # blobs and lettering, not lines (zoomed lines are ~20 px wide)
            continue
        ys, xs = np.nonzero(sub)
        pieces.append(np.column_stack([xs + x, ys + y]).astype(np.int32))
    pieces.sort(key=lambda p: -len(p))
    return pieces[:max_components]


def board_base_pixels(turf: np.ndarray, players: np.ndarray | None, bgr: np.ndarray | None = None, min_run: int = 25) -> np.ndarray:
    """Per column, the first turf row from the top that starts a solid run of turf
    (mostly turf for 40 rows below) with non-turf above it: where the far boards
    meet the floor. Scattered turf-coloured pixels in the crowd never qualify.

    With `bgr`, columns where a yellow band above the turf is much thicker than the
    frame's thinnest yellow (the kick plate) are dropped: there the turf ends at a
    painted line-change area in front of a bench (NLL rule 1.4: 3 ft deep), not at
    the boards."""
    h, w = turf.shape
    t = turf.astype(np.uint8)
    run = cv2.erode(t, np.ones((min_run, 1), np.uint8), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    density = cv2.blur(t.astype(np.float32), (1, 40), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    run &= (density >= 0.85).astype(np.uint8)
    cols = np.nonzero(run.any(axis=0))[0]
    rows = np.argmax(run[:, cols] > 0, axis=0)
    keep = (rows > 3) & (rows < h - min_run - 3)
    above = np.asarray([t[max(r - 10, 0) : r, c].mean() if r > 0 else 1.0 for r, c in zip(rows, cols)])
    keep &= above < 0.3
    if players is not None:
        near_player = cv2.dilate(players.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        keep &= ~near_player[np.clip(rows, 0, h - 1), cols]
    if bgr is not None and keep.any():
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
        yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 110)
        runs = np.zeros(len(cols), np.int32)
        for j in np.nonzero(keep)[0]:
            r, c = rows[j], cols[j]
            col = yellow[max(r - 120, 0) : max(r - 1, 0), c][::-1]  # upwards from just above the edge
            runs[j] = len(col) if col.all() else int(np.argmin(col))
        thin = runs[keep & (runs > 0)]
        if len(thin) >= 20:
            keep &= runs <= max(12, 2.0 * np.percentile(thin, 25))
    return smooth_edge(cols[keep], rows[keep])


def smooth_edge(cols: np.ndarray, rows: np.ndarray, window: int = 31, tol: float = 3.0) -> np.ndarray:
    """Drop edge points off the local running median (glass posts, sticks, logos on the
    boards). The board line is smooth across neighbouring columns."""
    if len(cols) < 5:
        return np.column_stack([cols, rows]).astype(np.int32)
    order = np.argsort(cols)
    cols, rows = cols[order], rows[order].astype(np.float64)
    med = np.asarray([np.median(rows[(cols >= c - window // 2) & (cols <= c + window // 2)]) for c in cols])
    keep = np.abs(rows - med) <= tol
    return np.column_stack([cols[keep], rows[keep]]).astype(np.int32)


def straight_inliers(pixels: np.ndarray, tol: float | None = None, iterations: int = 200) -> np.ndarray:
    """Pixels on the dominant straight line of a component (drops attached logo bits).
    The tolerance follows the line's width, which grows when the camera zooms in."""
    if len(pixels) < 20:
        return pixels
    rng = np.random.default_rng(0)
    p = pixels.astype(np.float64)
    if tol is None:
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        width = len(p) / max(float(np.ptp(centred @ vt[0])), 1.0)
        tol = max(4.0, 0.7 * width)
    best = None
    for _ in range(iterations):
        a, b = p[rng.choice(len(p), 2, replace=False)]
        d = b - a
        n = np.linalg.norm(d)
        if n < 10:
            continue
        normal = np.asarray([-d[1], d[0]]) / n
        inl = np.abs((p - a) @ normal) <= tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return pixels if best is None else pixels[best]


def load_union_mask(mask_dir: Path | None, frame_number: int, shape: tuple[int, int]) -> np.ndarray | None:
    if mask_dir is None:
        return None
    path = mask_dir / f"{frame_number:08d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        masks = data["masks"]
        if len(masks) == 0:
            return np.zeros(shape, dtype=bool)
        m = masks.astype(bool).any(axis=0)
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m


# ---------------------------------------------------------------------------
# Auto-labels from an existing fit
# ---------------------------------------------------------------------------


class FitProjector:
    def __init__(self, fit_path: Path):
        fit = json.loads(fit_path.read_text())
        cams = fit["cameras_field_ft"]["refined"]
        self.K, self.R, self.C = (np.asarray(cams[k]) for k in ("K", "R", "C"))
        self.index = {name: i for i, name in enumerate(fit["frame_names"])}
        self.status = fit.get("stage2_camera_refinement", {}).get("frame_status", {})
        self.dims = fm.FieldDims(**{k: v for k, v in fit.get("field_dims", {}).items() if k in fm.FieldDims.__dataclass_fields__})

    def distance_maps(self, name: str, shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        """Per landmark: image distance map to its projected curve, and (straight
        lines) the projected line direction."""
        i = self.index.get(name)
        if i is None or self.status.get(name) == "unreliable":
            return None
        H = self.K[i] @ np.column_stack([self.R[i][:, 0], self.R[i][:, 1], -self.R[i] @ self.C[i]])
        h, w = shape
        maps, directions = {}, {}
        for label in PAINT_LABELS:
            canvas = np.full(shape, 255, np.uint8)
            for poly in [fm.curve(label, self.dims, 2.0)] if label in STRAIGHT else fm.crease_polylines(label.split("_")[0], self.dims, 200):
                p = np.column_stack([poly, np.ones(len(poly))]) @ H.T
                ok = p[:, 2] > 0
                px = p[ok, :2] / p[ok, 2:3]
                inside = np.all(np.abs(px) < 4 * max(h, w), axis=1)
                if inside.sum() >= 2:
                    seg = px[inside]
                    cv2.polylines(canvas, [np.round(seg).astype(np.int32)], False, 0, 1)
                    if label in STRAIGHT:
                        d = seg[-1] - seg[0]
                        directions[label] = d / max(np.linalg.norm(d), 1e-9)
            maps[label] = cv2.distanceTransform(canvas, cv2.DIST_L2, 3) if (canvas == 0).any() else np.full(shape, 1e4, np.float32)
        return maps, directions


def line_shape(pixels: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Principal direction, length and fraction of pixels on one straight line."""
    p = pixels.astype(np.float64)
    centred = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = centred @ vt[0]
    return vt[0], float(np.ptp(along)), len(straight_inliers(pixels)) / len(pixels)


def auto_label(pixels: np.ndarray, projected: tuple[dict, dict] | None, max_px: float) -> tuple[str, float]:
    """Nearest projected landmark, if close and unambiguous.

    Crease labels use a tighter distance limit because the goal frame and net sit
    right on the crease. A straight-line label needs a straight piece running in the
    projected line's direction; logo artwork next to the midfield line is not. When
    a line merged with neighbouring paint into one piece (a midfield stub touching
    the logo), its dominant straight part is judged on its own, and only that part
    goes into the mask (see straight_inliers)."""
    if projected is None:
        return "unlabeled", float("nan")
    maps, directions = projected

    def score(px: np.ndarray, label: str) -> float:
        return float(np.median(maps[label][px[:, 1], px[:, 0]]))

    scores = sorted((score(pixels, label), label) for label in maps)
    (best, label), (second, _) = scores[0], scores[1]
    if label not in STRAIGHT and best <= max_px / 3.0 and best <= 0.6 * second:
        return label, best
    core = straight_inliers(pixels)
    if len(core) < 20:
        return "unlabeled", best
    direction, length, straight = line_shape(core)
    if length < 25.0 or straight < 0.75:
        return "unlabeled", best
    line_scores = sorted((score(core, lab), lab) for lab in maps)
    (best_line, lab), (second_line, _) = line_scores[0], line_scores[1]
    aligned = lab in directions and abs(float(direction @ directions[lab])) >= np.cos(np.deg2rad(12.0))
    if lab in STRAIGHT and aligned and best_line <= max_px and best_line <= 0.6 * second_line:
        return lab, best_line
    return "unlabeled", best


# ---------------------------------------------------------------------------
# Review sheets
# ---------------------------------------------------------------------------


def draw_review_tile(bgr: np.ndarray, name: str, pieces: list[np.ndarray], labels: list[str], base: np.ndarray, base_ok: bool) -> np.ndarray:
    """Labeled pieces in their landmark colour with a large id; unlabeled straight
    pieces (possible missed lines) in grey with an id; other unlabeled paint (logo
    artwork, text) faint and without an id."""
    vis = (bgr * 0.7).astype(np.uint8)
    for x, y in base[::2] if base_ok else []:
        cv2.circle(vis, (int(x), int(y)), 2, COLORS_BGR["board_base"], -1)
    tags = []
    for k, (px, label) in enumerate(zip(pieces, labels), start=1):
        labeled = label not in ("unlabeled", "reject")
        line_like = False
        if not labeled and len(px) >= 20:
            _, length, straight = line_shape(px)
            line_like = length >= 40 and straight >= 0.8
        mask = np.zeros(bgr.shape[:2], np.uint8)
        mask[px[:, 1], px[:, 0]] = 1
        if labeled or line_like:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
            vis[mask > 0] = COLORS_BGR[label]
            tags.append((k, px, label, labeled))
        else:
            vis[mask > 0] = (0.5 * vis[mask > 0] + 0.5 * np.asarray(COLORS_BGR["reject"])).astype(np.uint8)
    for k, px, label, labeled in tags:
        c = px[np.argmin(np.linalg.norm(px - px.mean(axis=0), axis=1))]
        text = f"{k}{SHORT[label]}"
        scale, thick = (1.6, 3) if labeled else (1.3, 2)
        org = (int(np.clip(c[0] + 8, 0, bgr.shape[1] - 140)), int(np.clip(c[1] - 8, 80, bgr.shape[0] - 6)))
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 5, cv2.LINE_AA)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS_BGR[label], thick, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (bgr.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(vis, name, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)
    return vis


def write_sheets(tiles: list[np.ndarray], out_dir: Path, cols: int, rows: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()
    paths = []
    per = cols * rows
    for s in range(0, len(tiles), per):
        chunk = [cv2.resize(t, (640, 360), interpolation=cv2.INTER_AREA) for t in tiles[s : s + per]]
        while len(chunk) < per:
            chunk.append(np.zeros_like(chunk[0]))
        sheet = np.vstack([np.hstack(chunk[r * cols : (r + 1) * cols]) for r in range(rows)])
        path = out_dir / f"sheet_{s // per:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(str(path))
    return paths


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--predictions", default=None, help="Use the frame list of this VGGT-Omega npz (default: every frame in --frames-dir).")
    parser.add_argument("--player-mask-dir", default=None, help="SAM3 player instance masks; players are removed from paint and board-base detections.")
    parser.add_argument("--fit", default=None, help="field_plane_fit.json used to propose labels.")
    parser.add_argument(
        "--review", default=None, help='JSON {"frame_name": {"<id>": "<label>|reject", "board_base": "reject"}} of checked labels (keys starting with "_" are comments).'
    )
    parser.add_argument("--closeup-height", type=float, default=0.45, help="A player this share of the image height marks a close-up frame.")
    parser.add_argument("--closeup-min-frames", type=int, default=5, help="Close-up frames needed in a row (gaps up to 4 frames) to count as a close-up shot.")
    parser.add_argument("--checked-masks", default=None, help="masks/ of a checked run on this clip's keyframes. Keyframes with at least two painted lines are copied; the others are detected and labeled from --fit.")
    parser.add_argument("--keyframe-offset", type=int, default=0, help="Frame of this clip that is the first frame of --checked-masks.")
    parser.add_argument("--keyframe-stride", type=int, default=1, help="Frames of this clip per frame of --checked-masks.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--floor-colours", default="green", help=f"Comma-separated turf colours from {sorted(FLOOR_COLOURS_HSV)}.")
    parser.add_argument("--max-components", type=int, default=30)
    parser.add_argument("--max-auto-label-px", type=float, default=45.0)
    parser.add_argument("--sheet-cols", type=int, default=3)
    parser.add_argument("--sheet-rows", type=int, default=3)
    return parser.parse_args()


def closeup_frames(mask_dir: Path | None, numbers: list[int], share: float, min_frames: int, max_gap: int = 4) -> set[int]:
    """Frames of sustained shots in which a player fills a large part of the image height."""
    if mask_dir is None:
        return set()
    tall = []
    for number in numbers:
        path = mask_dir / f"{number:08d}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            masks = data["masks"].astype(bool)
        if len(masks) == 0:
            continue
        rows = masks.any(axis=2)
        spans = [np.flatnonzero(r)[[0, -1]] for r in rows if r.any()]
        if spans and max(int(b - a + 1) for a, b in spans) >= share * masks.shape[1]:
            tall.append(number)
    runs: list[list[int]] = []
    for number in tall:
        if runs and number - runs[-1][-1] <= max_gap + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return {n for run in runs if len(run) >= min_frames for n in range(run[0], run[-1] + 1)}


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    all_frames = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    if args.predictions:
        with np.load(args.predictions) as data:
            names = [str(n) for n in data["frame_names"]]
    else:
        names = all_frames
    projector = FitProjector(Path(args.fit)) if args.fit else None
    review = json.loads(Path(args.review).read_text()) if args.review else {}
    out_dir = Path(args.output_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    player_dir = Path(args.player_mask_dir) if args.player_mask_dir else None

    closeups = closeup_frames(player_dir, [all_frames.index(n) for n in names], args.closeup_height, args.closeup_min_frames)
    report, tiles = {}, []
    counts = {label: 0 for label in PAINT_LABELS + ("unlabeled", "reject")}
    for name in names:
        number = all_frames.index(name)
        if number in closeups:
            with Image.open(frames_dir / name) as first:
                size = first.size
            np.savez_compressed(out_dir / "masks" / f"{number:08d}.npz", object_ids=np.zeros(0, np.int32), masks=np.zeros((0, size[1], size[0]), bool))
            report[name] = {"candidates": [], "board_base_columns": 0, "reviewed": False, "closeup": True}
            continue
        if args.checked_masks and number >= args.keyframe_offset and (number - args.keyframe_offset) % args.keyframe_stride == 0:
            checked = Path(args.checked_masks) / f"{(number - args.keyframe_offset) // args.keyframe_stride:08d}.npz"
            if checked.exists():
                with np.load(checked) as data:
                    lines = sum(1 for oid, m in zip(data["object_ids"], data["masks"]) if int(oid) != OBJECT_IDS["board_base"] and int(m.sum()) >= 50)
                if lines >= 2:
                    shutil.copyfile(checked, out_dir / "masks" / f"{number:08d}.npz")
                    report[name] = {"candidates": [], "board_base_columns": 0, "reviewed": True, "checked_keyframe": True}
                    continue
        bgr = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        shape = bgr.shape[:2]
        players = load_union_mask(player_dir, number, shape)
        turf = floor_mask(bgr, tuple(args.floor_colours.split(",")))
        pieces = paint_components(bgr, turf, players, args.max_components)
        base = board_base_pixels(turf, players, bgr)
        if pieces and len(base):
            # where a crease (or other floor paint) hides the board line, the turf edge
            # is the paint's edge, not the boards: drop points next to paint
            near_paint = np.zeros(shape, np.uint8)
            for px in pieces:
                near_paint[px[:, 1], px[:, 0]] = 1
            near_paint = cv2.dilate(near_paint, np.ones((17, 17), np.uint8)) > 0
            base = base[~near_paint[base[:, 1], base[:, 0]]]
        maps = projector.distance_maps(name, shape) if projector else None
        overrides = review.get(name, {})
        entries, labels = [], []
        for k, px in enumerate(pieces, start=1):
            auto, auto_px = auto_label(px, maps, args.max_auto_label_px)
            label = overrides.get(str(k), auto)
            if label not in PAINT_LABELS + ("unlabeled", "reject"):
                raise ValueError(f"{name}: unknown label {label!r} for candidate {k}")
            labels.append(label)
            counts[label] += 1
            entries.append({"id": k, "pixels": int(len(px)), "centroid": px.mean(axis=0).round(1).tolist(), "auto": auto, "auto_px": round(auto_px, 1), "label": label})
        base_ok = overrides.get("board_base", "ok") != "reject" and len(base) >= 20
        object_ids, masks = [], []
        for label in PAINT_LABELS:
            parts = [px for px, lab in zip(pieces, labels) if lab == label]
            if not parts:
                continue
            m = np.zeros(shape, bool)
            for px in parts:
                px = straight_inliers(px) if label in STRAIGHT else px
                m[px[:, 1], px[:, 0]] = True
            object_ids.append(OBJECT_IDS[label])
            masks.append(m)
        if base_ok:
            m = np.zeros(shape, bool)
            m[base[:, 1], base[:, 0]] = True
            object_ids.append(OBJECT_IDS["board_base"])
            masks.append(m)
        np.savez_compressed(out_dir / "masks" / f"{number:08d}.npz", object_ids=np.asarray(object_ids, np.int32), masks=np.asarray(masks, bool).reshape(-1, *shape))
        report[name] = {
            "candidates": entries,
            "board_base_columns": int(len(base)) if base_ok else 0,
            "reviewed": name in review,
        }
        tiles.append(draw_review_tile(bgr, name, pieces, labels, base, base_ok))

    sheets = write_sheets(tiles, out_dir / "review", args.sheet_cols, args.sheet_rows)
    (out_dir / "labels.json").write_text(
        json.dumps(
            {
                "schema": "field_paint_masks_v1",
                "frames_dir": args.frames_dir,
                "fit": args.fit,
                "review": args.review,
                "object_ids": OBJECT_IDS,
                "closeup_frames": [n for n in names if all_frames.index(n) in closeups],
                "frames": report,
            },
            indent=1,
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(out_dir), "frames": len(names), "closeup_frames": len(closeups), "candidate_labels": counts, "review_sheets": len(sheets)}, indent=2))


if __name__ == "__main__":
    main()
