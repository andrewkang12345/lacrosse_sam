"""Bird's-eye video of the players on the fitted NLL field.

Every SAM3 player's floor point (the bottom of its mask) is mapped to field feet
through that frame's refined image->field homography from
fit_field_plane_landmarks.py (field_homographies.json) and drawn as a circle in
its team colour, with a short trail. The field is drawn with the arena's fitted
dimensions, and the part of the floor the camera sees in each frame is shaded.
Frames without a homography (no landmarks visible, or flagged unreliable) show
no players.

Tracking glitches are replaced by interpolation along each player's track. A
point more than --outlier-ft from the median of its own track within
±--outlier-window-seconds is an outlier. A frame where at least two points, and
at least 40% of the points that could be tested, are outliers is a bad frame
(for example SAM3 IDs shifted between neighbouring players), and all its points
are dropped. Each track is then filled by linear interpolation wherever its kept
points on either side are at most --max-gap-seconds apart. Interpolated points
have no image point and are counted in the frame header.

Teams come from classify_sam3_teams_by_torso_color.py (`teams` per detection);
detections marked `ignore` (bench, stands) are skipped, as are players whose
feet are cut off by the bottom of the image or whose floor point lands off the
field. A SAM3 JSON without teams draws every player in one colour.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import field_model as fm
from fit_field_plane_landmarks import field_canvas

TEAM_STYLE_BGR = {  # fill, outline
    "white": ((245, 245, 245), (20, 20, 20)),
    "black": ((25, 25, 25), (235, 235, 235)),
    "referee": ((150, 150, 150), (20, 20, 20)),
    "unknown": ((40, 200, 255), (20, 20, 20)),
}
STATUS_TEXT = {"shared_camera": "broadcast camera", "own_camera": "separate camera (cut)", "no_landmarks": "no landmarks: camera from neighbouring frames"}
HEADER_PX, FOOTER_PX = 44, 40
BAD_FRAME_SHARE = 0.4  # share of a frame's tested points that are outliers for it to count as a bad frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--homographies", required=True, help="field_homographies.json from fit_field_plane_landmarks.py")
    parser.add_argument("--sam3-json", required=True, help="Team JSON from classify_sam3_teams_by_torso_color.py, or a SAM3 detection JSON.")
    parser.add_argument("--instance-mask-dir", required=True, help="SAM3 player instance masks (NNNNNNNN.npz per frame).")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-json", default=None, help="Per-frame player field positions.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--trail-seconds", type=float, default=2.0, help="0 disables trails (sensible at 1 fps).")
    parser.add_argument("--outlier-ft", type=float, default=4.0, help="Distance from the track's running median that marks a point as an outlier.")
    parser.add_argument("--outlier-window-seconds", type=float, default=0.3,
                        help="Half-width of the running median; shorter than one frame disables the outlier test.")
    parser.add_argument("--max-gap-seconds", type=float, default=1.0, help="Longest span between kept points filled by interpolation; 0 disables filling.")
    parser.add_argument("--team-names", default="", help='Legend names, e.g. "black=San Diego,white=Colorado".')
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def load_masks(mask_dir: Path, frame_idx: int) -> dict[int, np.ndarray]:
    path = mask_dir / f"{frame_idx:08d}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {int(i): m.astype(bool) for i, m in zip(data["object_ids"].tolist(), data["masks"])}


def foot_point(mask: np.ndarray | None, box: list[float]) -> tuple[float, float]:
    """Bottom of the player's mask (median x of its lowest 10% of rows), else the box bottom centre."""
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        low = ys >= np.quantile(ys, 0.90)
        return float(np.median(xs[low])), float(ys.max())
    x1, _, x2, y2 = (float(v) for v in box)
    return 0.5 * (x1 + x2), y2


def apply_h(H: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    q = H @ np.array([xy[0], xy[1], 1.0])
    return q[:2] / q[2]


def draw_player(img: np.ndarray, p: tuple[int, int], team: str, label: str) -> None:
    fill, outline = TEAM_STYLE_BGR.get(team, TEAM_STYLE_BGR["unknown"])
    cv2.circle(img, p, 10, outline, -1, cv2.LINE_AA)
    cv2.circle(img, p, 8, fill, -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)


def legend(width: int, names: dict[str, str]) -> np.ndarray:
    bar = np.full((FOOTER_PX, width, 3), (28, 74, 54), np.uint8)  # the field canvas surround
    x = 16
    for team in ("black", "white", "referee"):
        draw_player(bar, (x + 10, FOOTER_PX // 2), team, "")
        text = names.get(team, team)
        cv2.putText(bar, text, (x + 26, FOOTER_PX // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        x += 46 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.rectangle(bar, (x, 12), (x + 26, FOOTER_PX - 12), (120, 170, 150), -1)
    cv2.putText(bar, "floor in view this frame", (x + 34, FOOTER_PX // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return bar


def clean_tracks(raw: list[list[dict]], has_fit: list[bool], window: int, outlier_ft: float, max_gap: int) -> tuple[list[list[dict]], dict]:
    """Drop outlier points and bad frames, then fill each track by linear interpolation (windows and gaps in frames)."""
    tracks: dict[int, dict[int, dict]] = {}
    for idx, players in enumerate(raw):
        for item in players:
            tracks.setdefault(item["object_id"], {})[idx] = item
    tested, outliers, dropped = Counter(), Counter(), set()
    if window >= 1:
        for oid, samples in tracks.items():
            frames = sorted(samples)
            for f in frames:
                near = [samples[g]["field_point_ft"] for g in frames if abs(g - f) <= window]
                if len(near) < 3:
                    continue
                tested[f] += 1
                if np.linalg.norm(np.subtract(samples[f]["field_point_ft"], np.median(near, axis=0))) > outlier_ft:
                    outliers[f] += 1
                    dropped.add((f, oid))
    bad = {f for f in tested if outliers[f] >= 2 and outliers[f] >= BAD_FRAME_SHARE * tested[f]}

    cleaned: list[list[dict]] = [[] for _ in raw]
    kept_count = interpolated = 0
    for oid, samples in tracks.items():
        kept = {f: s for f, s in samples.items() if f not in bad and (f, oid) not in dropped}
        if not kept:
            continue
        team = Counter(s["team"] for s in kept.values()).most_common(1)[0][0]
        frames = sorted(kept)
        for f in frames:
            cleaned[f].append({**kept[f], "interpolated": False})
        kept_count += len(frames)
        for a, b in zip(frames, frames[1:]):
            if b - a < 2 or b - a > max_gap:
                continue
            pa, pb = np.asarray(kept[a]["field_point_ft"]), np.asarray(kept[b]["field_point_ft"])
            for f in range(a + 1, b):
                if has_fit[f]:
                    t = (f - a) / (b - a)
                    cleaned[f].append({"object_id": oid, "team": team, "image_point": None,
                                       "field_point_ft": ((1 - t) * pa + t * pb).tolist(), "interpolated": True})
                    interpolated += 1
    for players in cleaned:
        players.sort(key=lambda p: p["object_id"])
    stats = {"bad_frames": sorted(bad), "dropped_points": sum(len(s) for s in tracks.values()) - kept_count, "interpolated_points": interpolated}
    return cleaned, stats


def main() -> None:
    args = parse_args()
    hom = json.loads(Path(args.homographies).read_text())
    dims = fm.FieldDims(**hom["field_dims"])
    by_name = {h["frame_name"]: (np.asarray(h["matrix_image_to_world"], float), h.get("status", "")) for h in hom["homographies"]}
    sam3 = json.loads(Path(args.sam3_json).read_text())
    records = {int(r.get("frame", i)): r for i, r in enumerate(sam3["frames"])}
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    mask_dir = Path(args.instance_mask_dir)
    names = dict(item.split("=", 1) for item in args.team_names.split(",") if "=" in item)
    x_lo, x_hi = 100.0 - dims.half_length, 100.0 + dims.half_length

    field_h = args.height - HEADER_PX - FOOTER_PX
    base, to_px = field_canvas(dims, args.width, field_h, margin=36)
    origin, unit = to_px(np.array([[0.0, 0.0]]))[0], to_px(np.array([[1.0, 1.0]]))[0]
    scale_x, scale_y = unit - origin
    inside = np.zeros((field_h, args.width), np.uint8)
    cv2.fillPoly(inside, [np.round(to_px(fm.outline_polyline(dims))).astype(np.int32)], 1)
    rows, cols = np.nonzero(inside)
    floor_world = np.stack([(cols + 0.5 - origin[0]) / scale_x, (rows + 0.5 - origin[1]) / scale_y, np.ones(len(rows))])
    footer = legend(args.width, names)
    trail_frames = int(round(args.trail_seconds * args.fps))

    sizes, fits, raw = [], [], []
    for idx, path in enumerate(frame_paths):
        h, w = cv2.imread(str(path)).shape[:2]
        record = records.get(idx, {"object_ids": [], "boxes": []})
        players = []
        fit = by_name.get(path.name)
        if fit is not None:
            H = fit[0]
            masks = load_masks(mask_dir, idx)
            teams = record.get("teams") or ["unknown"] * len(record["object_ids"])
            for oid, box, team in zip(record["object_ids"], record["boxes"], teams):
                if team == "ignore" or float(box[3]) >= h - 2:
                    continue  # not on the floor, or feet cut off by the bottom of the image
                image_xy = foot_point(masks.get(int(oid)), box)
                x, y = apply_h(H, image_xy)
                if not (x_lo - 3 <= x <= x_hi + 3 and -3 <= y <= dims.width + 3):
                    continue  # bench, stands, or a player hidden behind the near boards
                players.append({"object_id": int(oid), "team": team, "image_point": list(image_xy), "field_point_ft": [float(x), float(y)]})
        sizes.append((h, w))
        fits.append(fit)
        raw.append(players)
    cleaned, stats = clean_tracks(raw, [f is not None for f in fits], int(round(args.outlier_window_seconds * args.fps)),
                                  args.outlier_ft, int(round(args.max_gap_seconds * args.fps)))
    bad_frames = set(stats["bad_frames"])

    history: dict[int, list[tuple[int, float, float, str]]] = {}
    video, positions = [], []
    for idx, path in enumerate(frame_paths):
        (h, w), fit, players = sizes[idx], fits[idx], cleaned[idx]
        canvas = base.copy()
        if fit is not None:
            H, status = fit
            G = np.linalg.inv(H)
            # sign of the projective scale for floor points in front of the camera
            front = np.sign((G @ np.append(apply_h(H, (0.5 * w, 0.9 * h)), 1.0))[2])
            uvw = G @ floor_world
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
            seen = (uvw[2] * front > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            canvas[rows[seen], cols[seen]] = (0.7 * canvas[rows[seen], cols[seen]] + 0.3 * 255).astype(np.uint8)
            for item in players:
                history.setdefault(item["object_id"], []).append((idx, *item["field_point_ft"], item["team"]))
            filled = sum(item["interpolated"] for item in players)
            header = f"{path.name}   t = {idx / args.fps:.1f} s   {STATUS_TEXT.get(status, status)}   {len(players)} players"
            if filled:
                header += f" ({filled} interpolated)"
        else:
            header = f"{path.name}   t = {idx / args.fps:.1f} s   no field fit for this frame"

        for oid, samples in history.items():
            recent = [(x, y) for f, x, y, _ in samples if idx - trail_frames <= f <= idx]
            if trail_frames and len(recent) >= 2 and samples[-1][0] == idx:
                fill = TEAM_STYLE_BGR.get(samples[-1][3], TEAM_STYLE_BGR["unknown"])[0]
                cv2.polylines(canvas, [np.round(to_px(np.asarray(recent))).astype(np.int32)], False, fill, 2, cv2.LINE_AA)
        for item in players:
            p = tuple(int(v) for v in np.round(to_px(np.asarray([item["field_point_ft"]]))[0]))
            draw_player(canvas, p, item["team"], str(item["object_id"]))

        top = np.full((HEADER_PX, args.width, 3), 18, np.uint8)
        cv2.putText(top, header, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
        video.append(np.vstack([top, canvas, footer]))
        positions.append({"frame": idx, "frame_name": path.name, "has_field_fit": fit is not None, "bad_frame": idx in bad_frames, "players": players})

    out = Path(args.output_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        out, fps=args.fps, codec="libx264", ffmpeg_log_level="error", macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame in video:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({
            "schema": "field_birds_eye_players_v1", "homographies": args.homographies, "sam3_json": args.sam3_json,
            "field_dims": dims.to_dict(), "fps": args.fps,
            "cleaning": {"outlier_ft": args.outlier_ft, "outlier_window_seconds": args.outlier_window_seconds,
                         "max_gap_seconds": args.max_gap_seconds, **stats},
            "frames": positions,
        }) + "\n")
    print(json.dumps({"video": str(out), "frames": len(video), "frames_with_fit": sum(p["has_field_fit"] for p in positions),
                      "player_points": sum(len(p["players"]) for p in positions), **stats}))


if __name__ == "__main__":
    main()
