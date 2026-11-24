"""Bird's-eye video of the players on the fitted NLL field.

Every SAM3 player's floor point is mapped to field feet through that frame's refined
image->field homography from fit_field_plane_landmarks.py (field_homographies.json) and
drawn as a circle in its team colour, with a short trail. The floor point is the lowest
row of the player's mask, at the median column of the mask's lower half: the legs and
hips, so it does not swing from foot to foot with each stride. The field is drawn with
the arena's fitted dimensions, and the part of the floor the camera sees in each frame
is shaded. Only frames on the broadcast camera are measured. Close-ups and other cuts
(fit status closeup, own_camera, no_landmarks or unreliable) give no positions, because
a few landmarks seen by a different camera do not pin down its position.

Positions come from field_player_tracks.py, which turns the floor points into smoothed
player tracks:
- each floor point is weighted by its uncertainty in feet, which is larger far from the
  camera and when the player is partly hidden (feet behind another player, or cut by
  the side of the image);
- outliers and bad frames (SAM3 IDs shifted between neighbouring players) are dropped;
- tracks are split where SAM3 moved an ID to a different player, and the pieces are
  rejoined by team and motion;
- gaps up to --max-gap-seconds are filled by the smoother, also through frames where
  the camera was lost briefly; longer shots show no players.
With --jersey-reads (the crop reads of jersey_number_sheets.py), a dot is labelled with
its jersey number when combine_jersey_numbers.py gives its track one: the track's read
segments agree on one number and no other track of that team has it at the same time.
Dots of unread or conflicting tracks have no label.

Teams come from classify_sam3_teams_by_torso_color.py (`teams` per detection);
detections marked `ignore` (bench, stands) are skipped, as are players whose
feet are cut off by the bottom of the image or whose floor point lands off the
field, and masks under 200 pixels (SAM3 remnants). A SAM3 JSON without teams
draws every player in one colour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import field_model as fm
import field_player_tracks as fpt
from combine_jersey_numbers import jersey_numbers
from fit_field_plane_landmarks import field_canvas

TEAM_STYLE_BGR = {  # fill, outline
    "white": ((245, 245, 245), (20, 20, 20)),
    "black": ((25, 25, 25), (235, 235, 235)),
    "referee": ((150, 150, 150), (20, 20, 20)),
    "unknown": ((40, 200, 255), (20, 20, 20)),
}
STATUS_TEXT = {"shared_camera": "broadcast camera"}
BROADCAST_STATUS = {"shared_camera", "rigid"}
HEADER_PX, FOOTER_PX = 44, 40


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
    parser.add_argument("--max-speed-ft-per-s", type=float, default=30.0, help="Fastest a player runs: track pieces further apart than this are not joined.")
    parser.add_argument("--position-noise-ft", type=float, default=3.0, help="Floor-point noise allowed on top of --max-speed-ft-per-s.")
    parser.add_argument("--max-gap-seconds", type=float, default=1.0, help="Longest gap a track is filled and joined across.")
    parser.add_argument("--team-names", default="", help='Legend names, e.g. "black=San Diego,white=Colorado".')
    parser.add_argument("--jersey-reads", default=None, help="Crop reads JSON (data/jersey_number_review/<clip>_reads.json); labels dots with jersey numbers.")
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
    """Lowest row of the mask, at the median column of its lower half; else the box bottom centre."""
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return float(np.median(xs[ys >= np.median(ys)])), float(ys.max())
    x1, _, x2, y2 = (float(v) for v in box)
    return 0.5 * (x1 + x2), y2


def apply_h(H: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    q = H @ np.array([xy[0], xy[1], 1.0])
    return q[:2] / q[2]


def h_jacobian(H: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    """Field feet per image pixel at xy; columns are the image x and y directions."""
    q = H @ np.array([xy[0], xy[1], 1.0])
    p = q[:2] / q[2]
    return (H[:2, :2] - np.outer(p, H[2, :2])) / q[2]


def draw_player(img: np.ndarray, p: tuple[int, int], team: str, label: str) -> None:
    fill, outline = TEAM_STYLE_BGR.get(team, TEAM_STYLE_BGR["unknown"])
    cv2.circle(img, p, 10, outline, -1, cv2.LINE_AA)
    cv2.circle(img, p, 8, fill, -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


def legend(width: int, names: dict[str, str], floor_in_view: bool = True) -> np.ndarray:
    bar = np.full((FOOTER_PX, width, 3), (28, 74, 54), np.uint8)  # the field canvas surround
    x = 16
    for team in ("black", "white", "referee"):
        draw_player(bar, (x + 10, FOOTER_PX // 2), team, "")
        text = names.get(team, team)
        cv2.putText(bar, text, (x + 26, FOOTER_PX // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        x += 46 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    if floor_in_view:
        cv2.rectangle(bar, (x, 12), (x + 26, FOOTER_PX - 12), (120, 170, 150), -1)
        cv2.putText(bar, "floor in view this frame", (x + 34, FOOTER_PX // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return bar


def floor_points(path: Path, idx: int, record: dict, mask_dir: Path, H: np.ndarray, dims: fm.FieldDims) -> list[fpt.FootPoint]:
    h = cv2.imread(str(path)).shape[0]
    masks = load_masks(mask_dir, idx)
    hidden = fpt.partly_hidden(masks)
    teams = record.get("teams") or ["unknown"] * len(record["object_ids"])
    x_lo, x_hi = 100.0 - dims.half_length, 100.0 + dims.half_length
    points = []
    for oid, box, team in zip(record["object_ids"], record["boxes"], teams):
        mask = masks.get(int(oid))
        if team == "ignore" or float(box[3]) >= h - 2 or (mask is not None and mask.sum() < fpt.MIN_MASK_PX):
            continue  # not on the floor, feet cut off by the bottom of the image, or a SAM3 remnant
        image_xy = foot_point(mask, box)
        x, y = apply_h(H, image_xy)
        if not (x_lo - 3 <= x <= x_hi + 3 and -3 <= y <= dims.width + 3):
            continue  # bench, stands, or a player hidden behind the near boards
        points.append(fpt.FootPoint(idx, int(oid), team, image_xy, np.array([x, y]), h_jacobian(H, image_xy), int(oid) in hidden))
    return points


def main() -> None:
    args = parse_args()
    hom = json.loads(Path(args.homographies).read_text())
    dims = fm.FieldDims(**hom["field_dims"])
    by_name = {h["frame_name"]: (np.asarray(h["matrix_image_to_world"], float), h["status"]) for h in hom["homographies"] if h["status"] in BROADCAST_STATUS}
    sam3 = json.loads(Path(args.sam3_json).read_text())
    records = {int(r.get("frame", i)): r for i, r in enumerate(sam3["frames"])}
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    mask_dir = Path(args.instance_mask_dir)
    names = dict(item.split("=", 1) for item in args.team_names.split(",") if "=" in item)
    settings = fpt.TrackerSettings(
        fps=args.fps, max_gap=int(round(args.max_gap_seconds * args.fps)), max_speed=args.max_speed_ft_per_s,
        noise_ft=args.position_noise_ft, confirm=max(2, int(round(0.3 * args.fps))),
    )

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

    fits = [by_name.get(path.name) for path in frame_paths]
    points = []
    for idx, (path, fit) in enumerate(zip(frame_paths, fits)):
        if fit is not None and idx in records:
            points += floor_points(path, idx, records[idx], mask_dir, fit[0], dims)
    tracks, stats = fpt.track_players(points, settings)
    players_by_frame: list[list[dict]] = [[] for _ in frame_paths]
    for track_id, track in enumerate(tracks, 1):
        for f, (state, _) in track["states"].items():
            p = track["measured"].get(f)
            players_by_frame[f].append({
                "track": track_id, "team": track["team"], "object_id": p.object_id if p else None,
                "image_point": list(p.image_point) if p else None,
                "measured_field_point_ft": p.field_point.tolist() if p else None,
                "field_point_ft": state[:2].tolist(), "interpolated": p is None, "number": None,
            })
    bad_frames = set(stats["bad_frames"])
    jersey = None
    if args.jersey_reads:
        jersey = jersey_numbers(json.loads(Path(args.jersey_reads).read_text()),
                                [{"frame": f, "frame_name": path.name, "players": players_by_frame[f]} for f, path in enumerate(frame_paths)])
        for players in players_by_frame:
            for item in players:
                item["number"] = jersey["track_numbers"].get(item["track"])

    h, w = cv2.imread(str(frame_paths[0])).shape[:2]
    history: dict[int, list[tuple[int, float, float, str]]] = {}
    video, positions = [], []
    for idx, path in enumerate(frame_paths):
        fit, players = fits[idx], players_by_frame[idx]
        canvas = base.copy()
        if fit is not None:
            H = fit[0]
            G = np.linalg.inv(H)
            # sign of the projective scale for floor points in front of the camera
            front = np.sign((G @ np.append(apply_h(H, (0.5 * w, 0.9 * h)), 1.0))[2])
            uvw = G @ floor_world
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
            seen = (uvw[2] * front > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            canvas[rows[seen], cols[seen]] = (0.7 * canvas[rows[seen], cols[seen]] + 0.3 * 255).astype(np.uint8)
        for item in players:
            history.setdefault(item["track"], []).append((idx, *item["field_point_ft"], item["team"]))
        filled = sum(item["interpolated"] for item in players)
        if fit is not None or players:
            source = STATUS_TEXT.get(fit[1], fit[1]) if fit is not None else "camera lost briefly"
            header = f"{path.name}   t = {idx / args.fps:.1f} s   {source}   {len(players)} players"
            if filled:
                header += f" ({filled} interpolated)"
        else:
            header = f"{path.name}   t = {idx / args.fps:.1f} s   close-up or camera cut: no field positions"

        for samples in history.values():
            recent = [(x, y) for f, x, y, _ in samples if idx - trail_frames <= f <= idx]
            if trail_frames and len(recent) >= 2 and samples[-1][0] == idx:
                fill = TEAM_STYLE_BGR.get(samples[-1][3], TEAM_STYLE_BGR["unknown"])[0]
                cv2.polylines(canvas, [np.round(to_px(np.asarray(recent))).astype(np.int32)], False, fill, 2, cv2.LINE_AA)
        for item in players:
            p = tuple(int(v) for v in np.round(to_px(np.asarray([item["field_point_ft"]]))[0]))
            draw_player(canvas, p, item["team"], item["number"] or "")

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
    tracking = {
        "max_gap_seconds": args.max_gap_seconds, "max_speed_ft_per_s": args.max_speed_ft_per_s, "position_noise_ft": args.position_noise_ft,
        "foot_noise_px": settings.foot_px, "partly_hidden_noise_px": settings.hidden_px, "acceleration_ft2_per_s3": settings.acceleration,
        "gate": settings.gate, "confirm_frames": settings.confirm,
        "floor_points": len(points), "floor_points_partly_hidden": sum(p.hidden for p in points),
        "interpolated_points": sum(item["interpolated"] for players in players_by_frame for item in players), **stats,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps({
            "schema": "field_birds_eye_players_v1", "homographies": args.homographies, "sam3_json": args.sam3_json,
            "field_dims": dims.to_dict(), "fps": args.fps, "tracking": tracking,
            "jersey_numbers": {"reads": args.jersey_reads, **jersey["summary"], "players": jersey["players"]} if jersey else None, "frames": positions,
        }) + "\n")
    print(json.dumps({"video": str(out), "frames": len(video), "frames_with_fit": sum(p["has_field_fit"] for p in positions),
                      "player_points": sum(len(p["players"]) for p in positions), **{k: v for k, v in tracking.items() if k != "bad_frames"},
                      "bad_frames": len(bad_frames), **({"tracks_numbered": jersey["summary"]["tracks_numbered"], "players": jersey["summary"]["players"],
                      "share_of_team_player_points_numbered": jersey["summary"]["share_of_team_player_points_numbered"]} if jersey else {})}))


if __name__ == "__main__":
    main()
