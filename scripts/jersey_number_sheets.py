"""Contact sheets of player upper-body crops for reading jersey numbers.

The unit is a segment: one SAM3 ID within one field track of render_field_birds_eye.py.
For each segment of a team player that lasts at least --min-seconds, up to
--crops-per-segment frames are picked, spread over the segment's time so that some show
the player's back. Within each time slice the frame with the tallest, sharpest mask
wins; partly hidden players (see field_player_tracks.partly_hidden) are skipped.
Goalies (segments whose median floor point lies in or next to a crease) and referees are
left out.

Each crop is the top 60% of the player's mask box, widened to a 3:4 tile, with
everything outside the player's mask dimmed so the tile is about one player. Tiles are
packed into numbered grids (--cols x --rows), one image per sheet.

Outputs in --output-dir:
- crops.json: every selected crop, keyed `<frame name>#<SAM3 id>`, with its segment,
  team, crop box and mask height;
- sheets/sheet_NNN.png and sheets.json: which crop sits in which tile.
--only-crops makes sheets of exactly the crops listed in a label file (for checking reads
against labels); those crops are rebuilt from their frame and SAM3 id, so the labels stay
usable when the selection changes. --tile-scale enlarges the tiles (for labelling by eye).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import field_model as fm
import field_player_tracks as fpt
from render_field_birds_eye import load_masks

TILE_W, TILE_H, STRIP_H = 96, 128, 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--birds-eye-json", required=True, help="birds_eye_players.json from render_field_birds_eye.py")
    parser.add_argument("--instance-mask-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crops-per-segment", type=int, default=6)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--candidates-per-segment", type=int, default=60, help="Frames scored per segment, evenly spaced.")
    parser.add_argument("--crease-margin-ft", type=float, default=4.0, help="Segments whose median floor point is this close to a crease are goalies.")
    parser.add_argument("--only-crops", default=None, help="Label JSON; sheets and crops.json hold exactly its crops.")
    parser.add_argument("--tile-scale", type=float, default=1.0)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--rows", type=int, default=5)
    return parser.parse_args()


def crop_box(mask_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Top 60% of the mask box, widened to the tile's 3:4 shape, around the box centre."""
    x0, y0, x1, y1 = mask_box
    h = y1 - y0
    top, bottom = y0 - int(0.05 * h), y0 + int(0.6 * h)
    width = max(x1 - x0, int(round(0.75 * (bottom - top))))
    height = int(round(width / 0.75))
    cx = (x0 + x1) // 2
    return cx - width // 2, top, cx - width // 2 + width, top + height


def render_tile(image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], scale: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    keep = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    dimmed = np.where(keep[..., None], image, (0.35 * image).astype(np.uint8))
    pad = max(0, -x0, -y0, x1 - image.shape[1], y1 - image.shape[0])
    padded = cv2.copyMakeBorder(dimmed, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    size = (int(round(TILE_W * scale)), int(round(TILE_H * scale)))
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA if crop.shape[1] > size[0] else cv2.INTER_CUBIC)


def goalie(xy: np.ndarray, dims: fm.FieldDims, margin: float) -> bool:
    return any(np.hypot(xy[0] - fm.goal_x(side, dims), xy[1] - dims.cy) < dims.crease_radius + margin for side in ("left", "right"))


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.birds_eye_json).read_text())
    dims = fm.FieldDims(**data["field_dims"])
    fps = float(data["fps"])
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    mask_dir = Path(args.instance_mask_dir)

    segments: dict[tuple[int, int], dict] = defaultdict(lambda: {"frames": [], "points": [], "team": None})
    for record in data["frames"]:
        for p in record["players"]:
            if not p["interpolated"]:
                seg = segments[(p["track"], p["object_id"])]
                seg["frames"].append(record["frame"])
                seg["points"].append(p["measured_field_point_ft"])
                seg["team"] = p["team"]
    eligible = {}
    skipped = {"short": 0, "goalie": 0, "referee": 0}
    for key, seg in segments.items():
        if seg["team"] == "referee":
            skipped["referee"] += 1
        elif len(seg["frames"]) < args.min_seconds * fps:
            skipped["short"] += 1
        elif goalie(np.median(np.asarray(seg["points"]), axis=0), dims, args.crease_margin_ft):
            skipped["goalie"] += 1
        else:
            eligible[key] = seg

    wanted: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key, seg in eligible.items():
        frames = seg["frames"]
        for i in np.unique(np.linspace(0, len(frames) - 1, min(len(frames), args.candidates_per_segment)).round().astype(int)):
            wanted[frames[i]].append(key)
    scored: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for frame, keys in sorted(wanted.items()):
        masks = load_masks(mask_dir, frame)
        hidden = fpt.partly_hidden(masks)
        image = None
        for key in keys:
            mask = masks.get(key[1])
            if mask is None or key[1] in hidden or mask.sum() < fpt.MIN_MASK_PX:
                continue
            rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            mask_box = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
            box = crop_box(mask_box)
            if image is None:
                image = cv2.imread(str(frame_paths[frame]))
            gray = cv2.cvtColor(image[max(0, box[1]):box[3], max(0, box[0]):box[2]], cv2.COLOR_BGR2GRAY)
            scored[key].append({"frame": frame, "height": mask_box[3] - mask_box[1], "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "box": box})

    crops = {}
    for key, cands in scored.items():
        cands.sort(key=lambda c: c["frame"])
        sharp = np.median([c["sharpness"] for c in cands]) or 1.0
        for chunk in np.array_split(np.arange(len(cands)), min(args.crops_per_segment, len(cands))):
            best = max((cands[i] for i in chunk), key=lambda c: c["height"] * np.sqrt(min(c["sharpness"] / sharp, 2.0)))
            name = frame_paths[best["frame"]].name
            crops[f"{name}#{key[1]}"] = {"frame": best["frame"], "frame_name": name, "object_id": key[1], "track": key[0],
                                         "team": eligible[key]["team"], "mask_height_px": best["height"], "crop_box": list(best["box"])}

    out = Path(args.output_dir)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    (out / "crops.json").write_text(json.dumps({"birds_eye_json": args.birds_eye_json, "segments": len(eligible), "skipped_segments": skipped, "crops": crops}, indent=1) + "\n")

    if args.only_crops:  # labelled crops are rebuilt from their frame and SAM3 id, selected or not
        owner = {(r["frame_name"], p["object_id"]): (r["frame"], p["track"], p["team"]) for r in data["frames"] for p in r["players"] if not p["interpolated"]}
        sheet_crops = {}
        for c in json.loads(Path(args.only_crops).read_text())["crops"]:
            name, oid = c.split("#")
            if c in crops:
                sheet_crops[c] = crops[c]
            elif (name, int(oid)) in owner:
                frame, track, team = owner[(name, int(oid))]
                mask = load_masks(mask_dir, frame)[int(oid)]
                rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
                sheet_crops[c] = {"frame": frame, "frame_name": name, "object_id": int(oid), "track": track, "team": team,
                                  "mask_height_px": int(rows[-1] - rows[0]), "crop_box": list(crop_box((cols[0], rows[0], cols[-1], rows[-1])))}
            else:
                raise SystemExit(f"labelled crop {c} has no measured floor point in {args.birds_eye_json}")
        crops = sheet_crops
        (out / "crops.json").write_text(json.dumps({"birds_eye_json": args.birds_eye_json, "only_crops": args.only_crops, "crops": crops}, indent=1) + "\n")
    ids = sorted(crops, key=lambda c: (crops[c]["track"], crops[c]["object_id"], crops[c]["frame"]))
    per_sheet = args.cols * args.rows
    strip = int(round(STRIP_H * args.tile_scale))
    tile_w, tile_h = int(round(TILE_W * args.tile_scale)), int(round(TILE_H * args.tile_scale))
    by_frame: dict[int, list[str]] = defaultdict(list)
    for c in ids:
        by_frame[crops[c]["frame"]].append(c)
    tiles = {}
    for frame, members in by_frame.items():
        image, masks = cv2.imread(str(frame_paths[frame])), load_masks(mask_dir, frame)
        for c in members:
            tiles[c] = render_tile(image, masks[crops[c]["object_id"]], tuple(crops[c]["crop_box"]), args.tile_scale)
    sheets = []
    for s, start in enumerate(range(0, len(ids), per_sheet)):
        members = ids[start:start + per_sheet]
        sheet = np.zeros((args.rows * (tile_h + strip), args.cols * tile_w, 3), np.uint8)
        for t, c in enumerate(members):
            r, col = divmod(t, args.cols)
            y, x = r * (tile_h + strip), col * tile_w
            sheet[y + strip:y + strip + tile_h, x:x + tile_w] = tiles[c]
            cv2.putText(sheet, str(t), (x + 3, y + strip - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * args.tile_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(sheet, (x + tile_w - 1, y), (x + tile_w - 1, y + strip + tile_h), (90, 90, 90), 1)
        name = f"sheet_{s:03d}.png"
        cv2.imwrite(str(out / "sheets" / name), sheet)
        sheets.append({"sheet": name, "width": sheet.shape[1], "height": sheet.shape[0], "tiles": members})
    (out / "sheets.json").write_text(json.dumps({"crops_json": str(out / "crops.json"), "tile_scale": args.tile_scale, "sheets": sheets}, indent=1) + "\n")
    tokens = sum(s["width"] * s["height"] / 750 for s in sheets)
    print(json.dumps({"segments": len(eligible), "skipped_segments": skipped, "crops": len(crops), "crops_on_sheets": len(ids),
                      "sheets": len(sheets), "image_tokens_estimate": int(tokens)}, indent=1))


if __name__ == "__main__":
    main()
