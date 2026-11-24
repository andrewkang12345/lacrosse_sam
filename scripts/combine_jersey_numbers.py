"""Jersey numbers per track segment, per field track, and player identities, from crop reads.

The reads come from looking at the contact sheets of jersey_number_sheets.py: Claude
Code opens each sheet image in a session and records, per crop, only the numbers it can
read with certainty, plus the crops that show a striped official. They are stored by crop
id (`<frame name>#<SAM3 id>`) in data/jersey_number_review/. Each crop is placed in its
segment (one SAM3 ID within one field track) through birds_eye_players.json, so the
reads stay valid when the tracks or the sheets are rebuilt.

Per segment:
- the number is the most-read number; a segment whose crops show two different numbers
  is a conflict (SAM3 moved the ID to another player, or a misread) and gets the number
  only if one is read at least twice as often;
- a segment where most read crops show an official is a referee inside a team track.
Per field track: the track carries a number, at every point including the unread segments
and the filled gaps, when its segments give exactly one number, none of them is an
official, and no other track of the same team carries that number at the same time.
Tracks with the same team and number are one player.

render_field_birds_eye.py --jersey-reads labels each dot with its track's jersey number.
With --labels (crops labelled by eye), the crop reads are scored against the labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

TEAMS = ("black", "white")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reads", required=True, help="Crop reads JSON (data/jersey_number_review/<clip>_reads.json).")
    parser.add_argument("--birds-eye-json", required=True, help="birds_eye_players.json from render_field_birds_eye.py")
    parser.add_argument("--labels", default=None, help="Crop labels JSON to score the reads against.")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def jersey_numbers(reads: dict, frames: list[dict]) -> dict:
    """Numbers per segment and per track; `frames` is the `frames` list of birds_eye_players.json."""
    numbers, officials = reads["numbers"], set(reads["officials"])
    owner: dict[tuple[str, int], tuple[int, int]] = {}
    span: dict[tuple[int, int], list[int]] = {}
    team: dict[int, str] = {}
    points: dict[int, set[int]] = defaultdict(set)  # track -> frames with a dot, filled gaps included
    for record in frames:
        for p in record["players"]:
            team[p["track"]] = p["team"]
            points[p["track"]].add(record["frame"])
            if p["interpolated"]:
                continue
            key = (p["track"], p["object_id"])
            owner[(record["frame_name"], p["object_id"])] = key
            span.setdefault(key, [record["frame"], record["frame"]])[1] = record["frame"]

    numbers_by_segment: dict[tuple[int, int], Counter] = defaultdict(Counter)
    officials_by_segment: Counter = Counter()
    read_by_segment: Counter = Counter()
    unplaced = 0
    for crop in set(numbers) | officials:
        name, oid = crop.split("#")
        key = owner.get((name, int(oid)))
        if key is None:
            unplaced += 1
            continue
        read_by_segment[key] += 1
        if crop in numbers:
            numbers_by_segment[key][numbers[crop]] += 1
        if crop in officials:
            officials_by_segment[key] += 1

    segments = {}
    for key in sorted(read_by_segment):
        counts = numbers_by_segment[key]
        top = counts.most_common(2)
        number = top[0][0] if top and (len(top) == 1 or top[0][1] >= 2 * top[1][1]) else None
        segments[f"{key[0]}/{key[1]}"] = {
            "track": key[0], "object_id": key[1], "team": team[key[0]], "frames": span[key], "reads": dict(counts),
            "number": number, "conflict": len(counts) > 1, "official": officials_by_segment[key] * 2 >= read_by_segment[key],
        }

    numbers_by_track: dict[int, set[str]] = defaultdict(set)
    official_tracks = set()
    for seg in segments.values():
        if seg["official"]:
            official_tracks.add(seg["track"])
        elif seg["number"] is not None:
            numbers_by_track[seg["track"]].add(seg["number"])
    candidates = {t: next(iter(n)) for t, n in numbers_by_track.items() if len(n) == 1 and t not in official_tracks and team[t] in TEAMS}
    by_player: dict[str, list[int]] = defaultdict(list)
    for t, n in sorted(candidates.items()):
        by_player[f"{team[t]} #{n}"].append(t)
    overlaps = []
    for player, tracks in by_player.items():
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                shared = points[a] & points[b]
                if len(shared) > 2:
                    overlaps.append({"player": player, "tracks": [a, b], "frames": [min(shared), max(shared)], "shared_frames": len(shared)})
    clash = {t for o in overlaps for t in o["tracks"]}
    track_numbers = {t: n for t, n in candidates.items() if t not in clash}
    players = {p: [t for t in tracks if t in track_numbers] for p, tracks in sorted(by_player.items())}
    players = {p: sorted(tracks, key=lambda t: min(points[t])) for p, tracks in players.items() if tracks}

    team_points = sum(len(f) for t, f in points.items() if team[t] in TEAMS)
    summary = {
        "segments_with_reads": len(segments),
        "segments_named": sum(s["number"] is not None and not s["official"] for s in segments.values()),
        "tracks_numbered": len(track_numbers),
        "share_of_team_player_points_numbered": round(sum(len(points[t]) for t in track_numbers) / max(team_points, 1), 3),
        "players": len(players),
        "reads_outside_any_segment": unplaced,
        "segments_with_two_numbers": [n for n, s in segments.items() if s["conflict"]],
        "tracks_with_two_numbers": sorted(t for t, n in numbers_by_track.items() if len(n) > 1),
        "officials_in_team_tracks": [n for n, s in segments.items() if s["official"] and s["team"] != "referee"],
        "same_number_at_the_same_time": overlaps,
    }
    return {"summary": summary, "track_numbers": track_numbers, "players": players, "segments": segments}


def score(numbers: dict[str, str], officials: set[str], labels_path: Path) -> dict:
    labels = json.loads(labels_path.read_text())
    unsure = set(labels.get("unsure", []))
    crops = labels["crops"]
    legible = [c for c, v in crops.items() if v is not None and c not in unsure]
    blank = [c for c, v in crops.items() if v is None]
    referees = set(labels.get("referee_in_team_track", []))
    return {
        "legible": {"crops": len(legible), "correct": sum(numbers.get(c) == crops[c] for c in legible),
                    "wrong": {c: {"label": crops[c], "read": numbers[c]} for c in legible if c in numbers and numbers[c] != crops[c]},
                    "missed": sum(c not in numbers for c in legible)},
        "no_legible_number": {"crops": len(blank), "false_reads": {c: numbers[c] for c in blank if c in numbers}},
        "unsure": {"crops": len(unsure), "read": sum(c in numbers for c in unsure), "read_as_labelled": sum(numbers.get(c) == crops[c] for c in unsure)},
        "officials": {"labelled": len(referees), "flagged": len(referees & officials), "flagged_elsewhere": len((officials - referees) & set(crops))},
    }


def main() -> None:
    args = parse_args()
    reads = json.loads(Path(args.reads).read_text())
    result = jersey_numbers(reads, json.loads(Path(args.birds_eye_json).read_text())["frames"])
    out = {"reads": args.reads, "birds_eye_json": args.birds_eye_json, "summary": result["summary"],
           "players": result["players"], "segments": result["segments"]}
    if args.labels:
        out["score"] = score(reads["numbers"], set(reads["officials"]), Path(args.labels))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({**result["summary"], "players": {p: len(t) for p, t in result["players"].items()},
                      **({"score": out["score"]} if args.labels else {})}, indent=1))


if __name__ == "__main__":
    main()
