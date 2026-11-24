"""Join SAM3 player tracks across the chunks of run_sam3_text_chunked.py.

SAM3 restarts its object ids in every chunk. At each chunk boundary the masks of the
chunk's last frame are warped into the next chunk's first frame through the tracked
background motion, and masks are matched one-to-one by overlap. Boundaries with no
reliable motion (broadcast cuts) link nothing. With --keyframe-teams, tracks that the
checked keyframe teams put on different teams are never joined.

Writes the player JSON and instance masks with the joined ids.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from fit_field_plane_landmarks import frame_motion_pairs

WORK_SIZE = (640, 360)


def load_masks(mask_dir: Path, frame: int) -> tuple[list[int], np.ndarray]:
    path = mask_dir / f"{frame:08d}.npz"
    if not path.exists():
        return [], np.zeros((0, WORK_SIZE[1], WORK_SIZE[0]), bool)
    with np.load(path) as data:
        ids, masks = data["object_ids"].astype(int).tolist(), data["masks"].astype(np.uint8)
    small = [cv2.resize(m, WORK_SIZE, interpolation=cv2.INTER_NEAREST) for m in masks]
    return ids, np.asarray(small, bool).reshape(-1, WORK_SIZE[1], WORK_SIZE[0])


def overlap_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa, fb = a.reshape(len(a), -1).astype(np.float32), b.reshape(len(b), -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1, keepdims=True) + fb.sum(1, keepdims=True).T - inter
    return inter / np.maximum(union, 1.0)


def keyframe_team_votes(teams_json: Path, keyframe_mask_dir: Path, mask_dir: Path, offset: int, stride: int, min_iou: float) -> dict[int, Counter]:
    """Per track: the checked team of each keyframe mask it matches one-to-one by overlap."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for record in json.loads(teams_json.read_text())["frames"]:
        k = int(record["frame"])
        team_of = dict(zip((int(o) for o in record["object_ids"]), record["teams"]))
        ids_key, masks_key = load_masks(keyframe_mask_dir, k)
        ids_clip, masks_clip = load_masks(mask_dir, offset + stride * k)
        if not ids_key or not ids_clip:
            continue
        iou = overlap_matrix(masks_clip, masks_key)
        for r, c in zip(*linear_sum_assignment(-iou)):
            team = team_of.get(ids_key[c])
            if iou[r, c] >= min_iou and team is not None:
                votes[ids_clip[r]][team] += 1
    return votes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam3-json", required=True)
    parser.add_argument("--instance-mask-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-mask-dir", required=True)
    parser.add_argument("--min-iou", type=float, default=0.4)
    parser.add_argument("--keyframe-teams", default=None, help="teams.json of a checked run on this clip's keyframes.")
    parser.add_argument("--keyframe-mask-dir", default=None)
    parser.add_argument("--keyframe-offset", type=int, default=0)
    parser.add_argument("--keyframe-stride", type=int, default=1)
    args = parser.parse_args()

    meta = json.loads(Path(args.sam3_json).read_text())
    chunk = int(meta["chunk_size"])
    mask_dir, out_dir, frames_dir = Path(args.instance_mask_dir), Path(args.output_mask_dir), Path(args.frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    parent: dict[int, int] = {}
    group_team: dict[int, str] = {}
    if args.keyframe_teams:
        votes = keyframe_team_votes(Path(args.keyframe_teams), Path(args.keyframe_mask_dir), mask_dir, args.keyframe_offset, args.keyframe_stride, 0.5)
        group_team = {oid: count.most_common(1)[0][0] for oid, count in votes.items()}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    scale = np.diag([WORK_SIZE[0] / 1280.0, WORK_SIZE[1] / 720.0, 1.0])
    links, cuts, vetoed = 0, 0, 0
    for boundary in range(chunk, len(names), chunk):
        pair = frame_motion_pairs(frames_dir, [names[boundary - 1], names[boundary]], 200)
        if not pair:
            cuts += 1
            continue
        H, _ = cv2.findHomography(pair[0].pa, pair[0].pb, cv2.RANSAC, 1.5)
        ids_a, masks_a = load_masks(mask_dir, boundary - 1)
        ids_b, masks_b = load_masks(mask_dir, boundary)
        if H is None or not ids_a or not ids_b:
            continue
        small_H = scale @ H @ np.linalg.inv(scale)
        warped = np.asarray([cv2.warpPerspective(m.astype(np.uint8), small_H, WORK_SIZE, flags=cv2.INTER_NEAREST) for m in masks_a], bool)
        iou = overlap_matrix(warped, masks_b)
        for r, c in zip(*linear_sum_assignment(-iou)):
            if iou[r, c] >= args.min_iou:
                root_a, root_b = find(ids_a[r]), find(ids_b[c])
                team_a, team_b = group_team.get(root_a), group_team.get(root_b)
                if team_a is not None and team_b is not None and team_a != team_b:
                    vetoed += 1
                    continue
                parent[root_b] = root_a
                if team_a is None and team_b is not None:
                    group_team[root_a] = team_b
                links += 1

    roots: dict[int, int] = {}
    joined: dict[int, list[int]] = {}
    for record in sorted(meta["frames"], key=lambda r: int(r["frame"])):
        new_ids = []
        listed = {int(oid) for oid in record["object_ids"]}  # masks emptied by undistortion are not listed
        for oid in record["object_ids"]:
            root = find(int(oid))
            new = roots.setdefault(root, len(roots))
            joined.setdefault(new, [])
            if int(oid) not in joined[new]:
                joined[new].append(int(oid))
            new_ids.append(new)
        record["object_ids"] = new_ids
        path = mask_dir / f"{int(record['frame']):08d}.npz"
        if path.exists():
            with np.load(path) as data:
                keep = [i for i, o in enumerate(data["object_ids"]) if int(o) in listed]
                masks = data["masks"][keep]
                ids = [roots[find(int(data["object_ids"][i]))] for i in keep]
            np.savez_compressed(out_dir / path.name, object_ids=np.asarray(ids, np.int32), masks=masks)
    meta["linked_tracks"] = {"tracks_before": sum(len(v) for v in joined.values()), "tracks": len(joined), "links": links, "links_vetoed_by_team": vetoed, "boundaries_without_motion": cuts}
    meta["linked_source_ids"] = {str(k): v for k, v in joined.items()}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta) + "\n")
    print(json.dumps({"output": str(output), **meta["linked_tracks"]}, indent=2))


if __name__ == "__main__":
    main()
