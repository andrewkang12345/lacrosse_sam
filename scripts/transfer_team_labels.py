"""Team review for a clip, from the checked teams of its keyframes.

The keyframes are every `--keyframe-stride`-th frame starting at `--keyframe-offset`, the
same images as the frames of the keyframe run. On each keyframe, player masks of the two
runs are matched one-to-one by overlap, and each track collects the checked team of its
matches. A track gets a team when one team holds at least `--min-share` of its votes.

The output is a review JSON for classify_sam3_teams_by_torso_color.py --review. Tracks
that touch no keyframe stay with the classifier; tracks with split votes are listed for
a look.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from link_sam3_chunk_tracks import keyframe_team_votes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam3-json", required=True, help="Player JSON of the clip.")
    parser.add_argument("--instance-mask-dir", required=True)
    parser.add_argument("--keyframe-teams", required=True, help="teams.json of the keyframe run.")
    parser.add_argument("--keyframe-mask-dir", required=True)
    parser.add_argument("--keyframe-offset", type=int, required=True)
    parser.add_argument("--keyframe-stride", type=int, required=True)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--min-share", type=float, default=0.7)
    parser.add_argument("--output-review", required=True)
    args = parser.parse_args()

    track_frames = Counter(int(o) for r in json.loads(Path(args.sam3_json).read_text())["frames"] for o in r["object_ids"])
    votes = keyframe_team_votes(Path(args.keyframe_teams), Path(args.keyframe_mask_dir), Path(args.instance_mask_dir), args.keyframe_offset, args.keyframe_stride, args.min_iou)

    review: dict[str, str] = {
        "_note": "Teams transferred from the checked keyframe run by mask overlap on the shared frames (transfer_team_labels.py)."
    }
    split = {}
    for oid in sorted(votes):
        team, count = votes[oid].most_common(1)[0]
        if count / sum(votes[oid].values()) >= args.min_share:
            review[str(oid)] = team
        else:
            split[str(oid)] = dict(votes[oid])
    output = Path(args.output_review)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=1) + "\n")
    labelled = len(review) - 1
    detections = sum(track_frames.values())
    covered = sum(track_frames[int(o)] for o in review if o != "_note")
    print(
        json.dumps(
            {
                "output": str(output),
                "tracks": len(track_frames),
                "tracks_labelled": labelled,
                "detections_in_labelled_tracks": round(covered / max(detections, 1), 3),
                "labels": dict(Counter(v for k, v in review.items() if k != "_note")),
                "split_votes": split,
                "tracks_without_keyframe": sum(1 for o in track_frames if o not in votes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
