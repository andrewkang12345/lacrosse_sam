from __future__ import annotations

import argparse
import json
from pathlib import Path


OBJECT_ID_BY_LANDMARK = {
    "left_upper": 8,
    "left_lower": 9,
    "right_upper": 6,
    "right_lower": 7,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert exact faceoff-point clicks into SAM2 video prompts.")
    parser.add_argument("--clicks", default="outputs/floor/clicks/floor_faceoff_corner_clicks.json")
    parser.add_argument("--output", default="outputs/floor/sam2_prompts/floor_faceoff_corner_sam2_prompts.json")
    args = parser.parse_args()

    data = json.loads(Path(args.clicks).read_text())
    prompts = []
    by_landmark: dict[str, int] = {}
    for click in data.get("clicks", []):
        landmark = str(click["landmark_id"])
        if landmark not in OBJECT_ID_BY_LANDMARK:
            continue
        prompts.append(
            {
                "frame": int(click["frame"]),
                "object_id": int(OBJECT_ID_BY_LANDMARK[landmark]),
                "feature": landmark,
                "points": [[float(click["image"]["x"]), float(click["image"]["y"])]],
                "labels": [1],
                "world": click["world"],
            }
        )
        by_landmark[landmark] = by_landmark.get(landmark, 0) + 1

    output = {
        "schema": "sam2_click_prompts_v1",
        "source_clicks": args.clicks,
        "object_id_map": OBJECT_ID_BY_LANDMARK,
        "prompts": prompts,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(out_path), "prompts": len(prompts), "by_landmark": by_landmark}, indent=2))


if __name__ == "__main__":
    main()
