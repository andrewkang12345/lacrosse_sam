from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_FEATURE_OBJECT_IDS = {
    "left_restraining_line": 1,
    "right_restraining_line": 2,
    "midfield_line": 3,
    "goal_crease": 4,
    "field_outline": 5,
}


def parse_mapping(value: str) -> dict[str, int]:
    mapping = DEFAULT_FEATURE_OBJECT_IDS.copy()
    if not value:
        return mapping
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("feature object mapping entries must be feature=id")
        feature, raw_id = item.split("=", 1)
        mapping[feature.strip()] = int(raw_id)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-clicks", default="outputs/floor_feature_clicks_no_outline.json")
    parser.add_argument("--output", default="outputs/floor_feature_sam2_prompts.json")
    parser.add_argument("--object-id-map", default="")
    parser.add_argument("--exclude-features", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.feature_clicks).read_text())
    mapping = parse_mapping(args.object_id_map)
    excluded = {item.strip() for item in args.exclude_features.split(",") if item.strip()}

    prompts = []
    for click in data.get("clicks", []):
        feature = str(click["feature"])
        if feature in excluded:
            continue
        if feature not in mapping:
            raise ValueError(f"No SAM2 object ID mapping for feature {feature!r}")
        image = click["image"]
        prompts.append(
            {
                "frame": int(click["frame"]),
                "object_id": int(mapping[feature]),
                "feature": feature,
                "points": [[float(image["x"]), float(image["y"])]],
                "labels": [1],
            }
        )

    output = {
        "schema": "sam2_click_prompts_v1",
        "source_feature_clicks": args.feature_clicks,
        "object_id_map": mapping,
        "prompts": prompts,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    by_feature = {}
    for prompt in prompts:
        by_feature[prompt["feature"]] = by_feature.get(prompt["feature"], 0) + 1
    print(json.dumps({"output": args.output, "prompts": len(prompts), "by_feature": by_feature}, indent=2))


if __name__ == "__main__":
    main()
