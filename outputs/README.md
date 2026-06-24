# Outputs Layout

`outputs/` is generated-local workspace data and is intentionally ignored by Git. Only this README should be tracked.

## Directories

- `clips/`: source-derived preview clips.
- `prompts/`: prompt-frame contact sheets and manifests.
- `sam2/player_tracks/`: SAM2 player click-track outputs.
- `sam2/floor_features/`: SAM2 floor-landmark tracks and masks.
- `sam3/text/`: SAM3 text-prompt detections and masks.
- `sam3/team_classification/`: torso-color and TransReID team-label outputs.
- `meshes/sam_body4d/`: SAM-Body4D renders, metadata, and mesh directories.
- `meshes/4d_humans/`: 4D-Humans renders, metadata, and mesh directories.
- `floor/`: click annotations, SAM2 floor prompts, and homography calibrations.
- `birds_eye/`: older homography-based top-down player-location outputs.
- `overlays/camera/`: camera-view floor-fit debug overlays.
- `debug/`: extracted review frames and temporary calibration diagnostics.
- `vggt/`: VGGT reconstruction, field fitting, viewers, and bird's-eye experiments.
- `misc/`: leftovers that do not fit a stable category yet.

The root README contains the reproducible commands for generating the VGGT + SAM3 + SAM-Body4D visualization.
