# Outputs Layout

`outputs/` is generated-local workspace data and is intentionally ignored by Git. Only this README and `CURRENT_ARTIFACTS.json` should be tracked.

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

## Current Recommended Run

The latest stack is the Week 1 `112825_osh_tor.mp4` clip from `00:08:20` to `00:10:53`, using undistorted 1 fps frames for VGGT/SAM3 field fitting and the 10 fps render rate used by the SAM-Body4D overlay.

Key artifacts:

```text
outputs/vggt/week1_112825_osh_tor_0820_1053_vggt_1fps_undistorted/vggt_predictions_compact.npz
outputs/vggt/week1_112825_osh_tor_0820_1053_sam3_text_field_fit_undistorted_partial_green_anchors/field_fit_vggt_sam3_text_masks_huddle_goalies.json
outputs/meshes/sam_body4d/week1_112825_osh_tor_0820_1053_sam3_text_player_all_undistorted_h264.mp4
outputs/vggt/week1_112825_osh_tor_0820_1053_sam3_text_sam_body4d_fused_all_players_undistorted_anchors_10fps/birds_eye_sam_body4d_meshes_and_field_masks_vggt_h264.mp4
outputs/vggt/week1_112825_osh_tor_0820_1053_sam3_text_sam_body4d_fused_all_players_undistorted_anchors_10fps/birds_eye_player_locations_sam_body4d_vggt_h264.mp4
```

See `CURRENT_ARTIFACTS.json` for a machine-readable index.
