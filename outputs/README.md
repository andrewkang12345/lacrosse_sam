# Outputs Layout

`outputs/` is generated-local workspace data and is intentionally ignored by Git. Only this README should be tracked.

## Directories

- `clips/`: source-derived preview clips.
- `prompts/`: prompt-frame contact sheets and manifests.
- `sam2/player_tracks/`: SAM2 player click-track outputs.
- `sam2/floor_features/`: SAM2 floor-landmark tracks and masks.
- `sam3/text/`: SAM3 text-prompt detections and masks (10 s clip).
- `sam3/week1_*_1fps/`, `sam3/week1_*_1fps_undistorted/`: Week 1 SAM3 masks on the original and the lens-undistorted frames.
- `sam3/team_classification/`: torso-color and TransReID team-label outputs.
- `meshes/sam_body4d/`: SAM-Body4D renders, metadata, and mesh directories.
- `meshes/4d_humans/`: 4D-Humans renders, metadata, and mesh directories.
- `floor/`: click annotations, SAM2 floor prompts, and homography calibrations.
- `birds_eye/`: older homography-based top-down player-location outputs.
- `overlays/camera/`: camera-view floor-fit debug overlays.
- `vggt_omega/<clip>/`: VGGT-Omega predictions (`vggt_omega_predictions.npz`), labeled landmark masks and their review sheets (`paint_masks/`), field fit (`field_fit/`), 3D videos (`scene_3d/`), the 518x518 VGGT-layout export (`vggt_layout/`), team labels (`teams/`) and the bird's-eye player video (`birds_eye/`).
- `vggt/`: VGGT (non-Omega) runs and fits that the existing CoTracker and SAM-Body4D fusion outputs were made from, plus lens-undistortion parameters (`undistorted/`, `week1_*_undistorted/`).
- `misc/`: leftovers that do not fit a stable category yet.

The root README contains the reproducible commands for generating the VGGT-Omega + SAM3 + SAM-Body4D visualization; `docs/field_plane_fit.md` explains the field fit.
