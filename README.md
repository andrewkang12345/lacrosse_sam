# Lacrosse SAM Tracking

Small workflow for segmenting and tracking lacrosse players in a 10 second clip using:

- SAM3 text prompts through Hugging Face Transformers
- SAM2 click prompts through Hugging Face Transformers
- TransReID appearance clustering for team/goalkeeper labels
- SAM-Body4D / SAM-3D-Body for per-player human meshes
- A local browser click annotator for interactive player prompts
- H.264 MP4 outputs viewable in VS Code

## Inputs

Source video:

```bash
data/lacrosse_long_source_iniyaa.mp4
```

Source videos, extracted frames, SMPL model files, checkpoints, third-party repos, and generated outputs are local-only and ignored by Git. Put those assets under `data/`, `checkpoints/`, and `third_party/` before running the scripts.

The workflow samples the first 10 seconds at 10 fps into:

```bash
data/frames_10fps/frame_*.jpg
```

## Outputs Layout

Generated artifacts under `outputs/` are now grouped by workflow stage:

- `outputs/clips/`, `outputs/prompts/`
- `outputs/sam2/player_tracks/`, `outputs/sam2/floor_features/`
- `outputs/sam3/text/`, `outputs/sam3/team_classification/`
- `outputs/meshes/sam_body4d/`, `outputs/meshes/4d_humans/`
- `outputs/floor/clicks/`, `outputs/floor/sam2_prompts/`, `outputs/floor/calibrations/`
- `outputs/birds_eye/`, `outputs/overlays/camera/`
- `outputs/vggt_omega/<clip>/`: VGGT-Omega predictions, `field_fit/`, `scene_3d/` 3D videos, `vggt_layout/`
- `outputs/vggt/`: VGGT (non-Omega) runs and fits referenced by the CoTracker and SAM-Body4D outputs

The older flat paths shown in historical command examples may need to be replaced with the corresponding grouped path. Large files under `outputs/` are intentionally not tracked.

## How The VGGT-Omega + SAM3 + SAM-Body4D Stack Works

This stack combines three separate estimates into one 3D view of the play:

- `SAM3` finds semantic regions in the broadcast frames from text prompts. In the current Week 1 workflow, it is used for `player`, `green field`, `white lines on green field`, and `yellow outline around field border`. The player masks define which image pixels belong to each detected person; the field masks seed a bootstrap field fit, and the fit itself uses the colour-detected, labeled paint masks from `build_field_paint_masks.py`.
- `VGGT-Omega` predicts camera intrinsics, camera poses and depth from the video frames, which give a 3D point cloud of the arena and a floor plane.
- `SAM-Body4D` reconstructs one animated human mesh per SAM3 player detection track. The mesh outputs are kept in their own local coordinate systems first, then transformed into the VGGT scene for visualization.

The field fitting step (`fit_field_plane_landmarks.py`) fits a plane to VGGT-Omega's floor points, lifts landmark pixels onto that plane by intersecting camera rays with it, and fits the NLL field there with a rigid similarity (no mirroring, no shape change). It then refines the broadcast camera against the known field: one shared camera position, plus per-frame pan, tilt, roll and zoom, and the arena's board corner radius. That removes VGGT's telephoto zoom/distance distortion. The landmarks are painted lines and the far board base, detected by colour and labeled per side (`build_field_paint_masks.py`), with the labels checked by eye; the field model follows the NLL rulebook (`field_model.py`). [docs/field_plane_fit.md](docs/field_plane_fit.md) explains why depth-based fitting fails on broadcast footage and reports the plane fit's accuracy.

The mesh fusion step places each SAM-Body4D human on the corresponding VGGT-Omega player blob by matching the SAM3 player mask to nearby VGGT 3D points. The all-player path keeps every SAM3 `player` detection, including substitutes visible near the boards. If field grounding introduces offsets, use `--no-ground-meshes-to-field` so placement follows the player blob in VGGT space instead.

The commands in the VGGT section below produce both an interactive `viser` scene and H.264 review videos from local input assets.

## Setup

Run from the repo root:

```bash
cd /mnt/data/lacrosse_sam
python -m pip install --upgrade transformers sam2 accelerate imageio imageio-ffmpeg supervision matplotlib
```

Set your Hugging Face token in the environment when running model scripts:

```bash
export HF_TOKEN='your_huggingface_token'
```

Do not commit tokens to files.

## 1. Prepare The Clip

Extract the 10 second, 10 fps working clip and prompt frames:

```bash
python scripts/prepare_lacrosse_clip.py \
  --video data/lacrosse_long_source_iniyaa.mp4 \
  --seconds 10 \
  --fps 10 \
  --frames-dir data/frames_10fps \
  --prompt-dir outputs/prompt_frames \
  --preview-video outputs/lacrosse_10s_10fps_preview.mp4
```

Useful outputs:

```bash
outputs/lacrosse_10s_10fps_preview.mp4
outputs/prompt_frames/manifest.json
outputs/prompt_frames/prompt_frame_*.jpg
outputs/prompt_frames/prompt_contact_sheet.jpg
```

## 2. Collect Click Prompts

Start the local annotator:

```bash
python scripts/click_annotator.py \
  --manifest outputs/prompt_frames/manifest.json \
  --frames-dir data/frames_10fps \
  --output outputs/click_prompts.json \
  --host 0.0.0.0 \
  --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Click each visible player's torso. Use `Next ID` for each new player. Reuse the same player ID on later frames if you need refinement clicks. Use `negative` only for background or mistaken regions.

Clicks are saved immediately to:

```bash
outputs/click_prompts.json
```

## 3. Run SAM2 Click Tracking

```bash
HF_TOKEN="$HF_TOKEN" python scripts/track_sam2_clicks.py \
  --frames-dir data/frames_10fps \
  --prompts outputs/click_prompts.json \
  --model facebook/sam2.1-hiera-large \
  --output outputs/sam2_click_tracks.mp4 \
  --fps 10
```

Outputs:

```bash
outputs/sam2_click_tracks.mp4
outputs/sam2_click_tracks.json
```

## 4. Run SAM3 Text Tracking

```bash
HF_TOKEN="$HF_TOKEN" python scripts/track_sam3_text.py \
  --frames-dir data/frames_10fps \
  --text 'lacrosse player' \
  --model facebook/sam3 \
  --output outputs/sam3_text_lacrosse_player.mp4 \
  --mask-dir outputs/sam3_text_label_masks \
  --fps 10
```

Outputs:

```bash
outputs/sam3_text_lacrosse_player.mp4
outputs/sam3_text_lacrosse_player.json
outputs/sam3_text_label_masks/
```

## 5. Run SAM-Body4D Meshes

Install and download SAM-Body4D assets from its repo, then run the box-driven mesh pipeline from the SAM3 text output. Use `--render-mode overlay` to composite meshes on the original camera frames:

```bash
PYOPENGL_PLATFORM=egl python scripts/run_sam_body4d_from_sam3_boxes.py \
  --repo-root third_party/sam-body4d \
  --config third_party/sam-body4d/configs/body4d.yaml \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_text_player_masks.json \
  --output-dir outputs/sam_body4d_player_overlay \
  --output-video outputs/sam_body4d_player_overlay_h264.mp4 \
  --render-mode overlay \
  --fps 10
```

Outputs:

```bash
outputs/sam_body4d_player_overlay_h264.mp4
outputs/sam_body4d_player_overlay_review_sheet.jpg
outputs/sam_body4d_player_overlay/mesh_4d_individual/<object_id>/*.ply
outputs/sam_body4d_player_overlay/focal_4d_individual/<object_id>/*.json
outputs/sam_body4d_player_overlay/metadata.json
```

## 6. Run 4D-Humans Meshes

Place the neutral SMPL model where the repo can load it, or convert it into the 4D-Humans cache:

```bash
python - <<'PY'
from pathlib import Path
import collections, inspect, sys
import numpy as np
if not hasattr(inspect, 'getargspec'):
    ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')
    inspect.getargspec = lambda fn: ArgSpec(*inspect.getfullargspec(fn)[:4])
for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex, 'object': object, 'str': str, 'unicode': str}.items():
    if not hasattr(np, name):
        setattr(np, name, value)
sys.path.insert(0, str(Path('third_party/4D-Humans').resolve()))
from hmr2.models import convert_pkl
src = 'data/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
dst = Path.home() / '.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl'
dst.parent.mkdir(parents=True, exist_ok=True)
convert_pkl(src, str(dst))
PY
```

Then run HMR2 from the same SAM3 `player` boxes:

```bash
PYOPENGL_PLATFORM=egl python scripts/run_4d_humans_from_sam3_boxes.py \
  --repo-root third_party/4D-Humans \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_text_player_masks.json \
  --output-dir outputs/4d_humans_sam3_player \
  --output-video outputs/4d_humans_sam3_player_h264.mp4 \
  --fps 10 \
  --save-mesh
```

Outputs:

```bash
outputs/4d_humans_sam3_player_h264.mp4
outputs/4d_humans_sam3_player_review_sheet.jpg
outputs/4d_humans_sam3_player/meshes_obj/<object_id>/*.obj
outputs/4d_humans_sam3_player/metadata.json
```

## 7. Team-Colored Meshes

The recommended team-color path uses one generic SAM3 `player` prompt, then classifies each tracked player by masked torso color composition. It measures white, yellow/gold, dark, and gray torso pixel fractions instead of relying on one average color. Occluded or low-visibility torso samples are excluded from the per-player color decision, and labels are smoothed by SAM3 object ID so a jersey number or temporary shadow does not flip a player frame-by-frame.

Run SAM3 once and export per-instance masks:

```bash
HF_TOKEN="$HF_TOKEN" python scripts/track_sam3_text.py \
  --frames-dir data/frames_10fps \
  --text 'player' \
  --model facebook/sam3 \
  --output outputs/sam3_text_player_instances.mp4 \
  --mask-dir outputs/sam3_text_player_instances_label_masks \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --fps 10
```

Classify teams from upper-torso color:

```bash
python scripts/classify_sam3_teams_by_torso_color.py \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_text_player_instances.json \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-json outputs/sam3_team_torso_color_detections.json \
  --output-video outputs/sam3_team_torso_color_detections_review.mp4 \
  --fps 10 \
  --classification-mode composition \
  --white-fraction-threshold 0.25 \
  --yellow-fraction-threshold 0.025 \
  --team-colors 'black=255,80,40;white=70,170,255;unknown=180,180,180'
```

Use the merged JSON in either mesh runner. `team_colors` in the JSON controls the mesh colors:

```bash
PYOPENGL_PLATFORM=egl python scripts/run_sam_body4d_from_sam3_boxes.py \
  --repo-root third_party/sam-body4d \
  --config third_party/sam-body4d/configs/body4d.yaml \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_team_torso_color_detections.json \
  --output-dir outputs/sam_body4d_torso_color_overlay \
  --output-video outputs/sam_body4d_torso_color_overlay_h264.mp4 \
  --render-mode overlay \
  --fps 10

PYOPENGL_PLATFORM=egl python scripts/run_4d_humans_from_sam3_boxes.py \
  --repo-root third_party/4D-Humans \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_team_torso_color_detections.json \
  --output-dir outputs/4d_humans_torso_color_overlay \
  --output-video outputs/4d_humans_torso_color_overlay_h264.mp4 \
  --fps 10 \
  --save-mesh
```

## 8. TransReID Team Clusters

Use TransReID when uniform color is not enough. This clusters each SAM3 tracked object by ReID embeddings from the official TransReID model. For this clip, use three clusters so the goalkeeper can separate from the two field-player teams:

```bash
git clone https://github.com/damo-cv/TransReID.git third_party/TransReID
mkdir -p checkpoints/transreid
gdown --id 1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK \
  -O checkpoints/transreid/msmt17_vit_transreid_stride.pth

python scripts/classify_sam3_teams_by_transreid.py \
  --clusters 3 \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_text_player_instances.json \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-json outputs/sam3_team_transreid_3clusters_detections.json \
  --output-video outputs/sam3_team_transreid_3clusters_detections_review.mp4 \
  --review-sheet outputs/sam3_team_transreid_3clusters_detections_review_sheet.jpg \
  --batch-size 16 \
  --fps 10
```

Then render meshes with the three-cluster labels:

```bash
PYOPENGL_PLATFORM=egl python scripts/run_sam_body4d_from_sam3_boxes.py \
  --repo-root third_party/sam-body4d \
  --config third_party/sam-body4d/configs/body4d.yaml \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --output-dir outputs/sam_body4d_transreid_3clusters_overlay \
  --output-video outputs/sam_body4d_transreid_3clusters_overlay_h264.mp4 \
  --render-mode overlay \
  --fps 10

PYOPENGL_PLATFORM=egl python scripts/run_4d_humans_from_sam3_boxes.py \
  --repo-root third_party/4D-Humans \
  --frames-dir data/frames_10fps \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --output-dir outputs/4d_humans_transreid_3clusters_overlay \
  --output-video outputs/4d_humans_transreid_3clusters_overlay_h264.mp4 \
  --fps 10 \
  --save-mesh
```

Current 3-cluster outputs:

```bash
outputs/sam3_team_transreid_3clusters_detections_review.mp4
outputs/sam3_team_transreid_3clusters_detections_review_sheet.jpg
outputs/sam_body4d_transreid_3clusters_overlay_h264.mp4
outputs/4d_humans_transreid_3clusters_overlay_h264.mp4
```

## 9. Bird's-Eye Player Locations

Calibrate image pixels to floor coordinates with clicked floor landmarks. The world coordinate system is x = 0..200 ft along the floor length and y = 0..85 ft across the floor width.

Start the free-click landmark UI:

```bash
python scripts/floor_free_click_annotator.py \
  --manifest outputs/prompt_frames/manifest.json \
  --frames-dir data/frames_10fps \
  --output outputs/floor_unlabeled_clicks.json \
  --host 0.0.0.0 \
  --port 8770
```

Open:

```text
http://127.0.0.1:8770
```

Click any visible floor landmarks. You do not need to label which landmark is which. Use 8 or more points per calibrated frame if possible. Good points are white-line intersections, center/face-off dots, crease extrema, goal posts, and yellow boundary tangencies. If the camera pans or zooms, click landmarks on multiple frames.

Render the top-down floor map from the three-cluster SAM3/TransReID JSON:

```bash
python scripts/render_birds_eye_locations.py \
  --calibration outputs/floor_unlabeled_clicks.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/birds_eye_player_locations_h264.mp4 \
  --output-json outputs/birds_eye_player_locations.json \
  --auto-fit-iterations 40000 \
  --fps 10
```

The renderer auto-matches the unlabeled clicks to the known NLL floor line/landmark model, uses each player's SAM3 mask bottom point, falls back to the detection box bottom-center, maps it through the nearest calibrated homography, and draws colored team dots on a 200 ft x 85 ft NLL floor diagram.

The more reliable calibration path uses coarse feature labels instead of fully free clicks. Select one feature category, then click any point on that feature:

```bash
python scripts/floor_feature_annotator.py \
  --manifest outputs/prompt_frames/manifest.json \
  --frames-dir data/frames_10fps \
  --output outputs/floor_feature_clicks.json \
  --host 0.0.0.0 \
  --port 8770

python scripts/fit_floor_homography_from_feature_clicks.py \
  --feature-clicks outputs/floor_feature_clicks.json \
  --initial-calibration outputs/floor_homography_dynamic.json \
  --output outputs/floor_homography_feature_fit.json \
  --reference-frame 78
```

The feature fitter models each goal crease as the visible circular arc segment, not as a full circle or rectangle.
To fit without field-outline clicks:

```bash
python scripts/fit_floor_homography_from_feature_clicks.py \
  --feature-clicks outputs/floor_feature_clicks.json \
  --initial-calibration outputs/floor_homography_feature_dynamic.json \
  --output outputs/floor_homography_feature_no_outline_fit.json \
  --reference-frame 78 \
  --exclude-features field_outline
```

For debugging and refinement, fit clicks as points on floor features and project the fitted floor model back onto the camera video:

```bash
python scripts/refine_floor_homography_from_unlabeled_clicks.py \
  --unlabeled-clicks outputs/floor_unlabeled_clicks.json \
  --initial-calibration outputs/floor_homography_inferred_frame77.json \
  --output outputs/floor_homography_curve_refined.json \
  --frame 77

python scripts/render_camera_homography_overlay.py \
  --calibration outputs/floor_homography_curve_refined.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/camera_floor_homography_overlay_refined_h264.mp4 \
  --fps 10
```

For broadcast camera motion, estimate a dynamic homography for every frame by registering each frame to the calibrated reference frame:

```bash
python scripts/estimate_dynamic_floor_homographies.py \
  --frames-dir data/frames_10fps \
  --base-calibration outputs/floor_homography_feature_fit.json \
  --output outputs/floor_homography_feature_dynamic.json \
  --reference-frame 78

python scripts/render_camera_homography_overlay.py \
  --calibration outputs/floor_homography_feature_dynamic.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/camera_floor_homography_overlay_feature_dynamic_h264.mp4 \
  --fps 10

python scripts/render_birds_eye_locations.py \
  --calibration outputs/floor_homography_feature_dynamic.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/birds_eye_player_locations_feature_dynamic_h264.mp4 \
  --output-json outputs/birds_eye_player_locations_feature_dynamic.json \
  --fps 10
```

To use SAM2 to segment and track the clicked floor landmarks, convert the coarse feature clicks into one SAM2 object per landmark class, propagate those masks, and refit the dynamic homographies from the tracked mask pixels:

```bash
python scripts/floor_feature_clicks_to_sam2_prompts.py \
  --feature-clicks outputs/floor_feature_clicks.json \
  --output outputs/floor_feature_sam2_prompts_with_outline.json

python scripts/track_sam2_clicks.py \
  --frames-dir data/frames_10fps \
  --prompts outputs/floor_feature_sam2_prompts_with_outline.json \
  --model facebook/sam2.1-hiera-large \
  --output outputs/sam2_floor_feature_tracks_with_outline_h264.mp4 \
  --mask-dir outputs/sam2_floor_feature_label_masks_with_outline \
  --instance-mask-dir outputs/sam2_floor_feature_instance_masks_with_outline \
  --fps 10

python scripts/fit_floor_homography_from_tracked_landmarks.py \
  --frames-dir data/frames_10fps \
  --initial-calibration outputs/floor_homography_feature_no_outline_dynamic.json \
  --feature-clicks outputs/floor_feature_clicks.json \
  --output outputs/floor_homography_sam2_landmarks_primitive_full_segment_with_outline_dynamic.json \
  --mask-source 'left_restraining_line=outputs/sam2_floor_feature_instance_masks_with_outline#1' \
  --mask-source 'right_restraining_line=outputs/sam2_floor_feature_instance_masks_with_outline#2' \
  --mask-source 'midfield_line=outputs/sam2_floor_feature_instance_masks_with_outline#3' \
  --mask-source 'goal_crease=outputs/sam2_floor_feature_instance_masks_with_outline#4' \
  --mask-source 'field_outline=outputs/sam2_floor_feature_instance_masks_with_outline#5' \
  --refine-mode primitive_full_homography \
  --primitive-sample-stride 4 \
  --regularization 0.035 \
  --max-nfev 240 \
  --min-features 1 \
  --max-single-line-rotation-deg 18 \
  --max-single-line-translation-px 120

python scripts/render_camera_homography_overlay.py \
  --calibration outputs/floor_homography_sam2_landmarks_primitive_full_segment_with_outline_dynamic.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/camera_floor_homography_overlay_sam2_landmarks_primitive_full_segment_with_outline_h264.mp4 \
  --draw-landmark-masks \
  --landmark-mask-alpha 0.45 \
  --fps 10

python scripts/render_birds_eye_locations.py \
  --calibration outputs/floor_homography_sam2_landmarks_primitive_full_segment_with_outline_dynamic.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/birds_eye_player_locations_sam2_landmarks_primitive_full_segment_with_outline_h264.mp4 \
  --output-json outputs/birds_eye_player_locations_sam2_landmarks_primitive_full_segment_with_outline.json \
  --draw-landmark-mask-points \
  --fps 10
```

The primitive SAM2 landmark pass fits image-space line primitives and visible line segments directly from SAM2 masks, filters the outline mask to the yellow lower rink envelope, and fits the visible crease as an arc segment. `primitive_full_homography` refines the full projective transform when multiple primitives are visible; frames with only one line use a bounded line-specific correction. `--draw-landmark-masks` overlays the SAM2 camera-space floor landmark segmentations on the camera video; `--draw-landmark-mask-points` projects sampled SAM2 landmark-mask pixels onto the top-down floor view.

## 10. VGGT-Omega + SAM3 + SAM-Body4D Visualization

This is the reproducible path for the 3D visualization. SAM3 text masks give players and field evidence, VGGT-Omega gives cameras and depth, the field is fitted to VGGT-Omega's floor plane (see [docs/field_plane_fit.md](docs/field_plane_fit.md)), and SAM-Body4D gives human meshes.

Install the external repos locally. The VGGT-Omega checkpoint is gated on Hugging Face (`facebook/VGGT-Omega`); put `vggt_omega_1b_512.pt` under `checkpoints/vggt_omega/`:

```bash
git clone https://github.com/facebookresearch/vggt-omega third_party/vggt_omega
git clone https://github.com/facebookresearch/vggt third_party/VGGT          # geometry helpers used by the VGGT-layout consumers
git clone https://github.com/gaomingqi/sam-body4d third_party/sam-body4d
python -m pip install -r third_party/vggt_omega/requirements.txt
```

Prepare a frame directory from a local source video. One frame per second is a good first pass for Week 1. Run SAM3 text tracking on the original frames: the `player` masks drive SAM-Body4D and CoTracker; the field masks drive the field fit.

```bash
export RAW=data/week1_112825_osh_tor_0820_1053_frames_1fps
export RAW_SAM3=outputs/sam3/week1_112825_osh_tor_0820_1053_1fps
ffmpeg -ss 00:08:20 -to 00:10:53 -i "data/Week 1/112825_osh_tor.mp4" -vf fps=1 "$RAW/frame_%05d.jpg"

for prompt in player 'green field' 'white lines on green field' 'yellow outline around field border'; do
  name=$(echo "$prompt" | cut -d' ' -f1-2 | tr ' ' '_')
  HF_TOKEN="$HF_TOKEN" python scripts/run_sam3_text_chunked.py --frames-dir "$RAW" --text "$prompt" \
    --output "$RAW_SAM3/${name}_tracks_chunked.mp4" --mask-dir "$RAW_SAM3/${name}_label_masks_chunked" \
    --instance-mask-dir "$RAW_SAM3/${name}_instance_masks_chunked" --fps 1
done
```

Remove broadcast lens distortion (radial k1 estimated from the straightness of the SAM3 white lines), remap the masks, and re-box the player detections:

```bash
export FRAMES=data/week1_112825_osh_tor_0820_1053_frames_1fps_undistorted
export SAM3=outputs/sam3/week1_112825_osh_tor_0820_1053_1fps_undistorted
export RUN=outputs/vggt_omega/week1_1fps_undistorted

python scripts/undistort_clip_from_text_line_masks.py \
  --frames-dir "$RAW" --white-mask-dir "$RAW_SAM3/white_lines_instance_masks_chunked" \
  --output-frames-dir "$FRAMES" \
  --calibration-json outputs/vggt/week1_112825_osh_tor_0820_1053_undistorted/lens_undistortion_text_masks.json \
  $(for n in player green_field white_lines yellow_outline; do echo --npz-mask-map "$RAW_SAM3/${n}_instance_masks_chunked:$SAM3/${n}_instance_masks_chunked"; done)

python scripts/rebox_sam3_json_from_instance_masks.py \
  --source-json "$RAW_SAM3/player_tracks_chunked.json" --instance-mask-dir "$SAM3/player_instance_masks_chunked" \
  --frames-dir "$FRAMES" --output-json "$SAM3/player_tracks_chunked_reboxed.json"
```

Run VGGT-Omega (about 45 s and 19 GB of GPU memory for 153 frames at 1280x720):

```bash
python scripts/run_vggt_omega_clip.py --frames-dir "$FRAMES" --output "$RUN/vggt_omega_predictions.npz"
```

Fit the NLL field to the VGGT-Omega floor plane, then refine the broadcast camera against it. The landmark masks come from `build_field_paint_masks.py`: painted lines and board edges detected by colour, with labels proposed from a fit of the clip and checked by eye on the review sheets it writes (`$RUN/paint_masks/review/sheet_*.jpg`). Corrections go in a review JSON keyed by frame and candidate id; the checked ones for both clips are in `data/field_label_review/`. A bootstrap fit from the SAM3 masks (`--landmark white_lines=... --landmark field_outline=... --detect-white-paint`) is enough to propose labels; without `--fit` every candidate starts unlabeled.

```bash
python scripts/build_field_paint_masks.py \
  --frames-dir "$FRAMES" --predictions "$RUN/vggt_omega_predictions.npz" \
  --player-mask-dir $SAM3/player_instance_masks_chunked \
  --fit "$RUN/field_fit/field_plane_fit.json" --review data/field_label_review/week1_1fps_undistorted.json \
  --output-dir "$RUN/paint_masks"

python scripts/fit_field_plane_landmarks.py \
  --predictions "$RUN/vggt_omega_predictions.npz" --frames-dir "$FRAMES" \
  --paint-masks "$RUN/paint_masks" --fps 1 --output-dir "$RUN/field_fit"
```

For the 10 s clip (purple end zones, SAM2 faceoff-dot masks added as extra landmarks):

```bash
RUN10=outputs/vggt_omega/clip10s_10fps_undistorted
DM=outputs/sam2/floor_features/sam2_faceoff_corner_instance_masks_undistorted
python scripts/build_field_paint_masks.py \
  --frames-dir data/frames_10fps_undistorted --predictions $RUN10/vggt_omega_predictions.npz \
  --player-mask-dir outputs/sam3/text/sam3_text_player_instance_masks_undistorted --floor-colours green,purple \
  --fit $RUN10/field_fit/field_plane_fit.json --review data/field_label_review/clip10s_10fps_undistorted.json \
  --output-dir $RUN10/paint_masks
python scripts/fit_field_plane_landmarks.py \
  --predictions $RUN10/vggt_omega_predictions.npz --frames-dir data/frames_10fps_undistorted \
  --paint-masks $RUN10/paint_masks --landmark faceoff_right_upper=$DM#6 --landmark faceoff_right_lower=$DM#7 \
  --output-dir $RUN10/field_fit
```

The field model (`field_model.py`) follows the NLL rulebook; the corner radius of the boards is estimated per arena (`field_dims` in the output).

Outputs: `field_plane_fit.json`, `field_homographies.json` (per-frame image->field homographies, readable by `render_birds_eye_locations.py --calibration` and `render_camera_homography_overlay.py --calibration`), `field_reprojection_h264.mp4` and `landmarks_on_field.jpg`.

Render the VGGT-Omega reconstruction in 3D on the fitted field: an orbit that ends top-down, and a per-frame timeline with the input frame, an oblique 3D view and a top-down view. Points are placed along the refined cameras' rays at VGGT-Omega's heights above the floor, so the floor paint lands on the fitted lines; `--geometry raw` shows VGGT-Omega's own geometry instead:

```bash
python scripts/render_vggt_scene_3d.py \
  --predictions "$RUN/vggt_omega_predictions.npz" --field-fit "$RUN/field_fit/field_plane_fit.json" \
  --frames-dir "$FRAMES" --output-dir "$RUN/scene_3d" --timeline-fps 3
```

The SAM-Body4D fusion, the `viser` viewer and the CoTracker scripts read the 518x518 VGGT layout. Export VGGT-Omega and the rigid planar fit into that layout:

```bash
python scripts/export_vggt_omega_vggt_layout.py \
  --predictions "$RUN/vggt_omega_predictions.npz" --field-fit "$RUN/field_fit/field_plane_fit.json" \
  --frames-dir "$FRAMES" --output-dir "$RUN/vggt_layout"
export VGGT_NPZ="$RUN/vggt_layout/vggt_predictions_compact.npz"
export FIELD_FIT="$RUN/vggt_layout/field_fit_vggt_layout.json"
```

Run SAM-Body4D from the SAM3 `player` boxes:

```bash
PYOPENGL_PLATFORM=egl python scripts/run_sam_body4d_from_sam3_boxes.py \
  --repo-root third_party/sam-body4d \
  --config third_party/sam-body4d/configs/body4d.yaml \
  --frames-dir "$FRAMES" \
  --sam3-json "$SAM3/player_tracks_chunked_reboxed.json" \
  --output-dir "$RUN/sam_body4d" \
  --output-video "$RUN/sam_body4d_overlay_h264.mp4" \
  --render-mode overlay \
  --fps 1
```

Render top-down H.264 review videos with SAM-Body4D mesh points and SAM3 field-mask points:

```bash
python scripts/render_sam_body4d_vggt_birds_eye.py \
  --vggt-repo third_party/VGGT \
  --vggt-json "$FIELD_FIT" \
  --vggt-npz "$VGGT_NPZ" \
  --frames-dir "$FRAMES" \
  --sam3-json "$SAM3/player_tracks_chunked_reboxed.json" \
  --mesh-root "$RUN/sam_body4d/mesh_4d_individual" \
  --focal-root "$RUN/sam_body4d/focal_4d_individual" \
  --floor-mask-dir "$SAM3/green_field_instance_masks_chunked" \
  --text-green-mask-dir "$SAM3/green_field_instance_masks_chunked" \
  --text-yellow-mask-dir "$SAM3/yellow_outline_instance_masks_chunked" \
  --text-white-mask-dir "$SAM3/white_lines_instance_masks_chunked" \
  --output-dir "$RUN/fused_birds_eye" \
  --min-depth-conf 1.0 \
  --fixed-field-player-size \
  --fps 1
```

Open the interactive `viser` viewer:

```bash
python scripts/view_vggt_reconstruction.py \
  --frames-dir "$FRAMES" \
  --predictions "$VGGT_NPZ" \
  --floor-fit-json "$FIELD_FIT" \
  --mesh-root "$RUN/sam_body4d/mesh_4d_individual" \
  --focal-root "$RUN/sam_body4d/focal_4d_individual" \
  --conf-threshold 20 \
  --fixed-field-player-size \
  --no-ground-meshes-to-field \
  --port 8097
```

Open `http://127.0.0.1:8097`. Use `--hide-vggt-points` if you want the viewer to start with only the field and SAM-Body4D meshes visible.

## 11. VGGT + CoTracker Player Trajectories

This path uses SAM3 player masks to seed CoTracker point tracks, then projects those tracked player points through VGGT depth and the fitted synthetic field. It writes a camera-view track overlay and a bird's-eye trajectory video.

Install CoTracker locally:

```bash
git clone https://github.com/facebookresearch/co-tracker third_party/co-tracker
```

Run trajectory projection from the VGGT-Omega stack of section 10 (`$VGGT_NPZ` and `$FIELD_FIT` from `export_vggt_omega_vggt_layout.py`):

```bash
python scripts/run_cotracker_vggt_player_trajectories.py \
  --frames-dir "$FRAMES" \
  --sam3-json "$SAM3/player_tracks_chunked_reboxed.json" \
  --player-mask-dir "$SAM3/player_instance_masks_chunked" \
  --vggt-npz "$VGGT_NPZ" \
  --field-fit-json "$FIELD_FIT" \
  --output-dir "$RUN/cotracker_trajectories" \
  --fps 1 \
  --points-per-object 3 \
  --min-object-frames 10 \
  --min-depth-conf-percentile 20 \
  --require-sam3-presence \
  --ignore-cotracker-visibility \
  --disable-mask-gate
```

`--require-sam3-presence` keeps trajectories tied to frames where SAM3 still detects that object ID. `--ignore-cotracker-visibility` is useful on broadcast footage because CoTracker's boolean visibility can be very conservative even when the point path remains visually usable.

For a true source-fps run, first extract a short clip at the broadcast frame rate and run SAM3 on those exact frames (`run_sam3_text_chunked.py --text player`, outputs under `$SRC/sam3/`). Then run CoTracker on the source-fps SAM3 masks:

```bash
export SRC=outputs/vggt/cotracker/week1_0820_0825_sourcefps
python scripts/run_cotracker_sourcefps_player_masks.py \
  --frames-dir data/week1_112825_osh_tor_0820_0825_frames_source_fps \
  --sam3-json "$SRC/sam3/player.json" \
  --player-mask-dir "$SRC/sam3/player_instance_masks" \
  --output-dir "$SRC/cotracker_120f" \
  --fps 59.94005994 \
  --max-frames 120 \
  --points-per-object 1 \
  --min-object-frames 5 \
  --require-sam3-presence \
  --ignore-cotracker-visibility
```

Open the source-fps tracks interactively in Viser:

```bash
python scripts/view_cotracker_sourcefps_tracks.py \
  --frames-dir data/week1_112825_osh_tor_0820_0825_frames_source_fps \
  --tracks-json "$SRC/cotracker_120f/cotracker_sourcefps_player_tracks.json" \
  --port 8111 \
  --fps 59.94005994
```

To render the trajectory JSON at the original video frame rate, interpolate the CoTracker/VGGT positions onto the source-video timeline:

```bash
python scripts/render_cotracker_trajectories_source_fps.py \
  --trajectory-json "$RUN/cotracker_trajectories/cotracker_vggt_player_trajectories.json" \
  --output-dir "$RUN/cotracker_trajectories_source_fps" \
  --input-fps 1 \
  --source-video "data/Week 1/112825_osh_tor.mp4" \
  --source-start-sec 500 \
  --duration-sec 153 \
  --output-fps 60000/1001
```

This keeps the VGGT reconstruction and SAM3 masks at their analysis cadence, but writes smooth H.264 camera and bird's-eye review videos at the broadcast frame rate.

## Output Video Format

All rendered tracking videos are written with `libx264` via `imageio-ffmpeg`:

- codec: H.264
- pixel format: `yuv420p`
- frame size: 1280x720
- frame rate: 10 fps

This makes the MP4 files viewable in VS Code.

## Script Summary

- `scripts/prepare_lacrosse_clip.py`: samples the source video, writes frames, preview video, and prompt images.
- `scripts/click_annotator.py`: local web UI for collecting full-resolution player clicks.
- `scripts/track_sam2_clicks.py`: runs SAM2 video tracking from click prompts and can export per-object mask stacks.
- `scripts/track_sam3_text.py`: runs SAM3 video tracking from a text prompt.
- `scripts/classify_sam3_teams_by_torso_color.py`: assigns team colors from masked torso color composition and smooths labels by SAM3 object ID.
- `scripts/classify_sam3_teams_by_transreid.py`: assigns team/goalkeeper clusters from TransReID embeddings and smooths labels by SAM3 object ID.
- `scripts/floor_homography_annotator.py`: local web UI for clicking floor landmarks with known world coordinates.
- `scripts/floor_free_click_annotator.py`: local web UI for unlabeled floor landmark clicks.
- `scripts/floor_feature_annotator.py`: local web UI for coarse floor-feature clicks.
- `scripts/floor_feature_clicks_to_sam2_prompts.py`: converts coarse floor-feature clicks into SAM2 object prompts.
- `scripts/fit_floor_homography_from_feature_clicks.py`: fits homography from coarse feature-labeled clicks, including the arc-segment crease model.
- `scripts/fit_floor_homography_from_tracked_landmarks.py`: refines per-frame homographies from SAM2/SAM-style tracked landmark masks using line, visible-segment, crease, and outline primitives.
- `scripts/estimate_dynamic_floor_homographies.py`: estimates per-frame camera motion and writes one floor homography per frame.
- `scripts/refine_floor_homography_from_unlabeled_clicks.py`: refines a homography by treating free clicks as points on modeled floor features.
- `scripts/render_camera_homography_overlay.py`: projects the fitted floor model and player floor-contact points back onto the camera video for calibration debugging.
- `scripts/render_birds_eye_locations.py`: projects SAM3 player floor points through homography and writes a top-down player-location MP4.
- `scripts/run_vggt_omega_clip.py`: runs VGGT-Omega on a frame directory and saves cameras, depth, confidence and colours.
- `scripts/fit_field_plane_landmarks.py`: fits the NLL field to the VGGT-Omega floor plane from landmark masks, refines the broadcast camera, and writes per-frame image->field homographies (see `docs/field_plane_fit.md`).
- `scripts/build_field_paint_masks.py`: detects painted lines and board edges by colour, proposes labels from a field fit, applies a checked review JSON, and writes landmark masks and review sheets for the field fit.
- `scripts/field_model.py`: parametric NLL floor (rulebook markings, per-arena corner radius): landmark distances, curves and drawing.
- `scripts/render_vggt_scene_3d.py`: renders the VGGT-Omega point cloud on the fitted field as H.264 videos (orbit to top-down, and a per-frame timeline).
- `scripts/export_vggt_omega_vggt_layout.py`: writes VGGT-Omega predictions and the rigid field fit in the 518x518 VGGT layout used by the SAM-Body4D fusion, the viewer and the CoTracker scripts.
- `scripts/nll_field_geometry.py`: NLL floor dimensions and line, crease and outline samplers.
- `scripts/run_sam3_text_chunked.py`: runs SAM3 text tracking in chunks for long clips.
- `scripts/undistort_clip_from_text_line_masks.py`, `scripts/undistort_clip_from_field_lines.py`: estimate radial lens distortion from straight field lines, write undistorted frames and remap masks.
- `scripts/rebox_sam3_json_from_instance_masks.py`: recomputes SAM3 detection boxes from (remapped) instance masks.
- `scripts/render_sam_body4d_vggt_birds_eye.py`: places SAM-Body4D meshes on VGGT player points and writes top-down review videos.
- `scripts/view_vggt_reconstruction.py`: `viser` viewer for the VGGT point cloud, fitted field and SAM-Body4D meshes.
- `scripts/run_vggt_reconstruction_clip.py`, `scripts/run_vggt_birds_eye.py`: VGGT (non-Omega) reconstruction and plane-homography bird's-eye path. `run_vggt_birds_eye.py` also provides shared helpers to the depth-based scripts above.
- `scripts/run_cotracker_vggt_player_trajectories.py`: seeds CoTracker point tracks from SAM3 player masks and projects the resulting tracks through VGGT into bird's-eye floor coordinates.
- `scripts/run_cotracker_sourcefps_player_masks.py`: runs CoTracker directly on source-fps SAM3 player masks and writes source-fps camera trajectories.
- `scripts/view_cotracker_sourcefps_tracks.py`: opens a Viser timeline for source-fps CoTracker player tracks.
- `scripts/render_cotracker_trajectories_source_fps.py`: interpolates CoTracker/VGGT trajectory JSON onto the source-video timeline and writes source-fps H.264 review videos.
- `scripts/merge_sam3_team_detections.py`: optional older path that merges black/white SAM3 prompt detections and assigns team colors using mask-average color for ties.
- `scripts/run_sam_body4d_from_masks.py`: runs SAM-Body4D from exported label masks.
- `scripts/run_sam_body4d_from_sam3_boxes.py`: runs SAM-Body4D from SAM3 text detections, preserving multiple players.
- `scripts/run_4d_humans_from_sam3_boxes.py`: runs 4D-Humans/HMR2 from SAM3 text detections and overlays SMPL meshes on the camera video.
- `scripts/video_utils.py`: shared frame loading, mask overlay, coordinate grid, and H.264 writing helpers.
- `scripts/faceoff_clicks_to_sam2_prompts.py`: converts faceoff-dot clicks into SAM2 prompts.
- `scripts/render_field_landmark_masks_debug.py`, `scripts/render_4dhumans_own_birds_eye.py`: debug renderers.

## Notes

- The current workflow uses 100 frames: 10 seconds at 10 fps.
- SAM2 click tracking is usually the corrective pass for exact player coverage.
- SAM3 text tracking is useful as an automatic baseline, but may miss early frames or create extra detections.
- The SAM-Body4D box pipeline uses SAM3 text boxes as rectangular person prompts. This avoids collapsed single-ID label masks and produces separate meshes per detected player, but early frames can be sparse if SAM3 detects few players.
- If `kernels` causes import errors with Transformers, uninstall it:

```bash
python -m pip uninstall -y kernels kernels-data
```
