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

The workflow samples the first 10 seconds at 10 fps into:

```bash
data/frames_10fps/frame_*.jpg
```

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

The feature fitter models the goal crease as a front semicircle plus a rear rectangle at each goal, not as a pure circle.
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
  --feature-clicks outputs/floor_feature_clicks_no_outline.json \
  --output outputs/floor_feature_sam2_prompts_no_outline.json

python scripts/track_sam2_clicks.py \
  --frames-dir data/frames_10fps \
  --prompts outputs/floor_feature_sam2_prompts_no_outline.json \
  --model facebook/sam2.1-hiera-large \
  --output outputs/sam2_floor_feature_tracks_no_outline_h264.mp4 \
  --mask-dir outputs/sam2_floor_feature_label_masks_no_outline \
  --instance-mask-dir outputs/sam2_floor_feature_instance_masks_no_outline \
  --fps 10

python scripts/fit_floor_homography_from_tracked_landmarks.py \
  --frames-dir data/frames_10fps \
  --initial-calibration outputs/floor_homography_feature_no_outline_dynamic.json \
  --feature-clicks outputs/floor_feature_clicks_no_outline.json \
  --output outputs/floor_homography_sam2_landmarks_no_outline_dynamic.json \
  --mask-source 'left_restraining_line=outputs/sam2_floor_feature_instance_masks_no_outline#1' \
  --mask-source 'right_restraining_line=outputs/sam2_floor_feature_instance_masks_no_outline#2' \
  --mask-source 'midfield_line=outputs/sam2_floor_feature_instance_masks_no_outline#3' \
  --mask-source 'goal_crease=outputs/sam2_floor_feature_instance_masks_no_outline#4' \
  --exclude-features field_outline \
  --min-features 1

python scripts/render_camera_homography_overlay.py \
  --calibration outputs/floor_homography_sam2_landmarks_no_outline_dynamic.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/camera_floor_homography_overlay_sam2_landmarks_masks_no_outline_h264.mp4 \
  --draw-landmark-masks \
  --landmark-mask-alpha 0.45 \
  --fps 10

python scripts/render_birds_eye_locations.py \
  --calibration outputs/floor_homography_sam2_landmarks_no_outline_dynamic.json \
  --sam3-json outputs/sam3_team_transreid_3clusters_detections.json \
  --frames-dir data/frames_10fps \
  --instance-mask-dir outputs/sam3_text_player_instance_masks \
  --output-video outputs/birds_eye_player_locations_sam2_landmarks_masks_no_outline_h264.mp4 \
  --output-json outputs/birds_eye_player_locations_sam2_landmarks_masks_no_outline.json \
  --draw-landmark-mask-points \
  --fps 10
```

The current no-outline SAM2 landmark pass refines 82 of 100 frames from tracked landmark masks and falls back to the prior dynamic calibration on the remaining frames. `--draw-landmark-masks` overlays the SAM2 camera-space floor landmark segmentations on the camera video; `--draw-landmark-mask-points` projects sampled SAM2 landmark-mask pixels onto the top-down floor view.

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
- `scripts/fit_floor_homography_from_feature_clicks.py`: fits homography from coarse feature-labeled clicks, including the semicircle-plus-rectangle crease model.
- `scripts/fit_floor_homography_from_tracked_landmarks.py`: refines per-frame homographies from SAM2/SAM-style tracked landmark masks.
- `scripts/estimate_dynamic_floor_homographies.py`: estimates per-frame camera motion and writes one floor homography per frame.
- `scripts/refine_floor_homography_from_unlabeled_clicks.py`: refines a homography by treating free clicks as points on modeled floor features.
- `scripts/render_camera_homography_overlay.py`: projects the fitted floor model and player floor-contact points back onto the camera video for calibration debugging.
- `scripts/render_birds_eye_locations.py`: projects SAM3 player floor points through homography and writes a top-down player-location MP4.
- `scripts/merge_sam3_team_detections.py`: optional older path that merges black/white SAM3 prompt detections and assigns team colors using mask-average color for ties.
- `scripts/run_sam_body4d_from_masks.py`: runs SAM-Body4D from exported label masks.
- `scripts/run_sam_body4d_from_sam3_boxes.py`: runs SAM-Body4D from SAM3 text detections, preserving multiple players.
- `scripts/run_4d_humans_from_sam3_boxes.py`: runs 4D-Humans/HMR2 from SAM3 text detections and overlays SMPL meshes on the camera video.
- `scripts/video_utils.py`: shared frame loading, mask overlay, coordinate grid, and H.264 writing helpers.

## Notes

- The current workflow uses 100 frames: 10 seconds at 10 fps.
- SAM2 click tracking is usually the corrective pass for exact player coverage.
- SAM3 text tracking is useful as an automatic baseline, but may miss early frames or create extra detections.
- The SAM-Body4D box pipeline uses SAM3 text boxes as rectangular person prompts. This avoids collapsed single-ID label masks and produces separate meshes per detected player, but early frames can be sparse if SAM3 detects few players.
- If `kernels` causes import errors with Transformers, uninstall it:

```bash
python -m pip uninstall -y kernels kernels-data
```
