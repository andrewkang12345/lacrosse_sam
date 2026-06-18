# Lacrosse SAM Tracking

Small workflow for segmenting and tracking lacrosse players in a 10 second clip using:

- SAM3 text prompts through Hugging Face Transformers
- SAM2 click prompts through Hugging Face Transformers
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
  --fps 10
```

Outputs:

```bash
outputs/sam3_text_lacrosse_player.mp4
outputs/sam3_text_lacrosse_player.json
```

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
- `scripts/track_sam2_clicks.py`: runs SAM2 video tracking from click prompts.
- `scripts/track_sam3_text.py`: runs SAM3 video tracking from a text prompt.
- `scripts/video_utils.py`: shared frame loading, mask overlay, coordinate grid, and H.264 writing helpers.

## Notes

- The current workflow uses 100 frames: 10 seconds at 10 fps.
- SAM2 click tracking is usually the corrective pass for exact player coverage.
- SAM3 text tracking is useful as an automatic baseline, but may miss early frames or create extra detections.
- If `kernels` causes import errors with Transformers, uninstall it:

```bash
python -m pip uninstall -y kernels kernels-data
```
