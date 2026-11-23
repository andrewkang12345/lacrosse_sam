# Week 1 at 10 fps: time estimate

Processing the Week 1 segment (08:20–10:53, 153 s) at the 10 s clip's frame rate means 1,530 frames instead of 153.
Producing every current output at 10 fps takes about a working day:

- **Compute:** about 3–3.5 h, unattended.
- **Code changes:** about 2–3 h, needed before the run.
- **Review by eye:** about 1–2 h.

Timings are from runs on this machine (NVIDIA A10G, 23 GB), measured on 2026-11-21.

## Compute

| step | measured | at 10 fps (1,530 frames) |
|---|---|---|
| frame extraction | 1 s for 100 frames | ~1 min |
| SAM3 `player` tracking (`run_sam3_text_chunked.py`) | 286 s for 100 frames of Week 1 at 10 fps (~1.5 s/frame plus ~30 s model reload per 45-frame chunk) | ~55 min |
| lens undistortion | — | a few minutes (the Week 1 lens calibration is reused) |
| VGGT-Omega (`run_vggt_omega_clip.py`) | 22 s / 14.5 GB for 100 frames; 43 s / 18.8 GB for 153 frames | ~10–15 min in ~11 overlapping chunks of ≤ 200 frames |
| paint masks (`build_field_paint_masks.py`) | 66 s for 153 frames | ~11 min |
| field fit (`fit_field_plane_landmarks.py`) | 205 s for 100 frames at 10 fps, with background points tracked between frames; 266 s for 153 frames at 1 fps | ~55 min for chunks in sequence, ~15–20 min in parallel |
| 3D orbit and timeline (`render_vggt_scene_3d.py`) | 630 s for 153 frames | ~1–1.5 h |
| teams and bird's-eye (`classify_sam3_teams_by_torso_color.py`, `render_field_birds_eye.py`) | under a minute for 153 frames | ~10 min |

Only the `player` prompt has to be run again. The other three SAM3 prompts are not needed:

- **Lens correction.** The Week 1 lens calibration already exists and is reused.
- **Field fit.** The fit uses the colour-detected paint masks, not the SAM3 field masks.

The table leaves out SAM-Body4D meshes and CoTracker trajectories, which were not timed.

## Code changes needed first

1. **VGGT-Omega in chunks.**
   - Memory grows by about 0.08 GB per frame on top of about 6.4 GB, so one pass holds about 200 frames.
   - The clip is reconstructed in overlapping chunks, and each chunk is fitted to the field.
   - The per-frame cameras, homographies and 3D renders are then merged into one clip. All of them are in field feet, so the chunks need no alignment with each other.
2. **Label proposals for the new frames.** `build_field_paint_masks.py --fit` proposes labels from a fit of the same frames. For the 10 fps frames, cameras are interpolated from the 1 fps fit: one shared camera centre, plus per-frame pan, tilt, roll and zoom.
3. **Linking SAM3 tracks across chunks.**
   - SAM3 restarts its track IDs every 45 frames: the 100-frame test produced 70 tracks, so 1,530 frames give about 780.
   - Linking tracks by mask overlap at chunk boundaries cuts that to a few hundred for the team review.

## Review by eye

- **Paint labels.** 1,530 frames are about 170 review sheets. Every 10th frame matches a 1 fps frame that is already checked in `data/field_label_review/`. The review covers a sample of the others plus every frame the fit flags (high error, `own_camera` or `unreliable`).
- **Automatic checks.** At 10 fps the fit ties neighbouring cameras to tracked background motion, and `camera_motion_check` lists every frame where the fitted camera still disagrees with that motion by more than 5 px. The bird's-eye JSON lists the frames where SAM3 shifted IDs between players (`cleaning`); those are interpolated. Only the listed frames need to be looked at.
- **Teams.** One crop per linked track, with corrections written to `data/team_review/`.

## Faster option: bird's-eye and field overlay only

If only the bird's-eye and field-overlay videos are needed at 10 fps, VGGT-Omega and the 3D renders stay at 1 fps:

- **Camera fit.** The 1 fps fit's camera centre and field dimensions are kept. Each 10 fps frame gets only its pan, tilt, roll and zoom, fitted to its paint masks and starting from the neighbouring 1 fps cameras.
- **Time.** About 1.5–2 h of compute, roughly half of the full version, plus the code change for the per-frame camera fit and the same review.
- **Cost.** The 3D point-cloud videos stay at 1 fps.
