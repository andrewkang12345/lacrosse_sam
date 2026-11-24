# Fitting the NLL field to VGGT-Omega through its floor plane

`scripts/fit_field_plane_landmarks.py` fits the NLL field to a VGGT-Omega reconstruction. It
treats the reconstruction as what it is for a lacrosse floor: one plane seen by one broadcast
camera. Landmarks are fitted in that plane, and the camera is then corrected against the known
field. It needs three things to be right:

- **Landmarks that are what they claim to be.** These are painted lines and board edges, detected
  in every frame, labeled, and checked by eye (`scripts/build_field_paint_masks.py`).
- **A field model that matches the painted floor.** This is the NLL rulebook geometry, with the
  arena's corner radius estimated (`scripts/field_model.py`).
- **A camera model that matches a broadcast camera.** This is one camera position with
  per-frame pan, tilt, roll and zoom.

The 3D renders place VGGT-Omega's points through the same refined cameras, so the floor paint in
the top-down view lands on the fitted lines.

| clip | frames | median reprojection error | 90th percentile |
|---|---|---|---|
| 10 s, San Diego | 100 at 10 fps | 2.9 px | 8.4 px |
| Week 1, Oshawa: keyframes | 153 at 1 fps | 1.8 px | 5.4 px |
| Week 1, Oshawa | 1,530 at 10 fps | 1.7 px | 5.2 px |

Error is the image-space distance from every landmark pixel to the projected model curve of
*its own* landmark: midfield line, each restraining line, each crease, the far board base, and
for the 10 s clip the two SAM2-tracked faceoff dots.

![10 s clip: the fitted field drawn on four frames](figures/clip10s_field_overlay.jpg)
![Week 1: the fitted field drawn on four frames](figures/week1_field_overlay.jpg)

## Why one plane and one camera

**VGGT-Omega's frames disagree with each other.** Every frame's floor is very planar on its own
(0.4% rms off-plane in the 10 s clip), but the floor offset and the camera height drift by 14–24%
across frames. Unprojecting each frame's landmark pixels with its own depth and camera puts the
right restraining line from different frames up to 11 ft apart: a band, not a line. Casting rays
through VGGT-Omega's cameras onto one plane gives the same spread, so the cameras themselves have
to be corrected. That is possible only when a lifted point depends on the camera alone, not on
depth.

**The reconstruction is projectively distorted.** For zoomed broadcast footage, VGGT-Omega confuses
zoom with dolly. In the 10 s clip it puts the camera about 290 ft from the play with a 9° field of
view, and slides it about 90 ft along its viewing direction while the operator zooms. The refined
camera is 180 ft away with a 16° field of view. A camera that is too far and too telephoto flattens
perspective, so the reconstructed floor is compressed more on the far side than the near side.
This distortion differs per frame, so no single similarity, affine or plane homography matches
every landmark in every frame.

**The field repeats.** NLL lines repeat every 42.5 ft (restraining, midfield, restraining), and the
two ends of the floor look alike. Matching pixels to their nearest model line lets a solution
shifted by one spacing, or turned end for end, score about as well as the right one. The fit
therefore uses labeled landmarks, allows only proper similarities (a camera looking down cannot see
a mirrored floor), and does not free a plane homography, which extrapolates wildly when the
landmarks cover part of the floor.

## How the fit works

### Landmark masks

SAM3's text masks cover only part of the visible lines and do not say *which* line a pixel belongs
to. `scripts/build_field_paint_masks.py` detects the landmarks in the frame itself and labels them
explicitly:

1. **Floor.** Green turf by colour, plus purple end-zone turf where it touches the green floor
   (the San Diego arena). Seats and banners of the same colour are dropped because they are not
   connected to the floor.
2. **Paint candidates.** Thin, bright, unsaturated structures on the floor are found by a white
   top-hat filter. Players are removed with their SAM3 masks. Each connected piece is a candidate
   with a stable id.
3. **Far board base.** In each column, this is the first row of solid turf below non-turf: where
   the far boards meet the floor. It is smoothed across columns and dropped next to players and
   next to paint (a crease touching the boards). It is also dropped where a thick yellow band
   marks a bench's line-change area rather than the kick plate.
4. **Labels.** Each candidate is proposed the landmark whose projected curve it lies on under a
   fit of the clip (`--fit`). A proposal needs the projected curve to be close and unambiguous.
   For a line it also needs a straight piece running in the projected line's direction. Logo
   artwork next to the midfield line fails that test.
5. **Review by eye.** Every frame of the 10 s clip and of the Week 1 keyframes is checked on the
   review sheets the script writes (`paint_masks/review/sheet_*.jpg`; the figure below is one of
   them).
   - Corrections are kept in `data/field_label_review/*.json`, keyed by frame and candidate id.
   - Week 1 has 37 corrections in 17 of 153 keyframes. These are mostly restraining-line pieces
     the proposal misses, and pieces in close-ups that cannot be confirmed and are set to
     unlabeled.
   - The 10 s clip has 36 corrections in 34 of 100 frames. These are mostly the zoomed frames at
     the start, where the midfield line shows only as a short stub between the boards and the logo.
   - Proposed from the final fit, 941 of the 942 labeled pieces keep their label. The final fit
     also proposes 10 Week 1 pieces that the review keeps unlabeled (stick shafts, logo artwork
     and close-ups behind the crease).
6. **Full frame rate.** At 10 fps, keyframes with at least two painted lines keep their checked
   masks (`--checked-masks`), and every other frame is labeled from cameras propagated from the
   keyframes (below).

![one review sheet: labeled paint, unlabeled candidates with ids, far board base in yellow](figures/week1_paint_review_sheet.jpg)

The landmarks are per side and per type: `midfield_line`, `left/right_restraining_line`,
`left/right_crease_lines` (the crease arc, its straight back and the goal line across it), and
`board_base`.

The near boards are not used. The broadcast camera looks over them, so the visible turf ends at
their top edge, about 4 ft up, not at the floor. Logos and floor ads that reach the bottom of the
image produce the same kind of edge. With these edges in the fit, the 10 s camera moves to 245 ft
behind the sideline and 20 of 100 frames become `unreliable`.

### Field model

`scripts/field_model.py` follows the NLL rulebook (2019-20, rules 1.5, 2.1, 3.1 and 6.2), and its
distances are analytic.

- **No centre faceoff circle.** Neither arena paints one: both have a team logo there.
  `--centre-circle-radius-ft` adds one for an arena that does.
- **The crease as painted.** A 9 ft 3 in arc centred on the goal-line midpoint, a straight back
  5 ft 6 in behind the goal line (1 ft behind the 4 ft 6 in goal base), and the goal line painted
  across the full crease.
- **Goals tied to the end boards.** The goal lines are 12 ft from the end boards, the restraining
  lines 42 ft 6 in from the centre line, and the corner faceoff spots 15 ft from the side boards
  and 15 ft outside the restraining lines.
- **The corner radius is estimated per arena.** The boards are those of the hockey rink under
  the floor, and rinks differ. The radius is a parameter of the camera refinement, with a soft
  prior towards 22 ft 8 in. The fit puts it at 25.2 ft in San Diego and 26.9 ft in Oshawa
  (27.3 ft at 10 fps). The 85 ft width stays fixed as the scale.
- **The floor length is fixed at the 200 ft rink standard.** No frame shows both ends, so the
  length trades off against camera distance and zoom. Freed on the 10 s clip with the faceoff dots
  left out, it goes to 209 ft, and the dots land 1.5 and 3.1 ft off, against 1.1 and 2.4 ft with
  the length fixed. `--estimate-dims half_length,corner_radius` frees it as a diagnostic.
- The fitted dimensions are written as `field_dims` in `field_plane_fit.json` and
  `field_homographies.json`. Every overlay and the 3D renders draw the field with them.

### Stage 1: a rigid planar fit

1. **Floor plane.** Fit a RANSAC plane, then polish it with least squares, on confident VGGT-Omega
   depth points on green turf. Players and boards are excluded by colour. The normal is oriented
   towards the cameras.
2. **Ray–plane lifting.** Every landmark pixel is lifted by intersecting its camera ray with that
   single plane. Per-pixel depth is never used, so depth noise cannot smear the landmarks, and each
   lifted point depends on the camera alone. That is what lets stage 2 correct the cameras.
3. **Landmark-specific observations.**
   - Painted lines, crease lines and board-base pixels from `build_field_paint_masks.py` use every
     pixel.
   - Other mask sources are accepted with `--landmark`:
     - a filled crease mask uses its outline;
     - a boards or kick-plate mask uses its bottom edge, kept only where turf lies directly below;
     - faceoff dots use one centroid per frame.
4. **Proper similarity, global search, robust refinement.** Plane coordinates map to field feet by
   scale, rotation and translation, never a mirror.
   - **Start.** A brute-force distance-transform search over rotation, scale and translation. The
     scale range comes from a physical prior: a broadcast camera is 20–120 ft above the floor. If
     two or more faceoff dots are visible, their closed-form similarity is added as a candidate.
   - **Refinement.** Every candidate is refined with soft-L1 loss in plane units (observation
     space), and the lowest robust cost wins.

### Stage 2: camera refinement against the planar field

The stage-1 transform puts VGGT-Omega's cameras in field feet. The model then becomes what a
broadcast camera physically is:

- **One shared camera centre** for the clip.
- **Per-frame pan, tilt, roll and zoom corrections**, bounded to 20° and a factor of 2.
- **The arena's corner radius**, shared by all frames, with a soft prior.
- **Measured camera motion between neighbouring frames.**
  - Background points are KLT-tracked from each frame to the next and checked forward and back.
  - The RANSAC homography inliers are kept, about 40 per frame pair. Moving players and fixed
    screen graphics fall out as outliers, and pairs with few inliers (cuts, heavy blur) are skipped.
  - A camera that turns and zooms about a fixed centre moves every static point by
    K<sub>b</sub> R<sub>b</sub> R<sub>a</sub><sup>T</sup> K<sub>a</sub><sup>−1</sup>, whatever its
    depth. The image distance between that prediction and the tracked point is a residual.
  - Points are tracked only when neighbouring frames are at most 0.25 s apart: the 10 fps clips,
    not the 1 fps keyframes.
  - This ties each frame's pan, tilt, roll and zoom to its neighbours'. Without it, stretches that
    show only two landmarks leave the camera underdetermined. Frames 22–45 of the 10 s clip show
    only the right restraining line and the far board base. Fitted without this term, the camera
    there tilts 60 px in one frame while the image barely moves vertically.
- **Priors**: a weak pull towards VGGT-Omega's own per-frame pointing (or the propagated cameras,
  `--seed-cameras`), plus smoothness between neighbouring frames. Smoothing is switched off across
  broadcast cuts.

Several starting centres are tried, from VGGT-Omega's centre to points much closer to the floor it
looks at. Moving the centre re-aims every camera at the same floor point and rescales its focal
length, which undoes the zoom-as-dolly motion. Residuals are floor distances divided by the local
feet-per-pixel, which approximates image pixels.

Frames the shared camera cannot explain (cuts to another camera) are dropped from the shared fit and
refitted with their own camera. If that still fails, they are flagged `unreliable` and left out of the
exported homographies.

### Full frame rate from checked keyframes

Week 1 is fitted twice. The keyframe pass (every 10th frame, 1 fps) is labeled and checked by eye
as above. The 10 fps pass reuses it:

- `propagate_field_cameras.py` gives every 10 fps frame a camera. Anchors are the keyframe cameras
  pinned by at least two painted lines or creases, because one line plus the boards lets the camera
  slide along the floor while still fitting. Between two anchors, the camera is carried forward
  and backward through the tracked background motion and the two chains are blended by time, so
  drift closes at the anchors.
- `build_field_paint_masks.py` keeps the checked keyframe masks and labels the other frames from
  the propagated cameras.
- `fit_field_plane_landmarks.py --seed-cameras` refines all 1,530 cameras together, with the
  motion term. Its `camera_motion_check` then lists any frame that still disagrees with the tracked
  motion.

### 3D placement

`render_vggt_scene_3d.py` places VGGT-Omega's points through the refined cameras:

- Each pixel goes on its refined camera's ray, at the height that VGGT-Omega gives it above that
  frame's own floor. VGGT-Omega's floor height drifts between frames, so a plane is fitted to
  each frame's turf points, and heights under 1 ft are snapped to the floor.
- Floor pixels land on z = 0, so the floor paint of every frame lands on the fitted lines.
- Players and boards keep VGGT-Omega's heights.
- Points too high for that, the crowd above the camera, keep VGGT-Omega's distance, rescaled per
  frame.
- The accumulated orbit cloud uses the frames the shared broadcast camera explains. Close-ups
  with their own camera and frames without landmarks are left out of it; the per-frame timeline
  shows every frame and labels its status.

`--geometry raw` maps VGGT-Omega's own points with the rigid plane transform instead. It shows the
reconstruction as VGGT-Omega made it: the telephoto distortion and frame-to-frame drift
described above, with a skewed logo and paint doubled where frames disagree.

![10 s clip, every 4th frame from above: raw VGGT-Omega geometry vs points placed through the refined cameras](figures/clip10s_topdown_raw_vs_refined.jpg)
![Week 1, every 3rd keyframe from above](figures/week1_topdown_raw_vs_refined.jpg)

## Evidence that the fit is right

Positions of each landmark's pixels on the field, through the refined cameras:

| landmark | model | 10 s clip | Week 1 keyframes |
|---|---|---|---|
| left restraining line | x = 57.5 ft | 57.50 ft (IQR 0.26) | 57.50 ft (IQR 0.28) |
| midfield line | x = 100 ft | 100.00 ft (IQR 0.32) | 99.98 ft (IQR 0.21) |
| right restraining line | x = 142.5 ft | 142.47 ft (IQR 0.28) | 142.50 ft (IQR 0.29) |
| creases (circle fit) | centre x = 12 / 188 ft, r = 9.25 ft | 187.9 ft, r 8.8 | 11.6 / 188.2 ft, r 9.1 / 9.0 |
| far board base | rounded outline | median 0.42 ft off | median 0.64 ft off |
| faceoff dots | (157.5, 70) / (157.5, 15) | (157.6, 70.5) / (157.7, 15.0) | not visible |

Crease radii are measured to the centre of a 5 in line.

- **Held-out faceoff dots (10 s clip).** The dots are the only near-side landmarks in this clip, so
  each is held out in turn:
  - With only the upper dot in the fit, the lower dot lands 0.65 ft from its spot.
  - With only the lower dot in the fit, the upper dot lands 1.33 ft from its spot.

  The fit uses both. Without any dot, nothing fixes the near side. Camera distance then trades
  against zoom, and the near dot drifts by about 3 ft while every far-side landmark still fits.
- **Held-out faceoff dots (Week 1).** No faceoff dot is used in the Week 1 fit. The painted dots
  are found by colour within 90 px of where the keyframe fit projects them: 171 detections in which
  exactly one small white blob lies near the projected spot. Lifted through the fit, they land
  0.71 ft from their rulebook spots (median; 90th percentile 1.6 ft; 13 px in the image).

  ![Week 1 held-out faceoff dots: green circle = the fit's projected spot](figures/week1_heldout_faceoff_dots.jpg)
- **Camera position.**
  - 10 s clip: the recovered camera is 114 ft behind the near sideline and 50 ft up, level with
    centre floor. VGGT-Omega's own estimate is 216 ft back and 75 ft up.
  - Week 1: the camera is 34 ft behind the near sideline and 20 ft up, with a 36° field of view
    (38° median at 10 fps). At 10 fps, 1,451 of 1,530 frames fit this shared camera and 79 are close-ups with no
    positions; none are flagged `unreliable`.
- **Camera motion check.** The refined cameras move the tracked background points to within
  0.42 px of where they are measured (median over 1,447 neighbouring-frame pairs of Week 1 at
  10 fps; worst pair 2.6 px) and 1.2 px on the 10 s clip (97 pairs; worst 4.4 px). No frame is over
  5 px. Fitted without the motion term, the 10 s clip has pairs at 83 px (frame 28) and 18 px
  (frame 46, where the faceoff dots enter the fit).
- **Per landmark** (median / 90th percentile px):

  | landmark | 10 s clip | Week 1 keyframes | Week 1 at 10 fps |
  |---|---|---|---|
  | midfield line | 4.8 / 9.6 | 2.5 / 5.2 | 2.7 / 6.1 |
  | left restraining line | 3.1 / 5.4 | 1.5 / 3.0 | 1.5 / 2.9 |
  | right restraining line | 2.9 / 7.0 | 1.8 / 4.3 | 1.7 / 3.7 |
  | left crease | not visible | 1.9 / 5.2 | 1.9 / 5.1 |
  | right crease | 3.2 / 10.7 | 2.4 / 6.9 | 2.3 / 6.5 |
  | far board base | 2.5 / 9.3 | 1.8 / 10.4 | 1.7 / 8.4 |
  | faceoff dots (upper / lower) | 5.4 / 11.0, 7.0 / 20.1 | not used | not used |

## Running it

The commands are in the README's Run section.

**Labeling a new clip.**

1. Bootstrap. Any fit of the clip can propose labels. On a new clip, run `fit_field_plane_landmarks.py`
   once on SAM masks (`--landmark white_lines=... --landmark field_outline=... --detect-white-paint`),
   or run `build_field_paint_masks.py` without `--fit`, in which case every candidate starts unlabeled.
2. Review. Look through `paint_masks/review/sheet_*.jpg`. Write corrections as
   `{"frame_00077.jpg": {"4": "right_restraining_line", "9": "reject"}, ...}`; `"board_base": "reject"`
   drops a frame's board edge.
3. Repeat. Rebuild the masks, refit, and propose again from the refit until the proposals stop
   changing.

**Options.**

- `--motion-max-interval-seconds` (0.25) is the largest frame spacing (1 / `--fps`) at which
  background points are tracked at all.
- `--estimate-dims none` keeps the rulebook floor exactly.
- `--estimate-dims corner_radius,crease_radius,...` frees more dimensions as a check against
  the rulebook.
- `--centre-circle-radius-ft` draws and fits a centre faceoff circle, for arenas that paint one.

Outputs in `--output-dir`:

- `field_plane_fit.json`: the plane, the plane→field similarity, the fitted `field_dims`, per-frame
  cameras in field feet (rigid and refined), per-landmark and per-frame errors, landmark positions
  and frame status.
  - It also holds `camera_motion_check`: how far the refined cameras move the tracked background
    points from where they are measured in the next frame.
  - The check gives the median, 95th percentile and maximum, and lists every frame over
    `--motion-check-px` (5 px).
  - The fit's printed summary carries the same block, so a camera jump shows up without
    looking at any video.
- `field_homographies.json`: per-frame image→field homographies in the `floor_homography_matrix_v1`
  schema, with `field_dims`. `render_field_birds_eye.py` maps the players' foot points with them.
- `field_reprojection_h264.mp4`: every frame with the rigid fit (top) and the refined fit (bottom) drawn on it.
- `landmarks_on_field.jpg`: stage-1 lifted landmarks on the field diagram.

`scripts/export_vggt_omega_vggt_layout.py` writes the reconstruction and the rigid fit in the
518×518 layout that the SAM-Body4D mesh scripts read (`vggt_predictions_compact.npz` and
`field_fit_vggt_layout.json`). The meshes are placed with the stage-1 rigid fit, not the per-frame
refined cameras.

`scripts/render_vggt_scene_3d.py` renders the VGGT-Omega point cloud on the fitted field:

- an orbit that ends in a top-down view
- a per-frame timeline: the input frame with the fitted field, an oblique 3D view and a top-down
  view

Points are placed through the refined cameras by default (see 3D placement above).

## Limitations

- **Near side of the floor.** Broadcast views rarely show painted marks near the near boards.
  Their geometry rests on the camera model and, in the 10 s clip, on the two faceoff dots.
  Without them, camera distance and zoom trade off, as described above.
- **Floor length.** The length is assumed to be the 200 ft rink standard.
- **Label review is done by eye.** It holds for these two clips. A new clip needs its own review
  pass, usually short once a bootstrap fit exists.
- **End symmetry.** On a clip where no landmark tells the ends apart, the solution is only fixed up
  to the field's 180° symmetry. Labeled creases and restraining lines remove this.
- **Moving cameras.** Stage 2 assumes the main broadcast camera stays in one place. Clips that
  mostly come from moving cameras need more frames flagged `own_camera`, and those frames are less
  constrained. Check the `status` field before trusting one.
- **Zoomed close-ups.** In a tight close-up that shows only the midfield stub and the far
  boards, zoom and tilt are only weakly separated. On Week 1, frames of the opening faceoff place
  the centre logo up to about 5 ft apart along the camera's viewing direction. The logo is
  therefore smeared in the accumulated top-down cloud, while every painted line is sharp.
- **Refined 3D placement.** Placing points through the refined cameras uses VGGT-Omega's heights
  above the floor. Players and boards are therefore as tall as VGGT-Omega makes them, and crowd
  points above the camera keep VGGT-Omega's distances.
