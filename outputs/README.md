# Outputs

`outputs/` is local, generated data and is ignored by Git; only this file is tracked. The root
README's Run section writes:

- `sam3/<clip>/`, `sam3/<clip>_undistorted/`: SAM3 detections, label masks and per-frame instance
  masks (`NNNNNNNN.npz`), on the original and on the lens-undistorted frames. The 10 s clip's are
  in `sam3/text/`.
- `sam2/floor_features/`: the 10 s clip's SAM2-tracked faceoff dots; `floor/faceoff_dots.json`:
  their clicks and SAM2 prompts.
- `vggt_omega/<clip>/`: everything after segmentation, listed in the root README's Outputs
  section (lens model, VGGT-Omega predictions, landmark masks, field fit, 3D videos, teams,
  bird's-eye, jersey numbers, meshes).
