"""Field cameras for every frame of a clip, from a field fit of its keyframes.

The keyframes are every `--keyframe-stride`-th frame starting at `--keyframe-offset`,
and they must be the same images the keyframe fit used.

Anchors are the keyframes whose camera the paint pins down: at least two different
painted lines or creases. One line plus the boards leaves the camera free to slide along
the floor while still fitting its landmarks. An anchor that disagrees with the tracked
motion to both of its neighbours is dropped.

Between two neighbouring anchors, the camera is carried forward from the first and
backward from the second through the background motion tracked between frames (pan,
tilt, roll and zoom about the fixed camera centre). The two estimates are blended by
time, so drift in either chain closes at the anchors.

The output has the layout of field_plane_fit.json where its readers need it:
`build_field_paint_masks.py --fit` uses it to propose labels, and
`fit_field_plane_landmarks.py --seed-cameras` starts its refinement from it.

Frames no chain reaches keep the nearest anchor's camera and are marked `unreliable`,
so no labels are proposed for them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import fit_field_plane_landmarks as fit
from build_field_paint_masks import OBJECT_IDS, PAINT_LABELS

PAINT_OBJECT_IDS = {OBJECT_IDS[name] for name in PAINT_LABELS}


def motion_step(K: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation and log zoom taking a camera with intrinsics K from frame a to frame b."""
    rays = np.column_stack([(pa[:, 0] - K[0, 2]) / K[0, 0], (pa[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(pa))])

    def residual(v: np.ndarray) -> np.ndarray:
        q = rays @ Rotation.from_rotvec(v[:3]).as_matrix().T
        zoom = np.exp(v[3])
        return np.column_stack([K[0, 0] * zoom * q[:, 0] / q[:, 2] + K[0, 2] - pb[:, 0], K[1, 1] * zoom * q[:, 1] / q[:, 2] + K[1, 2] - pb[:, 1]]).ravel()

    v = least_squares(residual, np.zeros(4), loss="soft_l1", f_scale=2.0, max_nfev=30, x_scale=[1e-3, 1e-3, 1e-3, 1e-2]).x
    return Rotation.from_rotvec(v[:3]).as_matrix(), float(v[3])


def walk(pairs: dict[int, fit.MotionPair], start: int, stop: int, K: np.ndarray, R: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Carry a camera from frame `start` towards `stop` until the tracked motion breaks."""
    step = 1 if stop > start else -1
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    t = start
    while t != stop:
        pair = pairs.get(max(t, t + step))  # keyed by the later frame of the pair
        if pair is None:
            break
        W, zoom = motion_step(K, pair.pa, pair.pb) if step > 0 else motion_step(K, pair.pb, pair.pa)
        K = K.copy()
        K[0, 0] *= np.exp(zoom)
        K[1, 1] *= np.exp(zoom)
        R = W @ R
        t += step
        out[t] = (K, R)
    return out


def blend(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], w: float) -> tuple[np.ndarray, np.ndarray]:
    K = a[0].copy()
    zoom = np.exp(w * np.log(b[0][0, 0] / a[0][0, 0]))
    K[0, 0] *= zoom
    K[1, 1] *= zoom
    R = Rotation.from_rotvec(w * Rotation.from_matrix(b[1] @ a[1].T).as_rotvec()).as_matrix() @ a[1]
    return K, R


def closure_px(chain: dict[int, tuple[np.ndarray, np.ndarray]], b: int, anchor: tuple[np.ndarray, np.ndarray]) -> float | None:
    """How far a chain arriving at anchor b misses it, in pixels at the image edge."""
    if b not in chain:
        return None
    angle = Rotation.from_matrix(chain[b][1] @ anchor[1].T).magnitude()
    return float(np.hypot(angle * anchor[0][0, 0], abs(np.log(chain[b][0][0, 0] / anchor[0][0, 0])) * anchor[0][0, 2]))


def painted_lines(mask_dir: Path, frame: int, min_pixels: int = 50) -> int:
    path = mask_dir / f"{frame:08d}.npz"
    if not path.exists():
        return 0
    with np.load(path) as data:
        return sum(1 for oid, m in zip(data["object_ids"], data["masks"]) if int(oid) in PAINT_OBJECT_IDS and int(m.sum()) >= min_pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keyframe-fit", required=True, help="field_plane_fit.json of the keyframes.")
    parser.add_argument("--keyframe-masks", required=True, help="masks/ directory of the keyframe run's paint masks.")
    parser.add_argument("--frames-dir", required=True, help="All frames of the clip.")
    parser.add_argument("--keyframe-offset", type=int, required=True)
    parser.add_argument("--keyframe-stride", type=int, required=True)
    parser.add_argument("--motion-points", type=int, default=40)
    parser.add_argument("--closure-check-px", type=float, default=25.0, help="Anchors that miss both neighbours by more than this are dropped; remaining intervals over it are listed.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    keyfit = json.loads(Path(args.keyframe_fit).read_text())
    refined = keyfit["cameras_field_ft"]["refined"]
    key_K, key_R, key_C = (np.asarray(refined[k]) for k in ("K", "R", "C"))
    key_status = keyfit["stage2_camera_refinement"]["frame_status"]
    frames_dir = Path(args.frames_dir)
    names = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    S = len(names)
    anchors = {}
    for k, key_name in enumerate(keyfit["frame_names"]):
        t = args.keyframe_offset + args.keyframe_stride * k
        if t < S and key_status.get(key_name) == "shared_camera" and painted_lines(Path(args.keyframe_masks), k) >= 2:
            anchors[t] = (key_K[k], key_R[k])
    order = sorted(anchors)
    if not order:
        raise RuntimeError("The keyframe fit has no shared-camera frames.")
    centre = np.median(np.asarray([key_C[(t - args.keyframe_offset) // args.keyframe_stride] for t in order]), axis=0)

    pairs = {p.b: p for p in fit.frame_motion_pairs(frames_dir, names, args.motion_points)}
    candidates = len(order)
    while len(order) > 2:
        miss = {}
        for a, b in zip(order[:-1], order[1:]):
            value = closure_px(walk(pairs, a, b, *anchors[a]), b, anchors[b])
            if value is not None:
                miss[(a, b)] = value
        both_sides = {b: min(miss[(a, b)], miss[(b, c)]) for a, b, c in zip(order[:-2], order[1:-1], order[2:]) if (a, b) in miss and (b, c) in miss}
        worst = max(both_sides, key=both_sides.get, default=None)
        if worst is None or both_sides[worst] <= args.closure_check_px:
            break
        order.remove(worst)
    anchors = {t: anchors[t] for t in order}
    cameras: dict[int, tuple[np.ndarray, np.ndarray]] = dict(anchors)
    status = {t: "keyframe" for t in order}
    closure = {}
    for t, cam in walk(pairs, order[0], -1, *anchors[order[0]]).items():
        if t >= 0:
            cameras[t], status[t] = cam, "propagated"
    for t, cam in walk(pairs, order[-1], S - 1, *anchors[order[-1]]).items():
        cameras[t], status[t] = cam, "propagated"
    for a, b in zip(order[:-1], order[1:]):
        forward = walk(pairs, a, b, *anchors[a])
        backward = walk(pairs, b, a, *anchors[b])
        value = closure_px(forward, b, anchors[b])
        if value is not None:
            closure[names[b]] = value
        for t in range(a + 1, b):
            f, g = forward.get(t), backward.get(t)
            if f is not None and g is not None:
                cameras[t], status[t] = blend(f, g, (t - a) / (b - a)), "propagated"
            elif f is not None or g is not None:
                cameras[t], status[t] = f if f is not None else g, "propagated"
    for t in range(S):
        if t not in cameras:
            nearest = min(order, key=lambda k: abs(k - t))
            cameras[t], status[t] = anchors[nearest], "unreliable"

    K = np.stack([cameras[t][0] for t in range(S)])
    R = np.stack([cameras[t][1] for t in range(S)])
    C = np.repeat(centre[None], S, axis=0)
    values = np.asarray(list(closure.values())) if closure else np.zeros(1)
    counts = {k: list(status.values()).count(k) for k in sorted(set(status.values()))}
    check = {
        "anchors_with_two_lines": candidates,
        "anchors_kept": len(order),
        "intervals_closed": len(closure),
        "anchor_intervals": len(order) - 1,
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
        "threshold_px": args.closure_check_px,
        "anchors_over_threshold": {n: round(v, 1) for n, v in closure.items() if v > args.closure_check_px},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "propagated_field_cameras_v1",
                "keyframe_fit": args.keyframe_fit,
                "keyframe_offset": args.keyframe_offset,
                "keyframe_stride": args.keyframe_stride,
                "frame_names": names,
                "field_dims": keyfit["field_dims"],
                "chain_closure_check": check,
                "stage2_camera_refinement": {"frame_status": {names[t]: status[t] for t in range(S)}},
                "cameras_field_ft": {"refined": {"K": K.tolist(), "R": R.tolist(), "C": C.tolist()}},
            }
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "frames": S, "motion_pairs": len(pairs), "status": counts, "chain_closure_check": check}, indent=2))


if __name__ == "__main__":
    main()
