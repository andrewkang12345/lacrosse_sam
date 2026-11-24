"""Player tracks on the field from per-frame foot points.

Each foot point is found in the image and mapped to the floor through the frame's
homography, so its uncertainty in feet follows from a few pixels of foot-point noise and
the homography's local stretch: players further away, and the depth direction, get the
most. A partly hidden player gets a larger pixel uncertainty: the mask bottom touches
another player who reaches as low (feet hidden behind the player in front, or a mask
that runs onto that player's legs), or the mask is cut by the side of the image.

Each SAM3 track is run through a constant-velocity Kalman filter. A foot point the filter
cannot explain, faster than a player runs, or with a different team is set aside. If the
track keeps disagreeing for `confirm` frames in a row and those points agree with each
other, SAM3 has moved the ID to another player and the track is split there; otherwise
the set-aside points are outliers. A frame where many tracks disagree at once is a bad
frame, and all its points are dropped.

The pieces are then joined across gaps of up to `max_gap` frames by one-to-one
assignment: same team, at the lowest cost of the second piece's start given the motion
at the end of the first. Each joined track is smoothed with a Rauch-Tung-Striebel
smoother, which also fills its gaps of up to `max_gap` frames. A SAM3 ID kept across a
longer gap (a close-up) stays one track, but the gap is not filled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FootPoint:
    frame: int
    object_id: int
    team: str
    image_point: tuple[float, float]
    field_point: np.ndarray  # feet
    jacobian: np.ndarray  # feet per image pixel; columns +u, +v
    hidden: bool  # partly hidden: see partly_hidden()


@dataclass
class TrackerSettings:
    fps: float
    # foot-point noise along image x and y, and the acceleration density, fitted by
    # likelihood to the Week 1 and 10 s clips
    foot_px: tuple[float, float] = (1.5, 3.0)
    hidden_px: tuple[float, float] = (8.0, 16.0)
    acceleration: float = 100.0  # ft^2/s^3
    start_speed: float = 10.0  # ft/s, spread of the unknown speed at a track start
    gate: float = 16.0  # Mahalanobis distance squared
    confirm: int = 3  # frames of agreement that confirm an ID switch
    max_gap: int = 10  # frames
    max_speed: float = 30.0  # ft/s
    noise_ft: float = 3.0


MIN_MASK_PX = 200  # smaller masks are SAM3 remnants, not players


def partly_hidden(masks: dict[int, np.ndarray], margin: int = 3) -> set[int]:
    """Masks cut by the side of the image, or whose lowest quarter touches another mask that reaches at least as low."""
    boxes = {}
    out = set()
    for oid, m in masks.items():
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        if m.sum() >= MIN_MASK_PX:
            boxes[oid] = (cols[0], rows[0], cols[-1], rows[-1])
            if cols[0] <= 1 or cols[-1] >= m.shape[1] - 2:
                out.add(oid)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    for oid, (x0, y0, x1, y1) in boxes.items():
        band = int(y1 - 0.25 * (y1 - y0))
        for other, (a0, b0, a1, b1) in boxes.items():
            if oid in out:
                break
            if other == oid or b1 < y1 - margin or a1 < x0 - margin or a0 > x1 + margin or b0 > y1 + margin:
                continue
            r0, r1 = max(band, b0 - margin, 0), y1 + 1
            c0, c1 = max(x0, a0 - margin, 0), min(x1, a1 + margin) + 1
            if r1 > r0 and c1 > c0 and (cv2.dilate(masks[other][r0:r1, c0:c1].astype(np.uint8), kernel).astype(bool) & masks[oid][r0:r1, c0:c1]).any():
                out.add(oid)
    return out


def measurement_cov(p: FootPoint, s: TrackerSettings) -> np.ndarray:
    px = s.hidden_px if p.hidden else s.foot_px
    return p.jacobian @ np.diag(np.square(px)) @ p.jacobian.T


def transition(frames: int, s: TrackerSettings) -> tuple[np.ndarray, np.ndarray]:
    dt = frames / s.fps
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Q = s.acceleration * np.kron(np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), np.eye(2))
    return F, Q


def start_state(p: FootPoint, s: TrackerSettings) -> tuple[np.ndarray, np.ndarray]:
    P = np.zeros((4, 4))
    P[:2, :2] = measurement_cov(p, s)
    P[2:, 2:] = s.start_speed**2 * np.eye(2)
    return np.r_[p.field_point, 0.0, 0.0], P


def predict(x: np.ndarray, P: np.ndarray, frames: int, s: TrackerSettings) -> tuple[np.ndarray, np.ndarray]:
    F, Q = transition(frames, s)
    return F @ x, F @ P @ F.T + Q


def innovation(x: np.ndarray, P: np.ndarray, p: FootPoint, s: TrackerSettings) -> tuple[np.ndarray, np.ndarray, float]:
    y = p.field_point - x[:2]
    S = P[:2, :2] + measurement_cov(p, s)
    return y, S, float(y @ np.linalg.solve(S, y))


def update(x: np.ndarray, P: np.ndarray, p: FootPoint, s: TrackerSettings) -> tuple[np.ndarray, np.ndarray]:
    y, S, _ = innovation(x, P, p, s)
    K = np.linalg.solve(S, P[:2, :]).T
    A = np.eye(4)
    A[:, :2] -= K
    return x + K @ y, A @ P @ A.T + K @ measurement_cov(p, s) @ K.T


def faster_than_a_player(distance_ft: float, frames: int, s: TrackerSettings) -> bool:
    return distance_ft > s.max_speed * frames / s.fps + s.noise_ft


def split_track(points: list[FootPoint], s: TrackerSettings) -> tuple[list[list[FootPoint]], list[FootPoint], Counter, Counter]:
    """Pieces of one SAM3 track that follow one player, the outliers, and per-frame tested/rejected counts."""
    pieces: list[list[FootPoint]] = []
    outliers: list[FootPoint] = []
    tested, rejected = Counter(), Counter()
    current: list[FootPoint] = []
    pending: list[FootPoint] = []
    x = P = None
    for p in points:
        if not current:
            x, P = start_state(p, s)
            current = [p]
            continue
        gap = p.frame - current[-1].frame
        xp, Pp = predict(x, P, gap, s)
        tested[p.frame] += 1
        moves = innovation(xp, Pp, p, s)[2] <= s.gate
        if gap > s.max_gap:  # across a long gap the filter accepts almost anything
            moves = moves and not faster_than_a_player(float(np.linalg.norm(p.field_point - current[-1].field_point)), gap, s)
        if moves and p.team == current[0].team:
            x, P = update(xp, Pp, p, s)
            current.append(p)
            outliers += pending
            pending = []
            continue
        rejected[p.frame] += int(not moves)
        if pending and (p.team != pending[-1].team or faster_than_a_player(float(np.linalg.norm(p.field_point - pending[-1].field_point)), p.frame - pending[-1].frame, s)):
            outliers += pending
            pending = []
        pending.append(p)
        if len(pending) >= s.confirm:
            pieces.append(current)
            current, pending = pending, []
            x, P = start_state(current[0], s)
            for a, b in zip(current, current[1:]):
                x, P = update(*predict(x, P, b.frame - a.frame, s), b, s)
    if current:
        pieces.append(current)
    if len(pending) >= s.confirm:
        pieces.append(pending)
    else:
        outliers += pending
    return pieces, outliers, tested, rejected


def smooth(points: list[FootPoint], s: TrackerSettings) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Rauch-Tung-Striebel smoothed state and covariance at every frame from the first point to the last."""
    by_frame = {p.frame: p for p in points}
    first, last = points[0].frame, points[-1].frame
    F, Q = transition(1, s)
    xf, Pf, xp, Pp = {}, {}, {}, {}
    x, P = start_state(points[0], s)
    for t in range(first, last + 1):
        if t > first:
            x, P = F @ x, F @ P @ F.T + Q
        xp[t], Pp[t] = x, P
        if t > first and t in by_frame:
            x, P = update(x, P, by_frame[t], s)
        xf[t], Pf[t] = x, P
    out = {last: (xf[last], Pf[last])}
    for t in range(last - 1, first - 1, -1):
        C = np.linalg.solve(Pp[t + 1].T, (Pf[t] @ F.T).T).T
        xs, Ps = out[t + 1]
        out[t] = (xf[t] + C @ (xs - xp[t + 1]), Pf[t] + C @ (Ps - Pp[t + 1]) @ C.T)
    return out


def link(pieces: list[list[FootPoint]], teams: list[str], s: TrackerSettings) -> list[list[int]]:
    """Chains of piece indices joined end to start, one-to-one, by the cost of the motion between them."""
    ends = []
    for piece in pieces:
        states = smooth(piece, s)
        ends.append((piece[0].frame, states[piece[0].frame], piece[-1].frame, states[piece[-1].frame]))
    n = len(pieces)
    cost = np.full((n, n), np.inf)
    for i, (_, _, end_i, (x, P)) in enumerate(ends):
        for j, (start_j, (xj, Pj), _, _) in enumerate(ends):
            gap = start_j - end_i
            if not 1 <= gap <= s.max_gap or (teams[i] != teams[j] and "unknown" not in (teams[i], teams[j])):
                continue
            xq, Pq = predict(x, P, gap, s)
            y = xj[:2] - xq[:2]
            S = Pq[:2, :2] + Pj[:2, :2]
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 <= s.gate and not faster_than_a_player(float(np.linalg.norm(y)), gap, s):
                cost[i, j] = d2 + float(np.log(np.linalg.det(S)))
    nxt: dict[int, int] = {}
    valid = np.isfinite(cost)
    if valid.any():
        rows, cols = linear_sum_assignment(np.where(valid, cost, 1e9))
        nxt = {int(i): int(j) for i, j in zip(rows, cols) if valid[i, j]}
    has_prev = set(nxt.values())
    chains = []
    for i in range(n):
        if i in has_prev:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        chains.append(chain)
    return chains


def track_players(points: list[FootPoint], s: TrackerSettings, bad_frame_share: float = 0.4) -> tuple[list[dict], dict]:
    """Smoothed field tracks: per track its team, SAM3 ids and per-frame states, plus statistics."""
    by_id: dict[int, list[FootPoint]] = {}
    for p in sorted(points, key=lambda p: p.frame):
        by_id.setdefault(p.object_id, []).append(p)
    tested, rejected = Counter(), Counter()
    for track in by_id.values():
        _, _, t, r = split_track(track, s)
        tested.update(t)
        rejected.update(r)
    bad = {f for f in tested if rejected[f] >= 2 and rejected[f] >= bad_frame_share * tested[f]}

    pieces, outliers = [], 0
    for track in by_id.values():
        kept = [p for p in track if p.frame not in bad]
        if kept:
            got, dropped, _, _ = split_track(kept, s)
            pieces += got
            outliers += len(dropped)
    teams = [Counter(p.team for p in piece).most_common(1)[0][0] for piece in pieces]
    chains = link(pieces, teams, s)

    tracks = []
    short = 0
    for chain in sorted(chains, key=lambda c: pieces[c[0]][0].frame):
        members = sorted((p for i in chain for p in pieces[i]), key=lambda p: p.frame)
        if len(members) < s.confirm:
            short += len(members)
            continue
        frames = [p.frame for p in members]
        filled = {t for a, b in zip(frames, frames[1:]) if b - a <= s.max_gap for t in range(a, b + 1)} | set(frames)
        states = {t: state for t, state in smooth(members, s).items() if t in filled}
        measured = {p.frame: p for p in members}
        tracks.append({
            "team": Counter(p.team for p in members).most_common(1)[0][0],
            "object_ids": sorted({p.object_id for p in members}),
            "states": states,
            "measured": measured,
        })
    stats = {
        "bad_frames": sorted(bad),
        "outlier_points": outliers,
        "points_in_short_tracks": short,
        "pieces": len(pieces),
        "splits": len(pieces) - sum(1 for track in by_id.values() if any(p.frame not in bad for p in track)),
        "pieces_joined": len(pieces) - len(chains),
        "tracks": len(tracks),
    }
    return tracks, stats
