from __future__ import annotations

import numpy as np


FLOOR_LENGTH_FT = 200.0
FLOOR_WIDTH_FT = 85.0
CORNER_RADIUS_FT = 22.0 + 8.0 / 12.0

CENTER_X_FT = FLOOR_LENGTH_FT / 2.0
CENTER_Y_FT = FLOOR_WIDTH_FT / 2.0

LEFT_GOAL_X_FT = 12.0
RIGHT_GOAL_X_FT = FLOOR_LENGTH_FT - LEFT_GOAL_X_FT
GOAL_WIDTH_FT = 4.0 + 9.0 / 12.0
GOAL_HALF_WIDTH_FT = GOAL_WIDTH_FT / 2.0
GOAL_REAR_POLE_LENGTH_FT = 4.0 + 6.0 / 12.0

CREASE_RADIUS_FT = 9.0 + 3.0 / 12.0
CREASE_LINE_WIDTH_FT = 5.0 / 12.0
CREASE_CHORD_BEHIND_GOAL_BASE_FT = 1.0
CREASE_CHORD_DEPTH_FROM_GOAL_LINE_FT = GOAL_REAR_POLE_LENGTH_FT + CREASE_CHORD_BEHIND_GOAL_BASE_FT


def line_points(x1: float, y1: float, x2: float, y2: float, samples: int = 120) -> np.ndarray:
    return np.column_stack([np.linspace(x1, x2, samples), np.linspace(y1, y2, samples)]).astype(np.float64)


def circle_points(cx: float, cy: float, radius: float, samples: int = 160, endpoint: bool = False) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=endpoint)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]).astype(np.float64)


def arc_points(cx: float, cy: float, radius: float, start_deg: float, stop_deg: float, samples: int = 120) -> np.ndarray:
    angles = np.deg2rad(np.linspace(start_deg, stop_deg, samples))
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]).astype(np.float64)


def rounded_floor_points(samples_per_corner: int = 24) -> np.ndarray:
    r = CORNER_RADIUS_FT
    centers = [
        (r, r, 180.0, 270.0),
        (FLOOR_LENGTH_FT - r, r, 270.0, 360.0),
        (FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT - r, 0.0, 90.0),
        (r, FLOOR_WIDTH_FT - r, 90.0, 180.0),
    ]
    return np.concatenate([arc_points(cx, cy, r, start, stop, samples_per_corner) for cx, cy, start, stop in centers], axis=0)


def rounded_outline_samples(samples_per_segment: int = 140) -> np.ndarray:
    r = CORNER_RADIUS_FT
    parts = [
        line_points(r, 0.0, FLOOR_LENGTH_FT - r, 0.0, samples_per_segment),
        line_points(FLOOR_LENGTH_FT, r, FLOOR_LENGTH_FT, FLOOR_WIDTH_FT - r, samples_per_segment // 2),
        line_points(FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT, r, FLOOR_WIDTH_FT, samples_per_segment),
        line_points(0.0, FLOOR_WIDTH_FT - r, 0.0, r, samples_per_segment // 2),
        arc_points(r, r, r, 180.0, 270.0, 80),
        arc_points(FLOOR_LENGTH_FT - r, r, r, 270.0, 360.0, 80),
        arc_points(FLOOR_LENGTH_FT - r, FLOOR_WIDTH_FT - r, r, 0.0, 90.0, 80),
        arc_points(r, FLOOR_WIDTH_FT - r, r, 90.0, 180.0, 80),
    ]
    return np.concatenate(parts, axis=0)


def goal_post_segment(goal_x: float, samples: int = 12) -> np.ndarray:
    return line_points(
        goal_x,
        CENTER_Y_FT - GOAL_HALF_WIDTH_FT,
        goal_x,
        CENTER_Y_FT + GOAL_HALF_WIDTH_FT,
        samples,
    )


def goal_crease_segments(goal_x: float, arc_samples: int = 160, chord_samples: int = 48) -> list[np.ndarray]:
    if goal_x < CENTER_X_FT:
        chord_x = goal_x - CREASE_CHORD_DEPTH_FROM_GOAL_LINE_FT
        theta = float(np.rad2deg(np.arccos((chord_x - goal_x) / CREASE_RADIUS_FT)))
        arc = arc_points(goal_x, CENTER_Y_FT, CREASE_RADIUS_FT, theta, -theta, arc_samples)
    else:
        chord_x = goal_x + CREASE_CHORD_DEPTH_FROM_GOAL_LINE_FT
        theta = float(np.rad2deg(np.arccos((chord_x - goal_x) / CREASE_RADIUS_FT)))
        arc = arc_points(goal_x, CENTER_Y_FT, CREASE_RADIUS_FT, theta, 360.0 - theta, arc_samples)
    chord = line_points(chord_x, arc[-1, 1], chord_x, arc[0, 1], chord_samples)
    return [arc, chord]


def goal_crease_samples(arc_samples: int = 180, chord_samples: int = 60) -> np.ndarray:
    parts: list[np.ndarray] = []
    for goal_x in (LEFT_GOAL_X_FT, RIGHT_GOAL_X_FT):
        parts.extend(goal_crease_segments(goal_x, arc_samples=arc_samples, chord_samples=chord_samples))
    return np.concatenate(parts, axis=0)


def goal_crease_side_samples(arc_samples: int = 180, chord_samples: int = 60) -> dict[str, np.ndarray]:
    return {
        "left": np.concatenate(goal_crease_segments(LEFT_GOAL_X_FT, arc_samples=arc_samples, chord_samples=chord_samples), axis=0),
        "right": np.concatenate(goal_crease_segments(RIGHT_GOAL_X_FT, arc_samples=arc_samples, chord_samples=chord_samples), axis=0),
    }


def dense_floor_model(step_ft: float = 4.0) -> np.ndarray:
    points: list[tuple[float, float]] = [
        (CENTER_X_FT, CENTER_Y_FT),
        (CENTER_X_FT, 0.0),
        (CENTER_X_FT, FLOOR_WIDTH_FT),
        (57.5, 0.0),
        (57.5, FLOOR_WIDTH_FT),
        (142.5, 0.0),
        (142.5, FLOOR_WIDTH_FT),
        (LEFT_GOAL_X_FT, CENTER_Y_FT),
        (RIGHT_GOAL_X_FT, CENTER_Y_FT),
        (LEFT_GOAL_X_FT, CENTER_Y_FT - GOAL_HALF_WIDTH_FT),
        (LEFT_GOAL_X_FT, CENTER_Y_FT + GOAL_HALF_WIDTH_FT),
        (RIGHT_GOAL_X_FT, CENTER_Y_FT - GOAL_HALF_WIDTH_FT),
        (RIGHT_GOAL_X_FT, CENTER_Y_FT + GOAL_HALF_WIDTH_FT),
        (LEFT_GOAL_X_FT + CREASE_RADIUS_FT, CENTER_Y_FT),
        (RIGHT_GOAL_X_FT - CREASE_RADIUS_FT, CENTER_Y_FT),
        (42.5, 15.0),
        (42.5, 70.0),
        (157.5, 15.0),
        (157.5, 70.0),
        (CORNER_RADIUS_FT, 0.0),
        (FLOOR_LENGTH_FT - CORNER_RADIUS_FT, 0.0),
        (CORNER_RADIUS_FT, FLOOR_WIDTH_FT),
        (FLOOR_LENGTH_FT - CORNER_RADIUS_FT, FLOOR_WIDTH_FT),
        (0.0, CENTER_Y_FT),
        (FLOOR_LENGTH_FT, CENTER_Y_FT),
    ]
    for x in [LEFT_GOAL_X_FT, 57.5, CENTER_X_FT, 142.5, RIGHT_GOAL_X_FT]:
        for y in np.arange(0.0, FLOOR_WIDTH_FT + 0.001, step_ft):
            points.append((x, float(y)))
    for y in [0.0, FLOOR_WIDTH_FT]:
        for x in np.arange(CORNER_RADIUS_FT, FLOOR_LENGTH_FT - CORNER_RADIUS_FT + 0.001, step_ft):
            points.append((float(x), y))
    for x, y in circle_points(CENTER_X_FT, CENTER_Y_FT, 11.0, 48):
        points.append((float(x), float(y)))
    for x, y in goal_crease_samples(96, 32):
        points.append((float(x), float(y)))
    points.extend((float(x), float(y)) for x, y in rounded_floor_points(samples_per_corner=28))
    unique = sorted({(round(x, 3), round(y, 3)) for x, y in points})
    return np.asarray(unique, dtype=np.float32)
