"""Parametric NLL floor model: landmark curves, analytic distances and drawing.

Coordinates are feet on the floor: x along the length (centre line at x = 100),
y across the width (0 = near sideline, 85 = far sideline), z up.

Markings follow the NLL rulebook (2019-20, rules 1.5, 2.1, 3.1, 6.2):
restraining lines 42'6" from the centre line; goal line 12' from the end
boards; crease radius 9'3" centred on the goal-line midpoint, cut by a straight
back 1' behind the 4'6" goal base (5'6" behind the goal line); corner faceoff
spots 15' from the side boards and 15' outside the restraining lines. The boards
are those of the hockey rink the floor is laid in (200 x 85 ft standard); their
corner radius differs between rinks, so fit_field_plane_landmarks.py estimates
it per arena (FieldDims.ESTIMABLE), keeping the 85 ft width as the scale
reference. The floor length and the rulebook dimensions can be freed as a
diagnostic (FieldDims.DIAGNOSTIC).

Only markings that are painted are modelled. Neither arena in this repo has a
centre faceoff circle (both have a team logo there), so it is off by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

CENTRE_X_FT = 100.0


@dataclass(frozen=True)
class FieldDims:
    half_length: float = 100.0  # end boards at x = 100 -/+ half_length
    corner_radius: float = 22.0 + 8.0 / 12.0  # standard rink corner, refitted per arena
    restraining_offset: float = 42.5  # restraining lines at x = 100 -/+ offset
    goal_from_end: float = 12.0  # goal lines 12' in from the end boards
    crease_radius: float = 9.0 + 3.0 / 12.0  # crease circle centred on the goal-line midpoint
    crease_back: float = 4.5 + 1.0  # straight back of the crease, 1' behind the 4'6" goal base
    dot_offset_x: float = 57.5  # faceoff dots at x = 100 -/+ offset ...
    dot_offset_y: float = 27.5  # ... and y = 42.5 -/+ offset
    centre_circle_radius: float = 0.0  # 0 = not painted
    width: float = 85.0  # scale reference, never estimated

    # Estimated by default: the corner radius of the boards (the rinks these floors sit on
    # differ). The rest is fixed by the rulebook; the floor length (standard 200 ft rink) is
    # not identifiable from a broadcast view that never shows both ends, so it is only
    # freed on request, like the rulebook dimensions, as a diagnostic.
    ESTIMABLE = ("corner_radius",)
    DIAGNOSTIC = ("half_length", "restraining_offset", "goal_from_end", "crease_radius", "crease_back", "dot_offset_x", "dot_offset_y")

    @property
    def cy(self) -> float:
        return self.width / 2.0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_vector(self, names: tuple[str, ...], values: np.ndarray) -> "FieldDims":
        return replace(self, **{n: float(v) for n, v in zip(names, values)})

    def vector(self, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([getattr(self, n) for n in names], dtype=np.float64)


LINE_NAMES = ("left_restraining_line", "midfield_line", "right_restraining_line")
DOT_NAMES = ("faceoff_left_lower", "faceoff_left_upper", "faceoff_right_lower", "faceoff_right_upper")
OUTLINE_NAMES = ("field_outline", "far_boards_edge", "board_base")
LANDMARK_NAMES = (
    LINE_NAMES
    + DOT_NAMES
    + OUTLINE_NAMES
    + ("left_goal_line", "right_goal_line", "left_goal_crease", "right_goal_crease", "goal_crease", "left_crease_lines", "right_crease_lines", "white_lines")
)


def line_x(name: str, d: FieldDims) -> float:
    return {"left_restraining_line": CENTRE_X_FT - d.restraining_offset, "midfield_line": CENTRE_X_FT, "right_restraining_line": CENTRE_X_FT + d.restraining_offset}[name]


def goal_x(side: str, d: FieldDims) -> float:
    offset = d.half_length - d.goal_from_end
    return CENTRE_X_FT - offset if side == "left" else CENTRE_X_FT + offset


def dot_xy(name: str, d: FieldDims) -> np.ndarray:
    sx = -1.0 if "_left_" in name else 1.0
    sy = -1.0 if name.endswith("lower") else 1.0
    return np.asarray([CENTRE_X_FT + sx * d.dot_offset_x, d.cy + sy * d.dot_offset_y])


# ---------------------------------------------------------------------------
# Analytic distances (ft). xy has shape (..., 2).
# ---------------------------------------------------------------------------


def _segment_distance(xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((xy - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(xy - (a + t[..., None] * ab), axis=-1)


def _crease_geometry(side: str, d: FieldDims) -> tuple[float, float, float, float]:
    """Goal x, direction towards the field, chord x and chord half-height."""
    gx = goal_x(side, d)
    toward = 1.0 if side == "left" else -1.0
    back = min(d.crease_back, 0.95 * d.crease_radius)
    return gx, toward, gx - toward * back, float(np.sqrt(d.crease_radius**2 - back**2))


def _crease_outline_distance(xy: np.ndarray, side: str, d: FieldDims) -> np.ndarray:
    gx, toward, chord_x, h = _crease_geometry(side, d)
    rel = xy - np.asarray([gx, d.cy])
    rho = np.linalg.norm(rel, axis=-1)
    cos_front = toward * rel[..., 0] / np.maximum(rho, 1e-9)
    back = abs(chord_x - gx)
    on_arc = cos_front >= -back / d.crease_radius
    d_arc = np.where(on_arc, np.abs(rho - d.crease_radius), np.inf)
    d_chord = _segment_distance(xy, np.asarray([chord_x, d.cy - h]), np.asarray([chord_x, d.cy + h]))
    return np.minimum(d_arc, d_chord)


def _goal_line_distance(xy: np.ndarray, side: str, d: FieldDims) -> np.ndarray:
    gx = goal_x(side, d)
    return _segment_distance(xy, np.asarray([gx, d.cy - d.crease_radius]), np.asarray([gx, d.cy + d.crease_radius]))


def _outline_distance(xy: np.ndarray, d: FieldDims) -> np.ndarray:
    """|signed distance| to the rounded-rectangle board line."""
    r = min(d.corner_radius, d.cy - 1e-3)
    q = np.abs(xy - np.asarray([CENTRE_X_FT, d.cy])) - (np.asarray([d.half_length, d.cy]) - r)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return np.abs(outside + inside - r)


def distance(name: str, xy: np.ndarray, d: FieldDims) -> np.ndarray:
    """Distance (ft) from floor points to a landmark's painted curve."""
    if name in LINE_NAMES:
        return np.abs(xy[..., 0] - line_x(name, d))
    if name in OUTLINE_NAMES:
        return _outline_distance(xy, d)
    if name in DOT_NAMES:
        return np.linalg.norm(xy - dot_xy(name, d), axis=-1)
    if name in ("left_goal_line", "right_goal_line"):
        return _goal_line_distance(xy, name.split("_")[0], d)
    if name in ("left_goal_crease", "right_goal_crease"):
        return _crease_outline_distance(xy, name.split("_")[0], d)
    if name == "goal_crease":
        return np.minimum(_crease_outline_distance(xy, "left", d), _crease_outline_distance(xy, "right", d))
    if name in ("left_crease_lines", "right_crease_lines"):
        side = name.split("_")[0]
        return np.minimum(_crease_outline_distance(xy, side, d), _goal_line_distance(xy, side, d))
    if name == "white_lines":
        parts = [distance(n, xy, d) for n in LINE_NAMES + ("left_crease_lines", "right_crease_lines")]
        if d.centre_circle_radius > 0:
            parts.append(np.abs(np.linalg.norm(xy - np.asarray([CENTRE_X_FT, d.cy]), axis=-1) - d.centre_circle_radius))
        return np.min(np.stack(parts), axis=0)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Sampled curves (for projection into images and drawing)
# ---------------------------------------------------------------------------


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _line(a: tuple[float, float], b: tuple[float, float], n: int) -> np.ndarray:
    return np.column_stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)])


def outline_polyline(d: FieldDims, n_arc: int = 60) -> np.ndarray:
    """Closed, ordered rounded rectangle (counter-clockwise from the near-left corner arc)."""
    r = min(d.corner_radius, d.cy - 1e-3)
    x0, x1, y0, y1 = CENTRE_X_FT - d.half_length, CENTRE_X_FT + d.half_length, 0.0, d.width
    parts = [
        _arc(x0 + r, y0 + r, r, np.pi, 1.5 * np.pi, n_arc),
        _arc(x1 - r, y0 + r, r, 1.5 * np.pi, 2.0 * np.pi, n_arc),
        _arc(x1 - r, y1 - r, r, 0.0, 0.5 * np.pi, n_arc),
        _arc(x0 + r, y1 - r, r, 0.5 * np.pi, np.pi, n_arc),
    ]
    return np.vstack(parts + [parts[0][:1]])  # straight sides join consecutive arcs


def crease_polylines(side: str, d: FieldDims, n_arc: int = 120) -> list[np.ndarray]:
    gx, toward, chord_x, h = _crease_geometry(side, d)
    half_angle = np.arccos((chord_x - gx) * toward / d.crease_radius)  # from the forward axis to the chord ends
    centre_angle = 0.0 if toward > 0 else np.pi
    arc = _arc(gx, d.cy, d.crease_radius, centre_angle - half_angle, centre_angle + half_angle, n_arc)
    chord = _line((chord_x, d.cy - h), (chord_x, d.cy + h), max(8, n_arc // 6))
    goal = _line((gx, d.cy - d.crease_radius), (gx, d.cy + d.crease_radius), max(8, n_arc // 4))
    return [arc, chord, goal]


def curve(name: str, d: FieldDims, density_per_ft: float = 4.0) -> np.ndarray:
    """Dense samples of a landmark's curve (single point for dots)."""
    n_line = int(d.width * density_per_ft)
    if name in LINE_NAMES:
        x = line_x(name, d)
        return _line((x, 0.0), (x, d.width), n_line)
    if name in DOT_NAMES:
        return dot_xy(name, d)[None]
    if name in OUTLINE_NAMES:
        poly = outline_polyline(d, 200)
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, s[-1], int(s[-1] * density_per_ft))
        return np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])
    n_arc = int(2 * np.pi * d.crease_radius * density_per_ft)
    if name in ("left_goal_line", "right_goal_line"):
        return crease_polylines(name.split("_")[0], d, n_arc)[2]
    if name in ("left_goal_crease", "right_goal_crease"):
        return np.vstack(crease_polylines(name.split("_")[0], d, n_arc)[:2])
    if name == "goal_crease":
        return np.vstack([curve("left_goal_crease", d, density_per_ft), curve("right_goal_crease", d, density_per_ft)])
    if name in ("left_crease_lines", "right_crease_lines"):
        return np.vstack(crease_polylines(name.split("_")[0], d, n_arc))
    if name == "white_lines":
        parts = [curve(n, d, density_per_ft) for n in LINE_NAMES + ("left_crease_lines", "right_crease_lines")]
        if d.centre_circle_radius > 0:
            parts.append(_arc(CENTRE_X_FT, d.cy, d.centre_circle_radius, 0, 2 * np.pi, int(2 * np.pi * d.centre_circle_radius * density_per_ft)))
        return np.vstack(parts)
    raise KeyError(name)


def drawing(d: FieldDims) -> list[tuple[str, np.ndarray]]:
    """(kind, polyline) pairs to draw; kinds: outline, lines, crease, dots."""
    items: list[tuple[str, np.ndarray]] = [("outline", outline_polyline(d))]
    for name in LINE_NAMES:
        items.append(("lines", curve(name, d, 1.0)))
    if d.centre_circle_radius > 0:
        items.append(("lines", _arc(CENTRE_X_FT, d.cy, d.centre_circle_radius, 0, 2 * np.pi, 90)))
    for side in ("left", "right"):
        arc, chord, goal = crease_polylines(side, d, 90)
        items += [("crease", arc), ("crease", chord), ("lines", goal)]
    for name in DOT_NAMES:
        items.append(("dots", dot_xy(name, d)[None]))
    return items
