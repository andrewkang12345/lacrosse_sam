"""Interactive viser viewer of the VGGT-Omega scene with the fitted field and SAM-Body4D meshes.

Shows VGGT-Omega's points (filtered by a confidence percentile, all frames or one), its
cameras with their images, the fitted field drawn on the floor plane through the rigid
plane->field fit, and the SAM-Body4D meshes of one chosen frame, placed as in
render_sam_body4d_on_field.py. Open http://127.0.0.1:<port>.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as viser_tf
from PIL import Image

import field_model as fm
from render_sam_body4d_on_field import body_extent_targets, frame_meshes, load_layout, mesh_frames

FIELD_COLORS = {"outline": (255, 216, 48), "lines": (245, 245, 235), "crease": (235, 55, 45), "dots": (245, 245, 235)}
TEAM_COLORS = {"black": (40, 40, 40), "white": (240, 240, 240), "referee": (150, 150, 150), "unknown": (90, 200, 255)}


def field_to_world(xy: np.ndarray, layout_fit: dict, offset: float) -> np.ndarray:
    """Field feet -> points on the VGGT-Omega floor plane, lifted by offset along its normal."""
    plane = layout_fit["floor_plane"]
    center, u, v = (np.asarray(plane[k], dtype=np.float64) for k in ("center", "basis_u", "basis_v"))
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    homog = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(np.asarray(layout_fit["plane_to_floor_homography"])).T
    uv = homog[:, :2] / homog[:, 2:]
    return center + uv[:, :1] * u + uv[:, 1:2] * v + offset * normal


def add_field(server: viser.ViserServer, layout_fit: dict, dims: fm.FieldDims, scene_center: np.ndarray) -> list:
    outline = fm.outline_polyline(dims)
    surface = np.vstack([[[100.0, dims.cy]], outline])
    faces = np.asarray([[0, i, i + 1] for i in range(1, len(outline))], dtype=np.int32)  # the outline is closed
    handles = [server.scene.add_mesh_simple("field/surface", vertices=(field_to_world(surface, layout_fit, 0.006) - scene_center).astype(np.float32),
                                            faces=faces, color=(24, 110, 72), opacity=0.22, side="double", flat_shading=True)]
    for k, (kind, poly) in enumerate(fm.drawing(dims)):
        if kind == "dots":
            t = np.linspace(0, 2 * np.pi, 33)
            poly = poly[0] + np.column_stack([np.cos(t), np.sin(t)])
        world = field_to_world(poly, layout_fit, 0.018) - scene_center
        segments = np.stack([world[:-1], world[1:]], axis=1).astype(np.float32)
        handles.append(server.scene.add_line_segments(f"field/{kind}_{k}", points=segments, colors=FIELD_COLORS[kind], line_width=3.0))
    return handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout-dir", required=True, help="Output folder of export_vggt_omega_vggt_layout.py.")
    parser.add_argument("--vggt-repo", default="third_party/VGGT")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--teams-json", required=True, help="teams.json from classify_sam3_teams_by_torso_color.py")
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--focal-root", required=True)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--conf-threshold", type=float, default=20.0, help="Initial confidence percentile of the shown points.")
    parser.add_argument("--min-conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-fit-vertices", type=int, default=1200)
    parser.add_argument("--fixed-field-player-size", action="store_true")
    parser.add_argument("--goalie-object-ids", default="")
    parser.add_argument("--goalie-extent-threshold", type=float, default=1.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.vggt_repo).resolve()))
    from vggt.utils.geometry import closed_form_inverse_se3
    from vggt.utils.load_fn import load_and_preprocess_images_square

    layout_dir, mesh_root, focal_root = Path(args.layout_dir), Path(args.mesh_root), Path(args.focal_root)
    layout_fit = json.loads((layout_dir / "field_fit_vggt_layout.json").read_text())
    dims = fm.FieldDims(**json.loads(Path(layout_fit["field_plane_fit"]).read_text())["field_dims"])
    layout = load_layout(layout_dir / "vggt_predictions_compact.npz", Path(args.vggt_repo))
    teams = {r["frame"]: dict(zip(r["object_ids"], r["teams"])) for r in json.loads(Path(args.teams_json).read_text())["frames"]}
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    frame_indices = layout["frame_indices"].tolist()
    images, _ = load_and_preprocess_images_square([str(frame_paths[i]) for i in frame_indices], target_size=layout["depth_map"].shape[1])
    images = images.cpu().numpy()
    with Image.open(frame_paths[0]) as image:
        image_shape = (image.height, image.width)
    goalie_ids = {int(v) for v in args.goalie_object_ids.split(",") if v.strip()}
    extents = body_extent_targets(mesh_root, args.fixed_field_player_size, goalie_ids, args.goalie_extent_threshold)
    lookup = {f: i for i, f in enumerate(frame_indices)}

    points = layout["world_points"].reshape(-1, 3)
    colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    conf = layout["depth_conf"].reshape(-1)
    s, h, w = layout["depth_conf"].shape
    point_frame = np.repeat(np.arange(s), h * w)
    scene_center = points.mean(axis=0)
    cam_to_world = closed_form_inverse_se3(layout["extrinsic"])[:, :3, :]
    cam_to_world[..., -1] -= scene_center

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    gui_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
    gui_points = server.gui.add_checkbox("Show VGGT-Omega points", initial_value=True)
    gui_field = server.gui.add_checkbox("Show field", initial_value=True)
    gui_conf = server.gui.add_slider("Confidence percentile", min=0, max=100, step=0.1, initial_value=args.conf_threshold)
    gui_point_frame = server.gui.add_dropdown("Points from frame", options=["All"] + [str(f) for f in frame_indices], initial_value="All")
    meshed = [f for f in mesh_frames(mesh_root) if f in lookup]
    gui_mesh_frame = server.gui.add_dropdown("SAM-Body4D mesh frame", options=["None"] + [str(f) for f in meshed], initial_value=str(meshed[0]) if meshed else "None")

    cloud = server.scene.add_point_cloud("points", points=np.zeros((0, 3), np.float32), colors=np.zeros((0, 3), np.uint8), point_size=0.001, point_shape="circle")
    field_handles = add_field(server, layout_fit, dims, scene_center)
    camera_handles = []
    for i, f in enumerate(frame_indices):
        transform = viser_tf.SE3.from_matrix(cam_to_world[i])
        axis = server.scene.add_frame(f"camera_{f}", wxyz=transform.rotation().wxyz, position=transform.translation(), axes_length=0.05, axes_radius=0.002, origin_radius=0.002)
        img = (images[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        frustum = server.scene.add_camera_frustum(f"camera_{f}/frustum", fov=2 * np.arctan2(img.shape[0] / 2, 1.1 * img.shape[0]), aspect=img.shape[1] / img.shape[0], scale=0.05, image=img, line_width=1.0)
        camera_handles += [axis, frustum]
    mesh_handles = []

    def update_points() -> None:
        cloud.visible = gui_points.value
        keep = (conf >= np.percentile(conf, gui_conf.value)) & (conf > 1e-5)
        if gui_point_frame.value != "All":
            keep &= point_frame == lookup[int(gui_point_frame.value)]
        cloud.points, cloud.colors = (points[keep] - scene_center).astype(np.float32), colors[keep]

    def update_meshes() -> None:
        for handle in mesh_handles:
            handle.remove()
        mesh_handles.clear()
        if gui_mesh_frame.value == "None":
            return
        f = int(gui_mesh_frame.value)
        for obj_id, fit, world in frame_meshes(layout, lookup[f], f, image_shape, mesh_root, focal_root, extents, args.min_conf_percentile, args.max_fit_vertices):
            team = teams.get(f, {}).get(obj_id, "unknown")
            if team != "ignore":
                mesh_handles.append(server.scene.add_mesh_simple(f"meshes/{obj_id}", vertices=(world - scene_center).astype(np.float32), faces=fit["faces"],
                                                                 color=TEAM_COLORS.get(team, TEAM_COLORS["unknown"]), material="toon5", side="double", flat_shading=True))
        print(f"frame {f}: {len(mesh_handles)} meshes")

    for gui in (gui_points, gui_conf, gui_point_frame):
        gui.on_update(lambda _: update_points())
    gui_mesh_frame.on_update(lambda _: update_meshes())
    gui_cameras.on_update(lambda _: [setattr(h_, "visible", gui_cameras.value) for h_ in camera_handles])
    gui_field.on_update(lambda _: [setattr(h_, "visible", gui_field.value) for h_ in field_handles])
    update_points()
    update_meshes()
    print(f"Open http://127.0.0.1:{args.port}")
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    main()
