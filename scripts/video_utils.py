from __future__ import annotations

import colorsys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_pil_frames(frames_dir: Path) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in sorted_frame_paths(frames_dir)]


def id_color(obj_id: int) -> np.ndarray:
    hue = ((obj_id * 0.61803398875) % 1.0)
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return np.array([int(channel * 255) for channel in rgb], dtype=np.uint8)


def normalize_mask(mask) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    return arr > 0


def overlay_masks(
    frame: Image.Image,
    masks_by_id: dict[int, np.ndarray],
    alpha: float = 0.45,
    draw_labels: bool = True,
) -> np.ndarray:
    base = np.asarray(frame.convert("RGB")).copy()
    out = base.astype(np.float32)
    for obj_id in sorted(masks_by_id):
        mask = normalize_mask(masks_by_id[obj_id])
        if mask.shape[:2] != base.shape[:2] or not mask.any():
            continue
        color = id_color(int(obj_id)).astype(np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color

        if draw_labels:
            ys, xs = np.where(mask)
            if len(xs):
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv = out.astype(np.uint8)
                # Keep drawing in PIL to avoid an OpenCV dependency in this helper.
                label_img = Image.fromarray(cv)
                draw = ImageDraw.Draw(label_img)
                label = str(obj_id)
                color_tuple = tuple(int(v) for v in id_color(int(obj_id)))
                draw.rectangle((x1, y1, x2, y2), outline=color_tuple, width=2)
                draw.rectangle((x1, max(0, y1 - 18), x1 + 34, y1), fill=color_tuple)
                draw.text((x1 + 4, max(0, y1 - 17)), label, fill=(0, 0, 0))
                out = np.asarray(label_img).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def write_h264_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="error",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def draw_coordinate_grid(frame: Image.Image, title: str) -> Image.Image:
    image = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    for x in range(0, width + 1, 100):
        draw.line((x, 0, x, height), fill=(255, 255, 255), width=1)
        draw.text((x + 3, 4), str(x), fill=(255, 255, 0), font=font)
    for y in range(0, height + 1, 100):
        draw.line((0, y, width, y), fill=(255, 255, 255), width=1)
        draw.text((4, y + 3), str(y), fill=(255, 255, 0), font=font)

    draw.rectangle((0, 0, min(width, 620), 28), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)
    return image
