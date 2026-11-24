"""Click the corner faceoff dots in a few frames; the clicks become SAM2 point prompts.

Serves a page at http://127.0.0.1:<port> showing the clip's frames. Pick a dot, click its
centre; repeat in a few frames where it is clearly visible. Every click is saved at once
to --output, together with the SAM2 prompts made from it (one SAM2 object per dot,
IDs 6-9), which track_sam2_clicks.py tracks through the clip. The tracked dots are
landmarks for fit_field_plane_landmarks.py (--landmark faceoff_right_upper=<masks>#6 ...).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image


SAM2_OBJECT_IDS = {"right_upper": 6, "right_lower": 7, "left_upper": 8, "left_lower": 9}
LANDMARKS = [
    {"id": "left_upper", "label": "Left upper face-off point", "x": 42.5, "y": 70.0},
    {"id": "right_upper", "label": "Right upper face-off point", "x": 157.5, "y": 70.0},
    {"id": "left_lower", "label": "Left lower face-off point", "x": 42.5, "y": 15.0},
    {"id": "right_lower", "label": "Right lower face-off point", "x": 157.5, "y": 15.0},
]


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Faceoff dots</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    header { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 14px; background: #1b1b1b; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 3; }
    label { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; }
    input, select, button { font: inherit; color: #eee; background: #292929; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    input[type="number"] { width: 82px; }
    button { cursor: pointer; }
    button:hover { background: #3a3a3a; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 420px; min-height: calc(100vh - 58px); }
    #stageWrap { overflow: auto; display: grid; place-items: start center; padding: 16px; }
    #stage { position: relative; width: min(1280px, 100%); }
    #frame { display: block; width: 100%; height: auto; user-select: none; }
    .dot { position: absolute; width: 18px; height: 18px; border: 2px solid #000; border-radius: 50%; transform: translate(-50%, -50%); background: #ffd43b; color: #000; display: grid; place-items: center; font: 800 10px system-ui; pointer-events: none; }
    aside { border-left: 1px solid #333; background: #171717; padding: 14px; overflow: auto; }
    #status { color: #bbb; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 5px 4px; border-bottom: 1px solid #303030; text-align: left; }
    .muted { color: #aaa; font-size: 12px; line-height: 1.35; }
    .ok { color: #7bd88f; }
    .warn { color: #ffcf70; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #333; } }
  </style>
</head>
<body>
  <header>
    <label>Frame <select id="frameSelect"></select></label>
    <label>Landmark <select id="landmarkSelect"></select></label>
    <label>World x <input id="worldX" type="number" step="0.001"></label>
    <label>World y <input id="worldY" type="number" step="0.001"></label>
    <button id="undo">Undo</button>
    <button id="clearFrame">Clear frame</button>
    <button id="clearAll">Clear all</button>
    <button id="save">Save</button>
    <span id="status"></span>
  </header>
  <main>
    <div id="stageWrap">
      <div id="stage"><img id="frame" draggable="false"><div id="dots"></div></div>
    </div>
    <aside>
      <div class="muted">
        Pick a faceoff dot, then click its centre. Click each dot in a few frames where it is clearly
        visible; SAM2 tracks it through the rest of the clip.
      </div>
      <h3>Frame Status</h3>
      <div id="frameStatus" class="muted"></div>
      <h3>Clicks</h3>
      <table>
        <thead><tr><th>Frame</th><th>Landmark</th><th>Image</th><th>World</th></tr></thead>
        <tbody id="clickRows"></tbody>
      </table>
    </aside>
  </main>
  <script>
    const frames = __FRAMES__;
    const width = __WIDTH__;
    const height = __HEIGHT__;
    const landmarks = __LANDMARKS__;
    let clicks = __CLICKS__;
    const frameSelect = document.getElementById('frameSelect');
    const landmarkSelect = document.getElementById('landmarkSelect');
    const worldX = document.getElementById('worldX');
    const worldY = document.getElementById('worldY');
    const img = document.getElementById('frame');
    const dots = document.getElementById('dots');
    const rows = document.getElementById('clickRows');
    const status = document.getElementById('status');
    const frameStatus = document.getElementById('frameStatus');

    function currentFrame() { return Number(frameSelect.value); }
    function frameUrl(frame) { return `/frame?idx=${frame}&t=${Date.now()}`; }
    function selectedLandmark() { return landmarks.find(l => l.id === landmarkSelect.value); }
    function setStatus(text) { status.textContent = text; }
    function countsByFrame() {
      const out = {};
      clicks.forEach(c => { out[c.frame] = (out[c.frame] || 0) + 1; });
      return out;
    }
    function updateWorldFields() {
      const lm = selectedLandmark();
      if (!lm) return;
      worldX.value = lm.x;
      worldY.value = lm.y;
    }
    async function postClicks() {
      const res = await fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({clicks})
      });
      const payload = await res.json();
      setStatus(`saved ${payload.count} clicks to ${payload.path}`);
    }
    function render() {
      const frame = currentFrame();
      img.src = frameUrl(frame);
      dots.innerHTML = '';
      const scaleX = img.clientWidth / width;
      const scaleY = img.clientHeight / height;
      clicks.forEach((c, i) => {
        if (c.frame !== frame) return;
        const d = document.createElement('div');
        d.className = 'dot';
        d.style.left = `${c.image.x * scaleX}px`;
        d.style.top = `${c.image.y * scaleY}px`;
        d.textContent = i + 1;
        dots.appendChild(d);
      });
      rows.innerHTML = clicks.map(c => {
        const image = `${Math.round(c.image.x)},${Math.round(c.image.y)}`;
        const world = `${c.world.x.toFixed(3)},${c.world.y.toFixed(3)}`;
        return `<tr><td>${c.frame}</td><td>${c.landmark_id}</td><td>${image}</td><td>${world}</td></tr>`;
      }).join('');
      const byFrame = countsByFrame();
      const count = byFrame[frame] || 0;
      frameStatus.innerHTML = `<span class="ok">Current frame has ${count} click(s).</span>`;
    }
    frames.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.frame;
      opt.textContent = `frame ${f.frame} | ${f.time_sec.toFixed(1)}s`;
      frameSelect.appendChild(opt);
    });
    landmarks.forEach(l => {
      const opt = document.createElement('option');
      opt.value = l.id;
      opt.textContent = `${l.label} (${l.x}, ${l.y})`;
      landmarkSelect.appendChild(opt);
    });
    frameSelect.addEventListener('change', render);
    landmarkSelect.addEventListener('change', updateWorldFields);
    window.addEventListener('resize', render);
    img.addEventListener('load', render);
    img.addEventListener('click', async (event) => {
      const rect = img.getBoundingClientRect();
      const x = (event.clientX - rect.left) * width / rect.width;
      const y = (event.clientY - rect.top) * height / rect.height;
      clicks.push({
        frame: currentFrame(),
        landmark_id: landmarkSelect.value,
        image: {x, y},
        world: {x: Number(worldX.value), y: Number(worldY.value)}
      });
      render();
      await postClicks();
    });
    document.getElementById('undo').onclick = async () => { clicks.pop(); render(); await postClicks(); };
    document.getElementById('clearFrame').onclick = async () => {
      const frame = currentFrame();
      if (confirm(`Clear all clicks for frame ${frame}?`)) {
        clicks = clicks.filter(c => c.frame !== frame);
        render();
        await postClicks();
      }
    };
    document.getElementById('clearAll').onclick = async () => {
      if (confirm('Clear all clicks?')) {
        clicks = [];
        render();
        await postClicks();
      }
    };
    document.getElementById('save').onclick = postClicks;
    updateWorldFields();
    render();
  </script>
</body>
</html>
"""


class FaceoffClickHandler(BaseHTTPRequestHandler):
    frames: list[dict] = []
    landmarks: list[dict] = LANDMARKS
    frames_dir: Path
    output_path: Path
    width: int
    height: int

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            clicks = []
            if self.output_path.exists():
                saved = json.loads(self.output_path.read_text())
                clicks = saved.get("clicks", [])
            html = (
                HTML.replace("__FRAMES__", json.dumps(self.frames))
                .replace("__WIDTH__", str(self.width))
                .replace("__HEIGHT__", str(self.height))
                .replace("__LANDMARKS__", json.dumps(self.landmarks))
                .replace("__CLICKS__", json.dumps(clicks))
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/frame":
            qs = parse_qs(parsed.query)
            idx = int(qs.get("idx", ["0"])[0])
            path = self.frames_dir / f"frame_{idx:05d}.jpg"
            if not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        clicks = []
        for click in payload.get("clicks", []):
            clicks.append(
                {
                    "frame": int(click["frame"]),
                    "landmark_id": str(click.get("landmark_id", "custom")),
                    "image": {
                        "x": float(click["image"]["x"]),
                        "y": float(click["image"]["y"]),
                    },
                    "world": {
                        "x": float(click["world"]["x"]),
                        "y": float(click["world"]["y"]),
                    },
                }
            )
        prompts = [{"frame": c["frame"], "object_id": SAM2_OBJECT_IDS[c["landmark_id"]], "feature": c["landmark_id"],
                    "points": [[c["image"]["x"], c["image"]["y"]]], "labels": [1]} for c in clicks if c["landmark_id"] in SAM2_OBJECT_IDS]
        data = {"landmarks": self.landmarks, "clicks": clicks, "object_id_map": SAM2_OBJECT_IDS, "prompts": prompts}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(data, indent=2) + "\n")
        self.send_json({"ok": True, "count": len(clicks), "path": str(self.output_path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, help="The clip's frames (frame_NNNNN.jpg), before undistortion.")
    parser.add_argument("--output", required=True, help="Clicks and SAM2 prompts JSON; reopened clicks are loaded from it.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    if not paths:
        raise SystemExit(f"No frame_*.jpg files in {args.frames_dir}")
    with Image.open(paths[0]) as image:
        width, height = image.size
    handler = FaceoffClickHandler
    handler.frames = [{"frame": int(p.stem.split("_")[-1]), "time_sec": int(p.stem.split("_")[-1]) / args.fps} for p in paths]
    handler.width, handler.height = width, height
    handler.frames_dir = Path(args.frames_dir)
    handler.output_path = Path(args.output)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"Faceoff dots: http://127.0.0.1:{args.port}  (clicks and SAM2 prompts are saved to {handler.output_path})")
    server.serve_forever()


if __name__ == "__main__":
    main()
