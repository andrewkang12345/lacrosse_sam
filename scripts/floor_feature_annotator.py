from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


FEATURES = [
    {
        "id": "left_restraining_line",
        "label": "Left restraining line",
        "color": "#ff7a45",
        "description": "Click anywhere along the x=57.5 ft restraining/attack line.",
        "world": {"type": "line", "x": 57.5, "y1": 0.0, "y2": 85.0},
    },
    {
        "id": "right_restraining_line",
        "label": "Right restraining line",
        "color": "#5aa7ff",
        "description": "Click anywhere along the x=142.5 ft restraining/attack line.",
        "world": {"type": "line", "x": 142.5, "y1": 0.0, "y2": 85.0},
    },
    {
        "id": "goal_crease",
        "label": "Goal crease",
        "color": "#f7d34a",
        "description": "Click anywhere on either NLL crease outline.",
        "world": {
            "type": "nll_crease_pair",
            "centers": [[12.0, 42.5], [188.0, 42.5]],
            "radius": 9.25,
            "rear_goal_base_ft": 4.5,
            "straight_chord_behind_goal_base_ft": 1.0,
        },
    },
    {
        "id": "midfield_line",
        "label": "Midfield line",
        "color": "#64d47b",
        "description": "Click anywhere along the x=100 ft centerline.",
        "world": {"type": "line", "x": 100.0, "y1": 0.0, "y2": 85.0},
    },
    {
        "id": "field_outline",
        "label": "Field outline",
        "color": "#d77cff",
        "description": "Click anywhere on the yellow rounded-rectangle field boundary.",
        "world": {"type": "rounded_rect_outline", "length": 200.0, "width": 85.0, "corner_radius": 22.667},
    },
]


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Floor Feature Annotation</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    header { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 14px; background: #1b1b1b; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 3; }
    label { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; }
    input, select, button { font: inherit; color: #eee; background: #292929; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    button { cursor: pointer; }
    button:hover { background: #3a3a3a; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 410px; min-height: calc(100vh - 58px); }
    #stageWrap { overflow: auto; display: grid; place-items: start center; padding: 16px; }
    #stage { position: relative; width: min(1280px, 100%); }
    #frame { display: block; width: 100%; height: auto; user-select: none; }
    .dot { position: absolute; width: 18px; height: 18px; border: 2px solid #000; border-radius: 50%; transform: translate(-50%, -50%); color: #000; display: grid; place-items: center; font: 800 10px system-ui; pointer-events: none; }
    aside { border-left: 1px solid #333; background: #171717; padding: 14px; overflow: auto; }
    #status { color: #bbb; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 5px 4px; border-bottom: 1px solid #303030; text-align: left; }
    .muted { color: #aaa; font-size: 12px; line-height: 1.35; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
    .ok { color: #7bd88f; }
    .warn { color: #ffcf70; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #333; } }
  </style>
</head>
<body>
  <header>
    <label>Frame <select id="frameSelect"></select></label>
    <label>Feature <select id="featureSelect"></select></label>
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
        Choose the coarse feature category, then click anywhere on that visible feature.
        These are point-on-feature constraints, not exact landmark/intersection clicks.
      </div>
      <h3>Feature Key</h3>
      <div id="featureKey" class="muted"></div>
      <h3>Frame Status</h3>
      <div id="frameStatus" class="muted"></div>
      <h3>Clicks</h3>
      <table>
        <thead><tr><th>#</th><th>Frame</th><th>Feature</th><th>x</th><th>y</th></tr></thead>
        <tbody id="clickRows"></tbody>
      </table>
    </aside>
  </main>
  <script>
    const frames = __FRAMES__;
    const width = __WIDTH__;
    const height = __HEIGHT__;
    const features = __FEATURES__;
    let clicks = __CLICKS__;
    const frameSelect = document.getElementById('frameSelect');
    const featureSelect = document.getElementById('featureSelect');
    const img = document.getElementById('frame');
    const dots = document.getElementById('dots');
    const rows = document.getElementById('clickRows');
    const status = document.getElementById('status');
    const frameStatus = document.getElementById('frameStatus');
    const featureKey = document.getElementById('featureKey');

    function currentFrame() { return Number(frameSelect.value); }
    function currentFeature() { return features.find(f => f.id === featureSelect.value); }
    function featureById(id) { return features.find(f => f.id === id) || features[0]; }
    function frameUrl(frame) { return `/frame?idx=${frame}&t=${Date.now()}`; }
    function setStatus(text) { status.textContent = text; }
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
        const f = featureById(c.feature);
        const d = document.createElement('div');
        d.className = 'dot';
        d.style.left = `${c.image.x * scaleX}px`;
        d.style.top = `${c.image.y * scaleY}px`;
        d.style.background = f.color;
        d.textContent = i + 1;
        dots.appendChild(d);
      });
      rows.innerHTML = clicks.map((c, i) => {
        const f = featureById(c.feature);
        return `<tr><td>${i + 1}</td><td>${c.frame}</td><td><span class="swatch" style="background:${f.color}"></span>${f.label}</td><td>${Math.round(c.image.x)}</td><td>${Math.round(c.image.y)}</td></tr>`;
      }).join('');
      const frameClicks = clicks.filter(c => c.frame === frame);
      const byFeature = {};
      frameClicks.forEach(c => { byFeature[c.feature] = (byFeature[c.feature] || 0) + 1; });
      frameStatus.innerHTML = features.map(f => {
        const count = byFeature[f.id] || 0;
        return `<div><span class="swatch" style="background:${f.color}"></span>${f.label}: <span class="${count ? 'ok' : 'warn'}">${count}</span></div>`;
      }).join('');
    }
    frames.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.frame;
      opt.textContent = `frame ${f.frame} | ${f.time_sec.toFixed(1)}s`;
      frameSelect.appendChild(opt);
    });
    features.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.id;
      opt.textContent = f.label;
      featureSelect.appendChild(opt);
    });
    featureKey.innerHTML = features.map(f => `<div><span class="swatch" style="background:${f.color}"></span><b>${f.label}</b>: ${f.description}</div>`).join('');
    frameSelect.addEventListener('change', render);
    featureSelect.addEventListener('change', render);
    window.addEventListener('resize', render);
    img.addEventListener('load', render);
    img.addEventListener('click', async (event) => {
      const rect = img.getBoundingClientRect();
      const x = (event.clientX - rect.left) * width / rect.width;
      const y = (event.clientY - rect.top) * height / rect.height;
      clicks.push({frame: currentFrame(), feature: currentFeature().id, image: {x, y}});
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
      if (confirm('Clear all feature clicks?')) {
        clicks = [];
        render();
        await postClicks();
      }
    };
    document.getElementById('save').onclick = postClicks;
    render();
  </script>
</body>
</html>
"""


class FeatureClickHandler(BaseHTTPRequestHandler):
    frames: list[dict] = []
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
                .replace("__FEATURES__", json.dumps(FEATURES))
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
        valid_features = {feature["id"] for feature in FEATURES}
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        clicks = []
        for click in payload.get("clicks", []):
            feature = str(click["feature"])
            if feature not in valid_features:
                continue
            clicks.append(
                {
                    "frame": int(click["frame"]),
                    "feature": feature,
                    "image": {
                        "x": float(click["image"]["x"]),
                        "y": float(click["image"]["y"]),
                    },
                }
            )
        data = {
            "schema": "floor_feature_clicks_v1",
            "world_units": "feet",
            "floor": {
                "length_ft": 200.0,
                "width_ft": 85.0,
                "corner_radius_ft": 22.667,
            },
            "features": FEATURES,
            "clicks": clicks,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(data, indent=2) + "\n")
        self.send_json({"ok": True, "count": len(clicks), "path": str(self.output_path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/prompt_frames/manifest.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--output", default="outputs/floor_feature_clicks.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    handler = FeatureClickHandler
    handler.frames = manifest["frames"]
    handler.width = int(manifest["width"])
    handler.height = int(manifest["height"])
    handler.frames_dir = Path(args.frames_dir)
    handler.output_path = Path(args.output)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Floor feature annotator: http://127.0.0.1:{args.port}")
    print(f"Writing clicks to: {handler.output_path}")
    print("Select a coarse feature category, then click points along that feature.")
    server.serve_forever()


if __name__ == "__main__":
    main()
