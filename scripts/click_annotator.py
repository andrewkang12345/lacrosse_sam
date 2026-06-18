from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lacrosse Click Prompts</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    header { display: flex; gap: 12px; align-items: center; padding: 10px 14px; background: #1c1c1c; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 3; }
    label { display: inline-flex; gap: 6px; align-items: center; font-size: 14px; }
    input, select, button { font: inherit; color: #eee; background: #2a2a2a; border: 1px solid #555; border-radius: 6px; padding: 6px 8px; }
    button { cursor: pointer; }
    button:hover { background: #383838; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 0; min-height: calc(100vh - 54px); }
    #stageWrap { overflow: auto; display: grid; place-items: start center; padding: 16px; }
    #stage { position: relative; width: min(1280px, 100%); }
    #frame { display: block; width: 100%; height: auto; user-select: none; }
    .dot { position: absolute; width: 18px; height: 18px; border: 2px solid #000; border-radius: 50%; transform: translate(-50%, -50%); display: grid; place-items: center; color: #000; font: 700 10px system-ui; pointer-events: none; }
    .neg { background: #ff3b30 !important; }
    aside { border-left: 1px solid #333; background: #171717; padding: 14px; overflow: auto; }
    #status { color: #aaa; font-size: 13px; min-width: 240px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 5px 4px; border-bottom: 1px solid #303030; text-align: left; }
    .muted { color: #999; font-size: 12px; line-height: 1.35; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #333; } }
  </style>
</head>
<body>
  <header>
    <label>Frame <select id="frameSelect"></select></label>
    <label>Player ID <input id="objId" type="number" min="1" step="1" value="1" style="width:70px"></label>
    <label><input id="negative" type="checkbox"> negative</label>
    <button id="nextObj">Next ID</button>
    <button id="undo">Undo</button>
    <button id="clear">Clear</button>
    <button id="save">Save</button>
    <span id="status"></span>
  </header>
  <main>
    <div id="stageWrap">
      <div id="stage"><img id="frame" draggable="false"><div id="dots"></div></div>
    </div>
    <aside>
      <div class="muted">
        Click each visible player's torso. Reuse the same Player ID for refinement clicks on later frames.
        Negative clicks mark background or mistaken regions for the current ID.
      </div>
      <h3>Clicks</h3>
      <table>
        <thead><tr><th>ID</th><th>Frame</th><th>x</th><th>y</th><th>Label</th></tr></thead>
        <tbody id="clickRows"></tbody>
      </table>
    </aside>
  </main>
  <script>
    const frames = __FRAMES__;
    const width = __WIDTH__;
    const height = __HEIGHT__;
    let clicks = __CLICKS__;
    const frameSelect = document.getElementById('frameSelect');
    const img = document.getElementById('frame');
    const dots = document.getElementById('dots');
    const rows = document.getElementById('clickRows');
    const status = document.getElementById('status');
    const objId = document.getElementById('objId');
    const negative = document.getElementById('negative');

    function colorFor(id) {
      const hue = (id * 137.508) % 360;
      return `hsl(${hue} 90% 60%)`;
    }
    function currentFrame() { return Number(frameSelect.value); }
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
      const rect = img.getBoundingClientRect();
      const scaleX = img.clientWidth / width;
      const scaleY = img.clientHeight / height;
      clicks.forEach((c, i) => {
        if (c.frame !== frame) return;
        const d = document.createElement('div');
        d.className = 'dot' + (c.label === 0 ? ' neg' : '');
        d.style.left = `${c.x * scaleX}px`;
        d.style.top = `${c.y * scaleY}px`;
        d.style.background = colorFor(c.object_id);
        d.textContent = c.object_id;
        dots.appendChild(d);
      });
      rows.innerHTML = clicks.map(c => `<tr><td>${c.object_id}</td><td>${c.frame}</td><td>${c.x}</td><td>${c.y}</td><td>${c.label ? '+' : '-'}</td></tr>`).join('');
    }

    frames.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.frame;
      opt.textContent = `frame ${f.frame} | ${f.time_sec.toFixed(1)}s`;
      frameSelect.appendChild(opt);
    });
    frameSelect.addEventListener('change', render);
    window.addEventListener('resize', render);
    img.addEventListener('load', render);
    img.addEventListener('click', async (event) => {
      const rect = img.getBoundingClientRect();
      const x = Math.round((event.clientX - rect.left) * width / rect.width);
      const y = Math.round((event.clientY - rect.top) * height / rect.height);
      const click = {object_id: Number(objId.value), frame: currentFrame(), x, y, label: negative.checked ? 0 : 1};
      clicks.push(click);
      render();
      await postClicks();
    });
    document.getElementById('nextObj').onclick = () => { objId.value = Number(objId.value) + 1; negative.checked = false; };
    document.getElementById('undo').onclick = async () => { clicks.pop(); render(); await postClicks(); };
    document.getElementById('clear').onclick = async () => {
      if (confirm('Clear all clicks?')) { clicks = []; render(); await postClicks(); }
    };
    document.getElementById('save').onclick = postClicks;
    render();
  </script>
</body>
</html>
"""


class AnnotatorHandler(BaseHTTPRequestHandler):
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
        clicks = payload.get("clicks", [])
        prompts = []
        for click in clicks:
            prompts.append(
                {
                    "object_id": int(click["object_id"]),
                    "frame": int(click["frame"]),
                    "x": int(click["x"]),
                    "y": int(click["y"]),
                    "label": int(click.get("label", 1)),
                }
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps({"prompts": prompts, "clicks": prompts}, indent=2) + "\n")
        self.send_json({"ok": True, "count": len(prompts), "path": str(self.output_path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/prompt_frames/manifest.json")
    parser.add_argument("--frames-dir", default="data/frames_10fps")
    parser.add_argument("--output", default="outputs/click_prompts.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    handler = AnnotatorHandler
    handler.frames = manifest["frames"]
    handler.width = int(manifest["width"])
    handler.height = int(manifest["height"])
    handler.frames_dir = Path(args.frames_dir)
    handler.output_path = Path(args.output)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Click annotator: http://127.0.0.1:{args.port}")
    print(f"Writing clicks to: {handler.output_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
