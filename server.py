"""Stage 7 — local chat server (stdlib only), connected to the real model."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from mind3 import MODEL_NAME, __version__
from mind3.engine import Engine

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
ARTIFACTS = os.path.join(ROOT, "artifacts", "full")
engine = Engine(ARTIFACTS)

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/info":
            return self._send(200, engine.info())
        if path == "/api/history":
            return self._send(200, {"history": engine.history})
        if path == "/":
            path = "/index.html"
        fp = os.path.normpath(os.path.join(STATIC, path.lstrip("/")))
        if fp.startswith(STATIC) and os.path.isfile(fp):
            ext = os.path.splitext(fp)[1]
            with open(fp, "rb") as f:
                data = f.read()
            return self._send(200, data, MIME.get(ext, "application/octet-stream"))
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}
        if path == "/api/chat":
            msg = (payload.get("message") or "").strip()
            if not msg:
                return self._send(400, {"error": "empty message"})
            return self._send(200, engine.answer(msg))
        if path == "/api/clear":
            engine.clear()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"{MODEL_NAME} v{__version__} chat server: http://127.0.0.1:{port} "
          f"(variant={engine.config['variant']}, units={len(engine.units)}, vocab={engine.registry.size})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()