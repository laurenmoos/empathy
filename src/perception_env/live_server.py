"""Small local HTTP server for the interactive learned-inference demo."""

from __future__ import annotations

import json
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .live_inference import LiveInferenceEngine, LiveInferenceError, LiveParameters


class _LiveDemoHandler(SimpleHTTPRequestHandler):
    engine: LiveInferenceEngine

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/infer":
            self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "request body must be valid JSON"})
            return
        required = {"gravity", "wind", "damping"}
        if not isinstance(payload, dict) or set(payload) != required:
            names = ", ".join(sorted(required))
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"request must contain exactly: {names}"},
            )
            return
        try:
            parameters = LiveParameters(**payload)
            result = self.engine.infer(parameters)
        except (LiveInferenceError, TypeError, ValueError) as error:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            return
        self._json(HTTPStatus.OK, result.to_payload())

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.path = "/explorer.html"
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(
    host: str,
    port: int,
    root: Path,
    engine: LiveInferenceEngine,
) -> ThreadingHTTPServer:
    """Create an injectable server; loading model checkpoints is the caller's job."""
    root = Path(root).resolve()
    if not (root / "explorer.html").is_file():
        raise FileNotFoundError(f"demo page not found: {root / 'explorer.html'}")
    class BoundLiveDemoHandler(_LiveDemoHandler):
        pass

    BoundLiveDemoHandler.engine = engine
    handler = partial(BoundLiveDemoHandler, directory=str(root))
    return ThreadingHTTPServer((host, port), handler)


def run_live_server(
    *,
    host: str,
    port: int,
    root: Path,
    model1_checkpoint: Path,
    model2_checkpoint: Path,
) -> None:
    """Load each model once, then serve inference requests until interrupted."""
    engine = LiveInferenceEngine.from_checkpoints(model1_checkpoint, model2_checkpoint)
    server = create_server(host, port, root, engine)
    print(f"Perception environment: http://{host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
