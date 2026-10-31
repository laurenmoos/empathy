"""Small local HTTP server for the interactive learned-inference demo."""

from __future__ import annotations

import json
import threading
import uuid
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .live_inference import LiveInferenceEngine, LiveInferenceError, LiveParameters


class LiveSessionRegistry:
    """Own fixed-world thinking sessions without reloading either checkpoint."""

    def __init__(self, engine: LiveInferenceEngine) -> None:
        self.engine = engine
        self._sessions: dict[str, tuple[Any, threading.Lock]] = {}
        self._lock = threading.Lock()

    def start(self, parameters: LiveParameters) -> str:
        session = self.engine.thinking_session(parameters)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = (session, threading.Lock())
        return session_id

    def next_episode(self, session_id: str):
        with self._lock:
            entry = self._sessions.get(session_id)
        if entry is None:
            raise KeyError("unknown session ID")
        session, session_lock = entry
        with session_lock:
            return session.next_episode()


def process_inference_request(
    engine: LiveInferenceEngine, body: bytes
) -> tuple[HTTPStatus, dict[str, Any]]:
    """Validate one JSON request and return its HTTP status and response payload."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return HTTPStatus.BAD_REQUEST, {"error": "request body must be valid JSON"}
    required = {"gravity", "wind", "damping"}
    if not isinstance(payload, dict) or set(payload) != required:
        names = ", ".join(sorted(required))
        return HTTPStatus.BAD_REQUEST, {"error": f"request must contain exactly: {names}"}
    try:
        parameters = LiveParameters(**payload)
        result = engine.infer(parameters)
    except (LiveInferenceError, TypeError, ValueError) as error:
        return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)}
    return HTTPStatus.OK, result.to_payload()


def process_session_start(
    registry: LiveSessionRegistry, body: bytes
) -> tuple[HTTPStatus, dict[str, Any]]:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return HTTPStatus.BAD_REQUEST, {"error": "request body must be valid JSON"}
    required = {"gravity", "wind", "damping"}
    if not isinstance(payload, dict) or set(payload) != required:
        names = ", ".join(sorted(required))
        return HTTPStatus.BAD_REQUEST, {"error": f"request must contain exactly: {names}"}
    try:
        parameters = LiveParameters(**payload)
        session_id = registry.start(parameters)
    except (TypeError, ValueError) as error:
        return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)}
    return HTTPStatus.OK, {
        "schema_version": "live-session-v1",
        "session_id": session_id,
        "parameters": {
            "gravity": parameters.gravity,
            "wind": parameters.wind,
            "damping": parameters.damping,
        },
    }


def process_session_next(
    registry: LiveSessionRegistry, body: bytes
) -> tuple[HTTPStatus, dict[str, Any]]:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return HTTPStatus.BAD_REQUEST, {"error": "request body must be valid JSON"}
    if not isinstance(payload, dict) or set(payload) != {"session_id"}:
        return HTTPStatus.BAD_REQUEST, {"error": "request must contain exactly: session_id"}
    if not isinstance(payload["session_id"], str) or not payload["session_id"]:
        return HTTPStatus.BAD_REQUEST, {"error": "session_id must be a nonempty string"}
    try:
        episode = registry.next_episode(payload["session_id"])
    except KeyError:
        return HTTPStatus.NOT_FOUND, {"error": "unknown session ID"}
    except LiveInferenceError as error:
        return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)}
    return HTTPStatus.OK, episode.to_payload()


class _LiveDemoHandler(SimpleHTTPRequestHandler):
    engine: LiveInferenceEngine
    registry: LiveSessionRegistry

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/infer", "/api/session/start", "/api/session/next"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length must be an integer"})
            return
        body = self.rfile.read(length)
        if self.path == "/api/session/start":
            status, payload = process_session_start(self.registry, body)
        elif self.path == "/api/session/next":
            status, payload = process_session_next(self.registry, body)
        else:
            status, payload = process_inference_request(self.engine, body)
        self._json(status, payload)

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
    registry: LiveSessionRegistry | None = None,
) -> ThreadingHTTPServer:
    """Create an injectable server; loading model checkpoints is the caller's job."""
    root = Path(root).resolve()
    if not (root / "explorer.html").is_file():
        raise FileNotFoundError(f"demo page not found: {root / 'explorer.html'}")
    class BoundLiveDemoHandler(_LiveDemoHandler):
        pass

    BoundLiveDemoHandler.engine = engine
    BoundLiveDemoHandler.registry = registry or LiveSessionRegistry(engine)
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
