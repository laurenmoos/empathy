"""Small local HTTP server for the interactive learned-inference demo."""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .inference import LiveInferenceEngine, LiveInferenceError, LiveParameters


@dataclass
class _SessionEntry:
    session: Any
    request_lock: threading.Lock
    active_requests: int = 0
    closing: bool = False


class LiveSessionRegistry:
    """Own fixed-world thinking sessions without reloading either checkpoint."""

    def __init__(self, engine: LiveInferenceEngine, *, max_sessions: int = 8) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.engine = engine
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionEntry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _close_session(entry: _SessionEntry) -> None:
        close = getattr(entry.session, "close", None)
        if close is not None:
            close()

    def start(self, parameters: LiveParameters) -> str:
        evicted: list[_SessionEntry] = []
        try:
            with self._lock:
                while len(self._sessions) >= self.max_sessions:
                    idle_id = next(
                        (
                            session_id
                            for session_id, entry in self._sessions.items()
                            if entry.active_requests == 0
                        ),
                        None,
                    )
                    if idle_id is None:
                        raise LiveInferenceError(
                            "all live session slots are busy; retry shortly"
                        )
                    evicted.append(self._sessions.pop(idle_id))
                session = self.engine.thinking_session(parameters)
                session_id = uuid.uuid4().hex
                self._sessions[session_id] = _SessionEntry(session, threading.Lock())
        finally:
            for entry in evicted:
                self._close_session(entry)
        return session_id

    def next_episode(self, session_id: str):
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or entry.closing:
                raise KeyError("unknown session ID")
            entry.active_requests += 1
            self._sessions.move_to_end(session_id)
        try:
            with entry.request_lock:
                return entry.session.next_episode()
        finally:
            closed_entry = None
            with self._lock:
                entry.active_requests -= 1
                if entry.closing and entry.active_requests == 0:
                    closed_entry = self._sessions.pop(session_id, None)
            if closed_entry is not None:
                self._close_session(closed_entry)

    def close(self, session_id: str) -> bool:
        closed_entry = None
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            entry.closing = True
            if entry.active_requests == 0:
                closed_entry = self._sessions.pop(session_id)
        if closed_entry is not None:
            self._close_session(closed_entry)
        return True


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


def process_session_close(
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
    return HTTPStatus.OK, {"closed": registry.close(payload["session_id"])}


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
        if self.path not in {
            "/api/infer",
            "/api/session/start",
            "/api/session/next",
            "/api/session/close",
        }:
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
        elif self.path == "/api/session/close":
            status, payload = process_session_close(self.registry, body)
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
) -> None:
    """Load the learned rollout model once, then serve until interrupted."""
    engine = LiveInferenceEngine.from_checkpoint(model1_checkpoint)
    server = create_server(host, port, root, engine)
    print(f"Empathy: http://{host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
