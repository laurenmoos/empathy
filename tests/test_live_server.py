import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from perception_env.live_inference import LiveInferenceError
from perception_env.live_server import create_server


class FakeResult:
    def to_payload(self) -> dict[str, object]:
        return {"schema_version": "live-inference-v1", "ok": True}


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def infer(self, parameters):
        self.calls.append(parameters)
        if parameters.wind == -1.0:
            raise LiveInferenceError("world has no valid reappearance")
        return FakeResult()


@pytest.fixture
def running_server(tmp_path: Path):
    (tmp_path / "index.html").write_text("<h1>demo</h1>")
    engine = FakeEngine()
    server = create_server("127.0.0.1", 0, tmp_path, engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", engine
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _post(url: str, payload: bytes):
    request = Request(
        f"{url}/api/infer",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request)


def test_serves_demo_and_runs_injected_engine(running_server) -> None:
    url, engine = running_server

    with urlopen(f"{url}/") as response:
        assert response.read() == b"<h1>demo</h1>"
    body = json.dumps({"gravity": -0.1, "wind": 0.2, "damping": 0.3}).encode()
    with _post(url, body) as response:
        payload = json.load(response)

    assert payload == {"schema_version": "live-inference-v1", "ok": True}
    assert engine.calls[0].gravity == pytest.approx(-0.1)
    assert engine.calls[0].wind == pytest.approx(0.2)
    assert engine.calls[0].damping == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        (b"not json", 400, "valid JSON"),
        (json.dumps({"gravity": 0, "wind": 0}).encode(), 400, "damping"),
        (
            json.dumps({"gravity": 0, "wind": -1, "damping": 0}).encode(),
            422,
            "reappearance",
        ),
    ],
)
def test_inference_errors_are_actionable(running_server, body, status, message) -> None:
    url, _ = running_server

    with pytest.raises(HTTPError) as caught:
        _post(url, body)

    assert caught.value.code == status
    payload = json.loads(caught.value.read())
    assert message in payload["error"]
