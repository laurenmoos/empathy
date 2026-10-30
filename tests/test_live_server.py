import json
from http import HTTPStatus
from pathlib import Path

import pytest

from perception_env.live_inference import LiveInferenceError
from perception_env.live_server import create_server, process_inference_request


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


def test_inference_request_runs_injected_engine_without_a_socket() -> None:
    engine = FakeEngine()
    body = json.dumps({"gravity": -0.1, "wind": 0.2, "damping": 0.3}).encode()

    status, payload = process_inference_request(engine, body)

    assert status is HTTPStatus.OK
    assert payload == {"schema_version": "live-inference-v1", "ok": True}
    assert engine.calls[0].gravity == pytest.approx(-0.1)
    assert engine.calls[0].wind == pytest.approx(0.2)
    assert engine.calls[0].damping == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        (b"not json", HTTPStatus.BAD_REQUEST, "valid JSON"),
        (json.dumps({"gravity": 0, "wind": 0}).encode(), HTTPStatus.BAD_REQUEST, "damping"),
        (
            json.dumps({"gravity": 0, "wind": -1, "damping": 0}).encode(),
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "reappearance",
        ),
        (
            json.dumps({"gravity": True, "wind": 0, "damping": 0}).encode(),
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "booleans",
        ),
    ],
)
def test_inference_errors_are_actionable(body, status, message) -> None:
    returned_status, payload = process_inference_request(FakeEngine(), body)

    assert returned_status is status
    assert message in payload["error"]


def test_server_validates_demo_root_before_binding(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="explorer.html"):
        create_server("127.0.0.1", 0, tmp_path, FakeEngine())
