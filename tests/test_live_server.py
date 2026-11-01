import json
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path

import pytest

from empathy.live.inference import LiveInferenceError, LiveParameters
from empathy.live.server import (
    LiveSessionRegistry,
    create_server,
    process_inference_request,
    process_session_close,
    process_session_next,
    process_session_start,
    run_live_server,
)


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


def test_server_bootstrap_loads_only_the_model1_checkpoint(monkeypatch, tmp_path) -> None:
    loaded = []

    class FakeServer:
        server_port = 0

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            return None

    def from_checkpoint(path):
        loaded.append(path)
        return FakeEngine()

    monkeypatch.setattr(
        "empathy.live.server.LiveInferenceEngine.from_checkpoint",
        from_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(
        "empathy.live.server.create_server",
        lambda host, port, root, engine: FakeServer(),
    )

    run_live_server(
        host="127.0.0.1",
        port=0,
        root=tmp_path,
        model1_checkpoint=Path("model1.pt"),
    )

    assert loaded == [Path("model1.pt")]


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


class FakeThinkingEpisode:
    def __init__(self, index: int) -> None:
        self.index = index

    def to_payload(self):
        return {"schema_version": "live-thinking-v1", "episode_index": self.index}


class FakeThinkingSession:
    def __init__(self, parameters, *, fail: bool = False) -> None:
        self.parameters = parameters
        self.fail = fail
        self.index = 0
        self.closed = False

    def next_episode(self):
        if self.fail:
            raise LiveInferenceError("no valid offset produced an occluded reappearance")
        episode = FakeThinkingEpisode(self.index)
        self.index += 1
        return episode

    def close(self):
        self.closed = True


class FakeThinkingEngine:
    def __init__(self) -> None:
        self.parameters = []
        self.sessions = []

    def thinking_session(self, parameters):
        self.parameters.append(parameters)
        session = FakeThinkingSession(parameters, fail=parameters.wind == -1.0)
        self.sessions.append(session)
        return session


def test_session_endpoints_lock_parameters_and_advance_distinct_episodes() -> None:
    engine = FakeThinkingEngine()
    registry = LiveSessionRegistry(engine)
    body = json.dumps({"gravity": -0.1, "wind": 0.2, "damping": 0.3}).encode()

    status, started = process_session_start(registry, body)
    first_status, first = process_session_next(
        registry, json.dumps({"session_id": started["session_id"]}).encode()
    )
    second_status, second = process_session_next(
        registry, json.dumps({"session_id": started["session_id"]}).encode()
    )

    assert status is HTTPStatus.OK
    assert first_status is second_status is HTTPStatus.OK
    assert first["episode_index"] == 0
    assert second["episode_index"] == 1
    assert len(engine.parameters) == 1
    assert engine.parameters[0].gravity == pytest.approx(-0.1)
    assert engine.parameters[0].wind == pytest.approx(0.2)
    assert engine.parameters[0].damping == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("operation", "body", "status", "message"),
    [
        ("start", b"bad json", HTTPStatus.BAD_REQUEST, "valid JSON"),
        (
            "start",
            json.dumps({"gravity": 0, "wind": 0}).encode(),
            HTTPStatus.BAD_REQUEST,
            "damping",
        ),
        (
            "next",
            json.dumps({"session_id": "missing"}).encode(),
            HTTPStatus.NOT_FOUND,
            "unknown session",
        ),
    ],
)
def test_session_request_errors_are_actionable(operation, body, status, message) -> None:
    registry = LiveSessionRegistry(FakeThinkingEngine())

    returned_status, payload = (
        process_session_start(registry, body)
        if operation == "start"
        else process_session_next(registry, body)
    )

    assert returned_status is status
    assert message in payload["error"]


def test_exhausted_offset_bank_returns_422() -> None:
    registry = LiveSessionRegistry(FakeThinkingEngine())
    _, started = process_session_start(
        registry,
        json.dumps({"gravity": 0, "wind": -1.0, "damping": 0}).encode(),
    )

    status, payload = process_session_next(
        registry, json.dumps({"session_id": started["session_id"]}).encode()
    )

    assert status is HTTPStatus.UNPROCESSABLE_ENTITY
    assert "valid offset" in payload["error"]


def test_registry_serializes_concurrent_requests_for_one_session() -> None:
    class RacingSession(FakeThinkingSession):
        def next_episode(self):
            index = self.index
            time.sleep(0.02)
            self.index = index + 1
            return FakeThinkingEpisode(index)

    class RacingEngine(FakeThinkingEngine):
        def thinking_session(self, parameters):
            return RacingSession(parameters)

    registry = LiveSessionRegistry(RacingEngine())
    session_id = registry.start(LiveParameters())

    with ThreadPoolExecutor(max_workers=2) as pool:
        episodes = list(pool.map(lambda _: registry.next_episode(session_id), range(2)))

    assert sorted(episode.index for episode in episodes) == [0, 1]


def test_registry_evicts_oldest_idle_session_at_capacity() -> None:
    engine = FakeThinkingEngine()
    registry = LiveSessionRegistry(engine, max_sessions=2)
    first = registry.start(LiveParameters())
    second = registry.start(LiveParameters(gravity=-0.1))
    third = registry.start(LiveParameters(wind=0.1))

    with pytest.raises(KeyError, match="unknown session"):
        registry.next_episode(first)
    assert engine.sessions[0].closed
    assert registry.next_episode(second).index == 0
    assert registry.next_episode(third).index == 0


def test_session_close_releases_the_registered_model_copy() -> None:
    engine = FakeThinkingEngine()
    registry = LiveSessionRegistry(engine)
    session_id = registry.start(LiveParameters())

    status, payload = process_session_close(
        registry, json.dumps({"session_id": session_id}).encode()
    )

    assert status is HTTPStatus.OK
    assert payload == {"closed": True}
    assert engine.sessions[0].closed
    with pytest.raises(KeyError, match="unknown session"):
        registry.next_episode(session_id)


def test_close_during_active_request_releases_session_after_request_finishes() -> None:
    from threading import Event

    entered = Event()
    release = Event()

    class BlockingSession(FakeThinkingSession):
        def next_episode(self):
            entered.set()
            assert release.wait(1.0)
            return super().next_episode()

    class BlockingEngine(FakeThinkingEngine):
        def thinking_session(self, parameters):
            session = BlockingSession(parameters)
            self.sessions.append(session)
            return session

    engine = BlockingEngine()
    registry = LiveSessionRegistry(engine)
    session_id = registry.start(LiveParameters())
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(registry.next_episode, session_id)
        assert entered.wait(1.0)
        assert registry.close(session_id)
        assert not engine.sessions[0].closed
        release.set()
        pending.result()
    assert engine.sessions[0].closed
