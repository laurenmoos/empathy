from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from empathy.live.observer import (
    AsyncActivationModel2Observer,
    AsyncModel2Observer,
    Model1ActivationCheckpoint,
    Model1BeliefCheckpoint,
)
from empathy.model2.activation_adapter import Model1ActivationPacket
from empathy.model2.contracts import BeliefInputs, WorldPrediction
from empathy.model2.imagination import ActivationImaginationLoop
from empathy.physics.world import WorldSchema


def _inputs(value: float) -> BeliefInputs:
    beliefs = np.zeros((1, 4, 3, 8, 8), dtype=np.float32)
    beliefs[0, 3, 1, 4, 4] = value
    beliefs[0, 3, 2] = beliefs[0, 3, 1] - beliefs[0, 3, 0]
    return BeliefInputs(beliefs, np.array([[False, False, False, True]]))


def _prediction(value: float) -> WorldPrediction:
    return WorldPrediction(
        torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]]),
        torch.full((1, 7), value),
    )


def test_submit_is_nonblocking_and_processes_the_newest_pending_checkpoint() -> None:
    started = threading.Event()
    release = threading.Event()
    seen: list[float] = []

    def predict(inputs: BeliefInputs) -> WorldPrediction:
        value = float(inputs.beliefs[0, 3, 1, 4, 4])
        seen.append(value)
        if np.isclose(value, 0.1):
            started.set()
            assert release.wait(1.0)
        return _prediction(value)

    observer = AsyncModel2Observer(predict)
    before = time.perf_counter()
    observer.submit(Model1BeliefCheckpoint(1, _inputs(0.1)))
    assert time.perf_counter() - before < 0.05
    assert started.wait(1.0)

    observer.submit(Model1BeliefCheckpoint(2, _inputs(0.2)))
    observer.submit(Model1BeliefCheckpoint(3, _inputs(0.3)))
    release.set()

    result = observer.wait_for_version(3, timeout=1.0)
    observer.close()
    assert result is not None
    assert result.version == 3
    assert seen == pytest.approx([0.1, 0.3])


def test_checkpoint_copies_inputs_before_background_prediction() -> None:
    entered = threading.Event()
    release = threading.Event()
    source = np.zeros((1, 4, 3, 8, 8), dtype=np.float32)
    source[0, 3, 1, 4, 4] = 0.4
    source[0, 3, 2] = source[0, 3, 1]
    inputs = BeliefInputs(source, np.array([[False, False, False, True]]))

    def predict(copied: BeliefInputs) -> WorldPrediction:
        entered.set()
        assert release.wait(1.0)
        return _prediction(float(copied.beliefs[0, 3, 1, 4, 4]))

    observer = AsyncModel2Observer(predict)
    checkpoint = Model1BeliefCheckpoint(4, inputs)
    observer.submit(checkpoint)
    assert entered.wait(1.0)
    source.fill(0.9)
    release.set()

    result = observer.wait_for_version(4, timeout=1.0)
    observer.close()
    assert result is not None
    assert float(result.prediction.normalized_parameters[0, 0]) == pytest.approx(0.4)
    assert not checkpoint.inputs.beliefs.flags.writeable


def test_prediction_failure_keeps_the_last_completed_guess_and_worker_continues() -> None:
    failed = threading.Event()

    def predict(inputs: BeliefInputs) -> WorldPrediction:
        value = float(inputs.beliefs[0, 3, 1, 4, 4])
        if np.isclose(value, 0.2):
            failed.set()
            raise RuntimeError("search failed")
        return _prediction(value)

    observer = AsyncModel2Observer(predict)
    observer.submit(Model1BeliefCheckpoint(1, _inputs(0.1)))
    first = observer.wait_for_version(1, timeout=1.0)
    observer.submit(Model1BeliefCheckpoint(2, _inputs(0.2)))
    assert failed.wait(1.0)
    observer.submit(Model1BeliefCheckpoint(3, _inputs(0.3)))
    third = observer.wait_for_version(3, timeout=1.0)
    observer.close()

    assert first is not None and third is not None
    assert third.version == 3
    assert observer.last_error is None


def test_close_rejects_new_work_without_waiting_for_active_prediction() -> None:
    started = threading.Event()
    release = threading.Event()

    def predict(inputs: BeliefInputs) -> WorldPrediction:
        del inputs
        started.set()
        release.wait(1.0)
        return _prediction(0.5)

    observer = AsyncModel2Observer(predict)
    observer.submit(Model1BeliefCheckpoint(1, _inputs(0.1)))
    assert started.wait(1.0)
    before = time.perf_counter()
    observer.close()
    assert time.perf_counter() - before < 0.05
    with pytest.raises(RuntimeError, match="closed"):
        observer.submit(Model1BeliefCheckpoint(2, _inputs(0.2)))
    release.set()


def test_activation_observer_processes_imagination_without_raw_perceptions() -> None:
    packet = Model1ActivationPacket(
        "episode-0",
        0,
        "online-1",
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )

    class Source:
        def activations_for(self, world):
            del world
            return (packet,)

    imagination = ActivationImaginationLoop((packet,), Source())
    seen = []

    def predict(loop: ActivationImaginationLoop) -> WorldPrediction:
        seen.append(loop)
        assert not hasattr(loop.observed[0], "frames")
        loop.discrepancies((WorldSchema.v1().world(),))
        return _prediction(0.4)

    observer = AsyncActivationModel2Observer(predict)
    observer.submit(Model1ActivationCheckpoint(1, imagination))
    result = observer.wait_for_version(1, timeout=1.0)
    observer.close()

    assert result is not None and result.version == 1
    assert seen == [imagination]

