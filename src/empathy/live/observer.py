"""Asynchronous observer for versioned Model 1 belief checkpoints."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from empathy.model2.contracts import BeliefInputs, WorldPrediction
from empathy.model2.imagination import ActivationImaginationLoop


@dataclass(frozen=True)
class Model1BeliefCheckpoint:
    """Immutable belief-only output produced by one Model 1 version."""

    version: int
    inputs: BeliefInputs

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0:
            raise ValueError("checkpoint version must be a nonnegative integer")
        if type(self.inputs) is not BeliefInputs:
            raise TypeError("checkpoint inputs must be BeliefInputs")
        object.__setattr__(
            self,
            "inputs",
            BeliefInputs(self.inputs.beliefs, self.inputs.history_mask),
        )


@dataclass(frozen=True)
class Model2CheckpointGuess:
    """Latest Model 2 prediction and the Model 1 version it observed."""

    version: int
    prediction: WorldPrediction


class AsyncModel2Observer:
    """Coalesce Model 1 checkpoints and search on one background worker."""

    def __init__(
        self,
        predict: Callable[[BeliefInputs], WorldPrediction],
    ) -> None:
        self._predict = predict
        self._condition = threading.Condition()
        self._pending: Model1BeliefCheckpoint | None = None
        self._latest: Model2CheckpointGuess | None = None
        self._last_submitted = -1
        self._last_error: Exception | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="empathy-model2-observer",
            daemon=True,
        )
        self._worker.start()

    @property
    def latest(self) -> Model2CheckpointGuess | None:
        with self._condition:
            return self._latest

    @property
    def last_error(self) -> Exception | None:
        with self._condition:
            return self._last_error

    def submit(self, checkpoint: Model1BeliefCheckpoint) -> None:
        if type(checkpoint) is not Model1BeliefCheckpoint:
            raise TypeError("observer accepts Model1BeliefCheckpoint instances")
        copied = Model1BeliefCheckpoint(checkpoint.version, checkpoint.inputs)
        with self._condition:
            if self._closed:
                raise RuntimeError("observer is closed")
            if copied.version <= self._last_submitted:
                raise ValueError("checkpoint versions must increase")
            self._last_submitted = copied.version
            self._pending = copied
            self._condition.notify()

    def wait_for_version(
        self, version: int, *, timeout: float
    ) -> Model2CheckpointGuess | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                not self._closed
                and (self._latest is None or self._latest.version < version)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._latest is not None and self._latest.version >= version:
                return self._latest
            return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                checkpoint = self._pending
                self._pending = None
            assert checkpoint is not None
            try:
                prediction = self._predict(checkpoint.inputs)
            except Exception as error:  # isolate Model 2 from the seeing stream
                with self._condition:
                    self._last_error = error
                    self._condition.notify_all()
                continue
            result = Model2CheckpointGuess(checkpoint.version, prediction)
            with self._condition:
                if self._closed:
                    return
                if self._latest is None or result.version > self._latest.version:
                    self._latest = result
                self._last_error = None
                self._condition.notify_all()


@dataclass(frozen=True)
class Model1ActivationCheckpoint:
    """One frozen Model 1 version and its activation-only imagination loop."""

    version: int
    imagination: ActivationImaginationLoop

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0:
            raise ValueError("checkpoint version must be a nonnegative integer")
        if type(self.imagination) is not ActivationImaginationLoop:
            raise TypeError("checkpoint imagination must be ActivationImaginationLoop")


class AsyncActivationModel2Observer:
    """Coalesce frozen activation-imagination checkpoints in one worker."""

    def __init__(
        self,
        predict: Callable[[ActivationImaginationLoop], WorldPrediction],
    ) -> None:
        self._predict = predict
        self._condition = threading.Condition()
        self._pending: Model1ActivationCheckpoint | None = None
        self._latest: Model2CheckpointGuess | None = None
        self._last_submitted = -1
        self._last_error: Exception | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="empathy-activation-model2-observer",
            daemon=True,
        )
        self._worker.start()

    @property
    def latest(self) -> Model2CheckpointGuess | None:
        with self._condition:
            return self._latest

    @property
    def last_error(self) -> Exception | None:
        with self._condition:
            return self._last_error

    def submit(self, checkpoint: Model1ActivationCheckpoint) -> None:
        if type(checkpoint) is not Model1ActivationCheckpoint:
            raise TypeError("observer accepts Model1ActivationCheckpoint instances")
        with self._condition:
            if self._closed:
                raise RuntimeError("observer is closed")
            if checkpoint.version <= self._last_submitted:
                raise ValueError("checkpoint versions must increase")
            self._last_submitted = checkpoint.version
            self._pending = checkpoint
            self._condition.notify()

    def wait_for_version(
        self, version: int, *, timeout: float
    ) -> Model2CheckpointGuess | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                not self._closed
                and (self._latest is None or self._latest.version < version)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._latest is not None and self._latest.version >= version:
                return self._latest
            return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                checkpoint = self._pending
                self._pending = None
            assert checkpoint is not None
            try:
                prediction = self._predict(checkpoint.imagination)
            except Exception as error:
                with self._condition:
                    self._last_error = error
                    self._condition.notify_all()
                continue
            result = Model2CheckpointGuess(checkpoint.version, prediction)
            with self._condition:
                if self._closed:
                    return
                if self._latest is None or result.version > self._latest.version:
                    self._latest = result
                self._last_error = None
                self._condition.notify_all()

