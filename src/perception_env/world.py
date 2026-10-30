"""Versioned latent-world descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    minimum: float
    maximum: float
    neutral_interval: tuple[float, float] | None = None

    def normalize(self, value: float) -> float:
        return (value - self.minimum) / (self.maximum - self.minimum)

    def denormalize(self, value: float) -> float:
        return self.minimum + value * (self.maximum - self.minimum)

    def validate(self, value: float, *, active: bool) -> None:
        if not self.minimum <= value <= self.maximum:
            raise ValueError(f"{self.name}={value} outside [{self.minimum}, {self.maximum}]")
        if active and self.neutral_interval is not None:
            low, high = self.neutral_interval
            if low <= value <= high:
                raise ValueError(f"active {self.name} lies in neutral interval [{low}, {high}]")


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    parameters: tuple[ParameterSpec, ...]


@dataclass(frozen=True)
class OperatorSlot:
    name: str
    active: bool
    parameters: tuple[float, ...]

    def parameter_dict(self, spec: OperatorSpec) -> dict[str, float]:
        return {parameter.name: value for parameter, value in zip(spec.parameters, self.parameters)}


@dataclass(frozen=True)
class WorldCode:
    schema_version: str
    slots: tuple[OperatorSlot, ...]

    def slot(self, name: str) -> OperatorSlot:
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise KeyError(name)

    def to_flat(self, schema: WorldSchema) -> np.ndarray:
        schema.validate(self)
        values: list[float] = []
        for spec, slot in zip(schema.specs, self.slots):
            values.append(float(slot.active))
            if slot.active:
                values.extend(
                    parameter.normalize(value)
                    for parameter, value in zip(spec.parameters, slot.parameters)
                )
            else:
                values.extend(0.0 for _ in spec.parameters)
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class WorldSchema:
    version: str
    specs: tuple[OperatorSpec, ...]

    @classmethod
    def v1(cls) -> WorldSchema:
        acceleration_y = ParameterSpec("a_y", -1.0, 1.0, (-0.05, 0.05))
        acceleration_x = ParameterSpec("a_x", -1.0, 1.0, (-0.05, 0.05))
        damping = ParameterSpec("lambda", 0.0, 1.0, (0.0, 0.01))
        position = ParameterSpec("position", 0.05, 0.95)
        restitution = ParameterSpec("restitution", 0.0, 1.0)
        return cls(
            version="v1",
            specs=(
                OperatorSpec("vertical_acceleration", (acceleration_y,)),
                OperatorSpec("horizontal_acceleration", (acceleration_x,)),
                OperatorSpec("damping", (damping,)),
                OperatorSpec("vertical_wall", (position, restitution)),
                OperatorSpec("horizontal_wall", (position, restitution)),
            ),
        )

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def field_order(self) -> tuple[str, ...]:
        fields: list[str] = []
        for spec in self.specs:
            fields.append(f"{spec.name}.active")
            fields.extend(f"{spec.name}.{parameter.name}" for parameter in spec.parameters)
        return tuple(fields)

    def spec(self, name: str) -> OperatorSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def world(
        self,
        *,
        inactive_parameters: Mapping[str, Mapping[str, float]] | None = None,
        **active_slots: Mapping[str, float],
    ) -> WorldCode:
        unknown = set(active_slots) - set(self.slot_names)
        if unknown:
            raise ValueError(f"unknown operator slots: {sorted(unknown)}")
        inactive_parameters = inactive_parameters or {}
        slots: list[OperatorSlot] = []
        for spec in self.specs:
            if spec.name not in active_slots:
                slots.append(
                    OperatorSlot(spec.name, False, tuple(0.0 for _ in spec.parameters))
                )
                continue
            supplied = active_slots[spec.name]
            expected = {parameter.name for parameter in spec.parameters}
            if set(supplied) != expected:
                raise ValueError(
                    f"{spec.name} parameters must be {sorted(expected)}, got {sorted(supplied)}"
                )
            values = tuple(float(supplied[parameter.name]) for parameter in spec.parameters)
            for parameter, value in zip(spec.parameters, values):
                parameter.validate(value, active=True)
            slots.append(OperatorSlot(spec.name, True, values))
        world = WorldCode(self.version, tuple(slots))
        self.validate(world)
        return world

    def from_flat(self, values: np.ndarray) -> WorldCode:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        expected_width = len(self.field_order)
        if vector.size != expected_width:
            raise ValueError(f"expected {expected_width} values, got {vector.size}")
        slots: list[OperatorSlot] = []
        cursor = 0
        for spec in self.specs:
            active = bool(vector[cursor] >= 0.5)
            cursor += 1
            normalized = vector[cursor : cursor + len(spec.parameters)]
            cursor += len(spec.parameters)
            if active:
                if np.any((normalized < 0.0) | (normalized > 1.0)):
                    raise ValueError(f"normalized parameters for {spec.name} must lie in [0, 1]")
                parameters = tuple(
                    round(parameter.denormalize(float(value)), 12)
                    for parameter, value in zip(spec.parameters, normalized)
                )
            else:
                parameters = tuple(0.0 for _ in spec.parameters)
            slots.append(OperatorSlot(spec.name, active, parameters))
        world = WorldCode(self.version, tuple(slots))
        self.validate(world)
        return world

    def validate(self, world: WorldCode) -> None:
        if world.schema_version != self.version:
            raise ValueError(
                f"world schema {world.schema_version!r} does not match {self.version!r}"
            )
        if tuple(slot.name for slot in world.slots) != self.slot_names:
            raise ValueError("world slots do not match schema order")
        for spec, slot in zip(self.specs, world.slots):
            if len(slot.parameters) != len(spec.parameters):
                raise ValueError(f"wrong parameter width for {slot.name}")
            if not slot.active and any(value != 0.0 for value in slot.parameters):
                raise ValueError(f"inactive {slot.name} must use canonical zero parameters")
            if slot.active:
                for parameter, value in zip(spec.parameters, slot.parameters):
                    parameter.validate(value, active=True)

