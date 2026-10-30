"""Typed configuration shared across environment modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.05
    steps: int = 40
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.steps < 1:
            raise ValueError("steps must be positive")


@dataclass(frozen=True)
class RenderConfig:
    width: int = 64
    height: int = 64
    ball_radius: int = 2

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 8:
            raise ValueError("render dimensions must be at least 8 pixels")
        if self.ball_radius < 1:
            raise ValueError("ball_radius must be positive")
