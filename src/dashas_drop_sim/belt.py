"""Seeded smooth conveyor speed changes, independent of the physics timestep."""

from __future__ import annotations

import math

import numpy as np

from .config import RunConfig


MODEL_VERSION = "cosine_interpolated_uniform_speed_v1"


class BeltSpeedProfile:
    def __init__(self, config: RunConfig, seed: int):
        self.nominal = config.belt_speed_mm_s
        self.amplitude = config.belt_speed_variation_mm_s
        self.interval = config.belt_variation_interval_s
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0xBE17]))
        self.values = [self.nominal]

    def speed_mm_s(self, elapsed_s: float) -> float:
        if self.amplitude == 0:
            return self.nominal
        position = max(0.0, elapsed_s) / self.interval
        index = math.floor(position)
        while len(self.values) <= index + 1:
            self.values.append(float(self.rng.uniform(self.nominal - self.amplitude,
                                                      self.nominal + self.amplitude)))
        blend = (1 - math.cos(math.pi * (position - index))) / 2
        return (1 - blend) * self.values[index] + blend * self.values[index + 1]

    def control_points(self) -> tuple[dict, ...]:
        if self.amplitude == 0:
            return ()
        return tuple({"elapsed_s": index * self.interval, "speed_mm_s": speed}
                     for index, speed in enumerate(self.values))
