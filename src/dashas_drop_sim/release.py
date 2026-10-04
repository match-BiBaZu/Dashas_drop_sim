"""Independent seeded streams for the height and lateral release distribution."""

import numpy as np

from .config import RunConfig


def sample_release(config: RunConfig, trial_seed: int) -> tuple[float, float]:
    values = []
    for index, (center, spread) in enumerate((
        (config.drop_height_mm, config.drop_height_spread_mm),
        (config.lateral_mm, config.lateral_spread_mm),
    )):
        if spread == 0:
            values.append(float(center))
        else:
            rng = np.random.default_rng(np.random.SeedSequence([trial_seed, 0xABF411, index]))
            values.append(float(rng.uniform(center - spread, center + spread)))
    return values[0], values[1]
