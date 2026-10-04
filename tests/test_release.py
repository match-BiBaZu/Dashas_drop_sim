"""Release ranges are symmetric, seeded and independent of other draws."""

from pathlib import Path

import numpy as np
import pytest

from dashas_drop_sim.config import RunConfig
from dashas_drop_sim.release import sample_release


def test_release_sampling_ranges_center_and_reproducibility():
    config = RunConfig(Path("not-needed.stl"), drop_height_mm=100, drop_height_spread_mm=20,
                       lateral_mm=-15, lateral_spread_mm=5)
    config.validate_parameters()
    samples = np.array([sample_release(config, seed) for seed in range(2048)])
    assert np.all((samples[:, 0] >= 80) & (samples[:, 0] <= 120))
    assert np.all((samples[:, 1] >= -20) & (samples[:, 1] <= -10))
    assert abs(samples[:, 0].mean() - 100) < 1
    assert abs(samples[:, 1].mean() + 15) < 0.25
    assert samples[:, 0].min() < 81 and samples[:, 0].max() > 119
    assert sample_release(config, 123) == sample_release(config, 123)
    before = sample_release(config, 123)
    config.lateral_spread_mm = 0
    assert sample_release(config, 123) == (before[0], -15)
    config.drop_height_spread_mm = 0
    assert sample_release(config, 123) == (100, -15)


@pytest.mark.parametrize("settings", [
    {"drop_height_mm": 190, "drop_height_spread_mm": 20},
    {"lateral_mm": 98, "lateral_spread_mm": 3},
    {"drop_height_spread_mm": -1},
])
def test_invalid_release_range_is_rejected(settings):
    config = RunConfig(Path("not-needed.stl"), **settings)
    with pytest.raises(ValueError):
        config.validate_parameters()
