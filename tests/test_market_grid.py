"""Tests for the market temporal backbone and system load profile.

Covers :class:`~sim_stochastic_pv.market.grid.TimeGrid` (calendar metadata)
and :class:`~sim_stochastic_pv.market.grid.LoadProfile` (multiplicative
system-load model). These are deterministic given a seed, so the assertions
target exact shapes, value ranges, and reproducibility.
"""

from __future__ import annotations

import numpy as np

from sim_stochastic_pv.market.config import QUARTERS_PER_YEAR
from sim_stochastic_pv.market.grid import TimeGrid, LoadProfile


def test_timegrid_shapes_and_ranges() -> None:
    """TimeGrid exposes full-year arrays with calendar values in range."""
    tg = TimeGrid()
    assert tg.n == QUARTERS_PER_YEAR == 35040
    assert tg.month.shape == (tg.n,)
    assert tg.hour.shape == (tg.n,)
    assert set(np.unique(tg.month)).issubset(set(range(1, 13)))
    assert tg.hour.min() == 0 and tg.hour.max() == 23
    assert tg.day_of_week.min() == 0 and tg.day_of_week.max() == 6
    assert tg.is_holiday.dtype == bool and not tg.is_holiday.any()


def test_timegrid_holiday_calendar() -> None:
    """Marking holidays flags exactly the quarter-hours of those days."""
    tg = TimeGrid()
    tg.set_holiday_calendar([0, 5])  # day-of-year 0 and 5
    assert tg.is_holiday.sum() == 2 * 96  # 96 quarter-hours per day


def test_loadprofile_shape_and_determinism() -> None:
    """LoadProfile.generate is reproducible under a fixed seed."""
    tg = TimeGrid()
    lp = LoadProfile(tg, p_peak_pu=1.0)
    a = lp.generate(np.random.default_rng(7), noise_sigma=0.04)
    b = lp.generate(np.random.default_rng(7), noise_sigma=0.04)
    assert a.shape == (tg.n,)
    assert np.allclose(a, b)
    assert (a > 0).all()


def test_loadprofile_noise_increases_dispersion() -> None:
    """Stochastic noise widens dispersion vs the deterministic profile."""
    tg = TimeGrid()
    lp = LoadProfile(tg)
    base = lp.generate(noise_sigma=0.0)
    noisy = lp.generate(np.random.default_rng(1), noise_sigma=0.10)
    assert noisy.std() > base.std()
    assert abs(noisy.mean() - base.mean()) / base.mean() < 0.05
