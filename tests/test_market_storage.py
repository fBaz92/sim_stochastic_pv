"""Tests for the grid-scale battery physical model (:class:`StorageUnit`).

Cover the cheap, pure parts of the storage model (efficiency split,
self-discharge compounding, SOC-gated synthetic inertia, constructor
validation). The full stateful storage *dispatch* loop is exercised by the
higher-level simulation smoke path, not here, because it iterates over all
35 040 quarter-hours and would exceed the per-test time budget.
"""

from __future__ import annotations

import pytest

from sim_stochastic_pv.market.config import P_PEAK_GW, QUARTERS_PER_DAY
from sim_stochastic_pv.market.storage import StorageUnit, build_storage_units


def _unit(**kw) -> StorageUnit:
    params = dict(name="bess", energy_capacity_gwh=4.0, power_capacity_gw=2.0)
    params.update(kw)
    return StorageUnit(**params)


def test_efficiency_split_is_symmetric_sqrt() -> None:
    """Round-trip efficiency splits symmetrically into sqrt on each leg."""
    u = _unit(efficiency_roundtrip=0.81)
    assert u.eta_charge == pytest.approx(0.9)
    assert u.eta_discharge == pytest.approx(0.9)
    assert u.eta_charge * u.eta_discharge == pytest.approx(0.81)


def test_per_unit_conversions() -> None:
    """Power/energy convert to per-unit of the system base correctly."""
    u = _unit(power_capacity_gw=6.0, energy_capacity_gwh=12.0)
    assert u.power_capacity_pu == pytest.approx(6.0 / P_PEAK_GW)
    assert u.energy_capacity_pu_h == pytest.approx(12.0 / P_PEAK_GW)


def test_self_discharge_compounds_to_daily_rate() -> None:
    """The per-qh self-discharge compounds over a day to the daily rate."""
    u = _unit(self_discharge_per_day=0.02)
    retained = (1.0 - u.self_discharge_per_qh) ** QUARTERS_PER_DAY
    assert retained == pytest.approx(1.0 - 0.02)


def test_inertia_contribution_gated_by_soc() -> None:
    """Synthetic inertia is offered inside the band and zeroed at the edges."""
    u = _unit(h_synthetic=4.0, soc_min_frac=0.1, soc_max_frac=0.9,
              inertia_soc_margin=0.02)
    wh, w = u.inertia_contribution(0.5)
    assert wh == pytest.approx(4.0 * u.power_capacity_pu)
    assert w == pytest.approx(u.power_capacity_pu)
    assert u.inertia_contribution(0.1) == (0.0, 0.0)  # at lower bound
    assert _unit(h_synthetic=0.0).inertia_contribution(0.5) == (0.0, 0.0)


def test_invalid_parameters_raise() -> None:
    """The constructor rejects out-of-range physical parameters."""
    with pytest.raises(ValueError):
        _unit(efficiency_roundtrip=1.5)
    with pytest.raises(ValueError):
        _unit(soc_min_frac=0.9, soc_max_frac=0.1)
    with pytest.raises(ValueError):
        _unit(initial_soc_frac=0.99)  # outside [soc_min, soc_max]


def test_build_storage_units_skips_degenerate() -> None:
    """The builder drops entries with non-positive power or energy."""
    cfg = {
        "ok": {"energy_capacity_gwh": 4.0, "power_capacity_gw": 2.0},
        "no_energy": {"energy_capacity_gwh": 0.0, "power_capacity_gw": 2.0},
        "no_power": {"energy_capacity_gwh": 4.0, "power_capacity_gw": 0.0},
    }
    units = build_storage_units(cfg)
    assert [u.name for u in units] == ["ok"]
    assert build_storage_units(None) == []
