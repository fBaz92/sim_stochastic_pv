"""Mix evolution over the investment horizon and the wholesale price surface.

This module turns the single-year market engine
(:func:`sim_stochastic_pv.market.simulation.run_monte_carlo`) into a
multi-year, pre-computed **price surface** that the household PV Monte Carlo
can consult cheaply, hour by hour, without ever re-running a dispatch inside
its own paths.

Two concepts live here:

- :class:`MixTrend` — how the installed generation mix (and, optionally, the
  fuel/CO2 mean prices) evolve across the horizon. It answers
  "what does the system look like in year *y*?" via
  :meth:`MixTrend.build_mix_for_year`. This is where a user expresses things
  like "renewables grow 8%/year" or "nuclear is absent until 2035, then
  5 GW come online".

- :class:`PriceSurface` — a frozen lookup table of wholesale prices indexed
  by ``(trajectory, year, month, hour)``, in EUR/kWh. It is produced once by
  :func:`build_price_surface` (an expensive operation meant to be cached on a
  saved market profile) and then read in O(1) by the PV economics.

Why a *bank of K trajectories* rather than a single mean surface:
    A single mean price per (year, month, hour) would make the wholesale
    price deterministic given the calendar, so all the uncertainty in the PV
    investment would come from the household's own weather/load — the market
    would contribute zero risk. That hides exactly the thing the project
    cares about: a gas-price shock translating into a wide profit band. So
    the surface keeps ``K`` independent market trajectories; each PV path
    draws one trajectory, and across paths the spread of market outcomes
    feeds the p05-p95 profit band.

Known limitation (documented in ROADMAP):
    Years within a single trajectory are simulated independently — the
    engine has no year-to-year persistence of fuel-price regimes. The
    *level* trend across years (mix evolution, fuel drift) is captured; a
    multi-year shock that persists (e.g. a 2021-2023-style crisis lasting
    several consecutive years inside one trajectory) is not. The household
    weather and the system weather are likewise independent across the two
    Monte Carlos (the surface is built with the system's own weather draws).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from sim_stochastic_pv.market.grid import TimeGrid
from sim_stochastic_pv.market.simulation import run_monte_carlo

MONTHS_PER_YEAR: int = 12
"""Number of calendar months per year (price-surface month axis length)."""

HOURS_PER_DAY: int = 24
"""Number of hours per day (price-surface hour axis length)."""

EUR_PER_MWH_TO_EUR_PER_KWH: float = 1.0e-3
"""Conversion factor: the engine clears in EUR/MWh, the PV economics use EUR/kWh."""


@dataclass(frozen=True)
class TechCapacityTrend:
    """Capacity trajectory of one technology across the investment horizon.

    The installed capacity of a single technology (e.g. ``"solar"``) is
    modelled as a compound annual growth from a base value, with an optional
    one-off step at a given year that re-bases the trajectory. The step lets
    a technology *appear* mid-horizon (the canonical example: nuclear sitting
    at 0 GW until 2035, then 5 GW coming online and growing from there).

    Formula::

        if step_year is None or year_index < step_year:
            capacity(y) = base_capacity_gw * (1 + annual_growth_pct/100)^y
        else:  # from the step year onward, re-based to step_capacity_gw
            capacity(y) = step_capacity_gw
                          * (1 + annual_growth_pct/100)^(y - step_year)

    Attributes:
        base_capacity_gw: Installed capacity in GW at year 0 (>= 0).
        annual_growth_pct: Compound year-on-year growth in percent (e.g.
            ``8.0`` for +8%/year). May be negative for a phase-out. Defaults
            to ``0.0`` (flat).
        step_year: Optional year index (0-based, relative to the horizon
            start) at which the trajectory re-bases to ``step_capacity_gw``.
            ``None`` (default) means no step.
        step_capacity_gw: Capacity in GW from ``step_year`` onward. Required
            when ``step_year`` is set, ignored otherwise.
    """

    base_capacity_gw: float
    annual_growth_pct: float = 0.0
    step_year: int | None = None
    step_capacity_gw: float | None = None

    def __post_init__(self) -> None:
        """Validate the trend parameters.

        Raises:
            ValueError: If a base/step capacity is negative, or a step year
                is given without a step capacity (or vice versa).
        """
        if self.base_capacity_gw < 0.0:
            raise ValueError(
                f"base_capacity_gw must be >= 0; got {self.base_capacity_gw!r}")
        if (self.step_year is None) != (self.step_capacity_gw is None):
            raise ValueError(
                "step_year and step_capacity_gw must be set together; got "
                f"step_year={self.step_year!r}, "
                f"step_capacity_gw={self.step_capacity_gw!r}")
        if self.step_year is not None and self.step_year < 0:
            raise ValueError(f"step_year must be >= 0; got {self.step_year!r}")
        if self.step_capacity_gw is not None and self.step_capacity_gw < 0.0:
            raise ValueError(
                f"step_capacity_gw must be >= 0; got {self.step_capacity_gw!r}")

    def capacity_for_year(self, year_index: int) -> float:
        """Return the installed capacity in GW for a given horizon year.

        Args:
            year_index: Year index, 0-based from the horizon start (>= 0).

        Returns:
            float: Installed capacity in GW (>= 0) for that year.

        Example:
            >>> t = TechCapacityTrend(base_capacity_gw=30.0, annual_growth_pct=10.0)
            >>> round(t.capacity_for_year(0), 1)
            30.0
            >>> round(t.capacity_for_year(2), 1)  # 30 * 1.1^2
            36.3
            >>> nuke = TechCapacityTrend(0.0, step_year=5, step_capacity_gw=4.0)
            >>> nuke.capacity_for_year(4)
            0.0
            >>> nuke.capacity_for_year(5)
            4.0
        """
        growth = 1.0 + self.annual_growth_pct / 100.0
        if self.step_year is None or year_index < self.step_year:
            return self.base_capacity_gw * growth ** year_index
        return self.step_capacity_gw * growth ** (year_index - self.step_year)


@dataclass(frozen=True)
class MixTrend:
    """Evolution of the whole generation mix (and fuel prices) over the horizon.

    Wraps a base mix (in the :data:`~sim_stochastic_pv.market.config.ITALIAN_MIX`
    shape: ``tech -> parameter dict``) with per-technology capacity trends and
    optional annual drifts on the gas and CO2 mean prices. It is the single
    source of truth for "what market does year *y* face?", consumed by
    :func:`build_price_surface`.

    A technology without an entry in :attr:`capacity_trends` keeps its base
    capacity flat across the horizon. A technology whose capacity is 0 in a
    given year is naturally excluded from the dispatch (the generator factory
    skips zero-capacity units), so a nuclear step at year 5 makes nuclear
    simply absent before then.

    Attributes:
        base_mix: Mapping ``tech -> parameter dict`` defining the year-0 mix
            and every technology's technical/economic parameters. The
            per-year capacity overrides only the ``capacity_gw`` field.
        capacity_trends: Mapping ``tech -> TechCapacityTrend``. Technologies
            absent from this mapping stay at their base capacity. A trend may
            reference a technology *not* present in ``base_mix`` only if that
            technology is also fully described in ``base_mix`` (capacity is
            overridden, all other parameters come from ``base_mix``).
        gas_mu_drift_annual: Compound annual drift applied to the gas
            scenario mean price ``mu`` (e.g. ``0.02`` = +2%/year). Defaults
            to ``0.0``.
        co2_mu_drift_annual: Compound annual drift applied to the CO2
            scenario mean price ``mu``. Defaults to ``0.0``.
    """

    base_mix: Mapping[str, Mapping[str, float]]
    capacity_trends: Mapping[str, TechCapacityTrend] = field(default_factory=dict)
    gas_mu_drift_annual: float = 0.0
    co2_mu_drift_annual: float = 0.0

    def __post_init__(self) -> None:
        """Validate that every trended technology exists in the base mix.

        Raises:
            ValueError: If a capacity trend names a technology that is not
                described in ``base_mix`` (its non-capacity parameters would
                be unknown).
        """
        unknown = set(self.capacity_trends) - set(self.base_mix)
        if unknown:
            raise ValueError(
                "capacity_trends reference technologies missing from "
                f"base_mix: {sorted(unknown)}")

    def build_mix_for_year(self, year_index: int) -> dict[str, dict[str, float]]:
        """Build the generation-mix dict for a given horizon year.

        Every technology keeps its base technical/economic parameters; only
        ``capacity_gw`` is replaced by the value its trend predicts for the
        requested year. Technologies without a trend keep their base capacity.

        Args:
            year_index: Year index, 0-based from the horizon start (>= 0).

        Returns:
            dict: A fresh ``tech -> parameter dict`` mapping suitable for
                :func:`~sim_stochastic_pv.market.generators.build_generators`.
                Zero-capacity technologies are kept in the dict (the generator
                factory skips them), so the structure is stable across years.

        Example:
            >>> from sim_stochastic_pv.market.config import ITALIAN_MIX
            >>> trend = MixTrend(
            ...     base_mix=ITALIAN_MIX,
            ...     capacity_trends={
            ...         "solar": TechCapacityTrend(30.0, annual_growth_pct=10.0),
            ...     },
            ... )
            >>> round(trend.build_mix_for_year(1)["solar"]["capacity_gw"], 1)
            33.0
        """
        mix: dict[str, dict[str, float]] = {}
        for tech, params in self.base_mix.items():
            new_params = dict(params)
            trend = self.capacity_trends.get(tech)
            if trend is not None:
                new_params["capacity_gw"] = trend.capacity_for_year(year_index)
            mix[tech] = new_params
        return mix

    def gas_scenario_for_year(
        self,
        base_gas_scenario: Mapping[str, float],
        year_index: int,
    ) -> dict[str, float]:
        """Return the gas price scenario drifted to a given horizon year.

        Only the mean price ``mu`` is drifted; volatility ``sigma`` and
        mean-reversion ``theta`` are kept constant.

        Args:
            base_gas_scenario: Year-0 gas scenario (keys ``mu``, ``sigma``,
                ``theta``).
            year_index: Year index, 0-based (>= 0).

        Returns:
            dict: A fresh gas scenario with ``mu`` scaled by
                ``(1 + gas_mu_drift_annual)^year_index``.
        """
        out = dict(base_gas_scenario)
        out["mu"] = base_gas_scenario["mu"] * (
            1.0 + self.gas_mu_drift_annual) ** year_index
        return out

    def co2_scenario_for_year(
        self,
        base_co2_scenario: Mapping[str, float],
        year_index: int,
    ) -> dict[str, float]:
        """Return the CO2 price scenario drifted to a given horizon year.

        Args:
            base_co2_scenario: Year-0 CO2 scenario (keys ``mu``, ``sigma``,
                ``theta``).
            year_index: Year index, 0-based (>= 0).

        Returns:
            dict: A fresh CO2 scenario with ``mu`` scaled by
                ``(1 + co2_mu_drift_annual)^year_index``.
        """
        out = dict(base_co2_scenario)
        out["mu"] = base_co2_scenario["mu"] * (
            1.0 + self.co2_mu_drift_annual) ** year_index
        return out


@dataclass(frozen=True)
class PriceSurface:
    """Pre-computed wholesale price surface, indexed (trajectory, year, month, hour).

    A frozen lookup table produced by :func:`build_price_surface`. It holds
    ``K`` independent market trajectories, each spanning the full horizon, so
    that a household PV path can draw one trajectory and read an hourly
    wholesale price in O(1) without running any dispatch. Prices are in
    **EUR/kWh** (the engine clears in EUR/MWh; the conversion is applied at
    build time).

    Attributes:
        price_eur_per_kwh: Array of shape ``(n_trajectories, n_years, 12, 24)``
            with the mean wholesale price for each (trajectory, year,
            calendar-month, hour-of-day), in EUR/kWh. The month axis is
            0-based (index 0 = January); the hour axis is 0-based
            (index 0 = 00:00-01:00).
        n_trajectories: Number of market trajectories ``K`` (axis 0 length).
        n_years: Horizon length in years (axis 1 length).

    Example:
        >>> # surface.price_at(trajectory=3, year=0, month=7, hour=13)
        >>> # -> EUR/kWh for a July 13:00 in year 0 along trajectory 3
    """

    price_eur_per_kwh: np.ndarray
    n_trajectories: int
    n_years: int

    def price_at(self, trajectory: int, year: int, month: int, hour: int) -> float:
        """Look up the wholesale price for one (trajectory, year, month, hour).

        Args:
            trajectory: Trajectory index in ``[0, n_trajectories)``.
            year: Year index in ``[0, n_years)``.
            month: Calendar month, **1-based** (1 = January, 12 = December).
            hour: Hour of day in ``[0, 24)``.

        Returns:
            float: Wholesale price in EUR/kWh.
        """
        return float(self.price_eur_per_kwh[trajectory, year, month - 1, hour])

    def trajectory_grid(self, trajectory: int) -> np.ndarray:
        """Return the full (n_years, 12, 24) price grid for one trajectory.

        Intended for the PV Monte Carlo: each path picks a trajectory once
        and then indexes this grid by ``[year, month-1, hour]`` hour by hour.

        Args:
            trajectory: Trajectory index in ``[0, n_trajectories)``.

        Returns:
            np.ndarray: View of shape ``(n_years, 12, 24)`` in EUR/kWh.
        """
        return self.price_eur_per_kwh[trajectory]

    def mean_grid(self) -> np.ndarray:
        """Return the across-trajectory mean price grid.

        Returns:
            np.ndarray: Shape ``(n_years, 12, 24)`` in EUR/kWh, the mean over
                the trajectory axis. Useful for the heat-map view.
        """
        return self.price_eur_per_kwh.mean(axis=0)

    def percentile_grid(self, q: float) -> np.ndarray:
        """Return an across-trajectory percentile price grid.

        Args:
            q: Percentile in ``[0, 100]`` (e.g. ``5`` for p05, ``95`` for p95).

        Returns:
            np.ndarray: Shape ``(n_years, 12, 24)`` in EUR/kWh.
        """
        return np.percentile(self.price_eur_per_kwh, q, axis=0)


def _month_hour_index(time_grid: TimeGrid) -> np.ndarray:
    """Map each quarter-hour to a flat (month, hour) bin index in ``[0, 288)``.

    Args:
        time_grid: The market temporal backbone.

    Returns:
        np.ndarray: Integer array of shape ``(time_grid.n,)`` with values
            ``(month - 1) * 24 + hour`` for fast binning with ``np.bincount``.
    """
    return (time_grid.month - 1) * HOURS_PER_DAY + time_grid.hour


def _grid_from_marginal_price(
    marginal_price: np.ndarray,
    mh_index: np.ndarray,
) -> np.ndarray:
    """Average a quarter-hourly price series into a (12, 24) month x hour grid.

    Args:
        marginal_price: System marginal price for one dispatched year, shape
            ``(35040,)`` in EUR/MWh.
        mh_index: Precomputed bin index from :func:`_month_hour_index`, shape
            ``(35040,)``.

    Returns:
        np.ndarray: Shape ``(12, 24)`` of mean prices per (month, hour) in
            EUR/MWh.
    """
    n_bins = MONTHS_PER_YEAR * HOURS_PER_DAY
    sums = np.bincount(mh_index, weights=marginal_price, minlength=n_bins)
    counts = np.bincount(mh_index, minlength=n_bins)
    return (sums / counts).reshape(MONTHS_PER_YEAR, HOURS_PER_DAY)


def _interpolate_years(
    rep_years: Sequence[int],
    rep_grids: np.ndarray,
    n_years: int,
) -> np.ndarray:
    """Linearly interpolate per-year price grids over the full horizon.

    The surface is computed only at a set of representative years (to bound
    cost) and the intermediate years are filled by linear interpolation
    (with flat extrapolation outside the representative range).

    Args:
        rep_years: Representative year indices actually simulated, ascending.
        rep_grids: Price grids at those years, shape
            ``(len(rep_years), K, 12, 24)``.
        n_years: Full horizon length to expand to.

    Returns:
        np.ndarray: Shape ``(K, n_years, 12, 24)`` with every year filled.
    """
    rep = np.asarray(rep_years, dtype=float)
    all_years = np.arange(n_years, dtype=float)
    if rep.size == 1:
        # A single representative year: every horizon year takes that grid.
        expanded = np.repeat(rep_grids, n_years, axis=0)  # (n_years, K, 12, 24)
    else:
        # Interpolate each (K, month, hour) cell independently along the year
        # axis. Flatten the trailing axes so np.interp runs once per cell.
        flat = rep_grids.reshape(rep.size, -1)            # (R, K*12*24)
        out = np.empty((n_years, flat.shape[1]))
        for col in range(flat.shape[1]):
            out[:, col] = np.interp(all_years, rep, flat[:, col])
        expanded = out.reshape((n_years,) + rep_grids.shape[1:])
    # Move the year axis after the trajectory axis: (K, n_years, 12, 24).
    return np.moveaxis(expanded, 0, 1)


def build_price_surface(
    mix_trend: MixTrend,
    base_gas_scenario: Mapping[str, float],
    *,
    n_years: int,
    n_trajectories: int = 30,
    coal_scenario: Mapping[str, float] | None = None,
    co2_scenario: Mapping[str, float] | None = None,
    representative_years: Sequence[int] | None = None,
    interconnections_cfg: Mapping[str, Mapping] | None = None,
    price_areas_cfg: Mapping[str, Mapping] | None = None,
    price_area_correlations: Mapping | None = None,
    storage_cfg: Mapping[str, Mapping] | None = None,
    seed: int = 42,
) -> PriceSurface:
    """Build the multi-year wholesale price surface (a bank of K trajectories).

    For each representative year the market Monte Carlo is run with
    ``n_trajectories`` independent draws; the per-draw system marginal price
    is reduced to a ``(12, 24)`` month-by-hour grid. Intermediate years are
    filled by linear interpolation. The result is a frozen
    :class:`PriceSurface` of shape ``(K, n_years, 12, 24)`` in EUR/kWh,
    intended to be computed once and cached on a saved market profile, then
    read in O(1) by the PV economics.

    Cost scales as ``len(representative_years) * n_trajectories`` dispatches,
    not ``n_pv_paths * n_years`` — the whole point of pre-computing the
    surface instead of running the market inside each PV path.

    Args:
        mix_trend: Evolution of the generation mix and fuel-price drifts.
        base_gas_scenario: Year-0 gas O-U scenario (keys ``mu``, ``sigma``,
            ``theta``), drifted per year via the trend.
        n_years: Horizon length in years (> 0).
        n_trajectories: Number of market trajectories ``K`` (> 0). Defaults
            to 30. More trajectories give smoother risk bands at linear cost.
        coal_scenario: Optional year-0 coal O-U scenario; ``None`` lets the
            engine use its default.
        co2_scenario: Optional year-0 CO2 O-U scenario, drifted per year via
            the trend; ``None`` lets the engine use its default (no CO2 drift
            is applied in that case).
        representative_years: Ascending year indices to actually simulate;
            the rest are interpolated. ``None`` (default) simulates every year
            ``0..n_years-1`` (most accurate, most expensive).
        interconnections_cfg: Optional cross-border link configuration
            (enables the "full" market with imports/exports).
        price_areas_cfg: Foreign price-area parameters; required when
            ``interconnections_cfg`` is supplied.
        price_area_correlations: Optional pairwise foreign-price correlations.
        storage_cfg: Optional grid-scale storage configuration.
        seed: Base random seed. Each representative year uses a distinct
            derived seed so trajectories are independent across years.

    Returns:
        PriceSurface: The frozen price lookup table in EUR/kWh.

    Raises:
        ValueError: If ``n_years <= 0``, ``n_trajectories <= 0``, or
            ``representative_years`` is empty or out of range.

    Example:
        >>> from sim_stochastic_pv.market.config import ITALIAN_MIX, GAS_SCENARIOS
        >>> trend = MixTrend(base_mix=ITALIAN_MIX)
        >>> surf = build_price_surface(
        ...     trend, GAS_SCENARIOS["base"],
        ...     n_years=2, n_trajectories=2, seed=0)
        >>> surf.price_eur_per_kwh.shape
        (2, 2, 12, 24)
    """
    if n_years <= 0:
        raise ValueError(f"n_years must be > 0; got {n_years!r}")
    if n_trajectories <= 0:
        raise ValueError(f"n_trajectories must be > 0; got {n_trajectories!r}")

    if representative_years is None:
        rep_years = list(range(n_years))
    else:
        rep_years = sorted(set(int(y) for y in representative_years))
        if not rep_years:
            raise ValueError("representative_years must be non-empty")
        if rep_years[0] < 0 or rep_years[-1] >= n_years:
            raise ValueError(
                "representative_years must lie within [0, n_years); got "
                f"{rep_years} for n_years={n_years}")

    time_grid = TimeGrid()
    mh_index = _month_hour_index(time_grid)

    rep_grids = np.empty((len(rep_years), n_trajectories,
                          MONTHS_PER_YEAR, HOURS_PER_DAY))

    for slot, year in enumerate(rep_years):
        mix_year = mix_trend.build_mix_for_year(year)
        gas_year = mix_trend.gas_scenario_for_year(base_gas_scenario, year)
        co2_year = (
            mix_trend.co2_scenario_for_year(co2_scenario, year)
            if co2_scenario is not None else None)

        captured: list[np.ndarray] = []

        def _capture(run_index: int, result) -> None:
            captured.append(_grid_from_marginal_price(
                result.marginal_price, mh_index))

        # A distinct seed per representative year keeps the K trajectories of
        # different years statistically independent (the documented "no
        # cross-year persistence" simplification).
        run_monte_carlo(
            mix_config=mix_year,
            gas_scenario=gas_year,
            coal_scenario=coal_scenario,
            co2_scenario=co2_year,
            n_runs=n_trajectories,
            seed=seed + year * 100_003,
            interconnections_cfg=interconnections_cfg,
            price_areas_cfg=price_areas_cfg,
            price_area_correlations=price_area_correlations,
            storage_cfg=storage_cfg,
            dispatch_callback=_capture,
        )
        rep_grids[slot] = np.stack(captured, axis=0)

    rep_grids *= EUR_PER_MWH_TO_EUR_PER_KWH
    surface = _interpolate_years(rep_years, rep_grids, n_years)
    return PriceSurface(
        price_eur_per_kwh=surface,
        n_trajectories=n_trajectories,
        n_years=n_years,
    )
