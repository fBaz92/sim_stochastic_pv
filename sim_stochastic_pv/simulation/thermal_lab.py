"""
Phase 19 — Thermal laboratory comparison engine.

Lets the user reason about the building envelope and heat pump *before* the
full economic scenario: fix a climate profile (Phase 15), a heat pump, the
comfort setpoints and an occupancy pattern, then run a small Monte Carlo and
compare several **house configurations** (insulation presets ``poor`` /
``standard`` / ``good`` or a custom ``UA``) side-by-side.

The engine is a thin orchestration layer on top of the building blocks that
already exist:

- :class:`sim_stochastic_pv.simulation.thermal.ThermalModel` — the stochastic
  ambient-temperature generator (per path).
- :class:`sim_stochastic_pv.simulation.thermal_load.HvacController` — the RC
  house + heat-pump model that turns ambient temperature into an hourly
  electric HVAC draw and (in dynamic mode) an indoor-temperature trajectory.

Outputs are organised for the (future) webapp charts: per-variant scalar KPIs
(annual HVAC energy + cost + comfort breaches + peak power + worst-case indoor
temperature) plus a calendar-aligned **typical-year** daily series (outdoor
temperature, HVAC energy, representative indoor temperature) and the worst
heating / cooling days.

The economic comparison uses a *scalar* electricity price (€/kWh) in this
iteration — coupling the stochastic GBM price model (Phase 2) is left to the
UI slice where the full scenario price profile is already in scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .prices import PriceModel
from .thermal import ThermalModel, month_of_year
from .thermal_load import (
    HeatPumpConfig,
    HouseThermalConfig,
    HvacController,
    SetpointConfig,
    ThermalLoadConfig,
)


#: Days per simulated year. Matches :class:`ThermalModel`'s 365-day calendar
#: (``day_of_year = day_index % 365``) so the typical-year folding below is
#: exact and a daily index maps cleanly to January 1 … December 31.
DAYS_PER_YEAR: int = 365

#: Default residential electricity price (€/kWh) used for the cost KPI when
#: the caller does not supply one. Rough Italian residential average 2024.
DEFAULT_ELECTRICITY_PRICE_EUR_PER_KWH: float = 0.25

#: Offset added to the per-path seed for the *price* RNG stream so the price
#: trajectory is statistically independent from the temperature path (which
#: uses ``seed + path_index``). Large constant ⇒ no overlap in practice.
_PRICE_SEED_OFFSET: int = 2_000_003


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HouseVariant:
    """
    One labelled house configuration to compare in the thermal lab.

    Attributes:
        label: Human-readable name shown in the UI legend / table
            (e.g. ``"Casa anni '70"``, ``"NZEB"``). Used as the result key.
        house: The :class:`HouseThermalConfig` (envelope + capacitance +
            internal gains) for this variant. All other knobs (heat pump,
            setpoints, occupancy) are shared across the compared variants —
            only the envelope changes, which is the whole point of the
            comparison.
    """

    label: str
    house: HouseThermalConfig


@dataclass(frozen=True)
class ThermalLabConfig:
    """
    Full specification for a thermal-lab comparison run.

    Attributes:
        house_variants: Tuple of :class:`HouseVariant` to compare (at least
            one). Coerced to a tuple in ``__post_init__`` so the config stays
            hashable.
        heat_pump: Shared :class:`HeatPumpConfig` (COP + max power).
        setpoint: Shared :class:`SetpointConfig` (single setpoints or a
            time-of-day schedule).
        dynamic: When ``True`` the RC integrator (Phase 18) runs, producing
            the real indoor-temperature trajectory and exposing under-sizing
            as the house drifting off setpoint. When ``False`` the
            steady-state path runs (indoor temperature pinned to the
            setpoint by assumption).
        home_hours_of_day: Optional subset of ``range(24)`` marking the hours
            the user is at home (applied to every day). ``None`` (default)
            means always at home — the conservative choice for HVAC sizing.
        electricity_price_eur_per_kwh: Flat price used for the annual-cost
            KPI (€/kWh). Defaults to
            :data:`DEFAULT_ELECTRICITY_PRICE_EUR_PER_KWH`.

    Raises:
        ValueError: If ``house_variants`` is empty, if
            ``home_hours_of_day`` contains values outside ``range(24)``, or
            if ``electricity_price_eur_per_kwh`` is negative.
    """

    house_variants: tuple[HouseVariant, ...]
    heat_pump: HeatPumpConfig
    setpoint: SetpointConfig
    dynamic: bool = False
    home_hours_of_day: Optional[tuple[int, ...]] = None
    electricity_price_eur_per_kwh: float = DEFAULT_ELECTRICITY_PRICE_EUR_PER_KWH

    def __post_init__(self) -> None:
        coerced = tuple(self.house_variants)
        if not coerced:
            raise ValueError("house_variants must contain at least one variant")
        object.__setattr__(self, "house_variants", coerced)
        if self.home_hours_of_day is not None:
            hours = tuple(int(h) for h in self.home_hours_of_day)
            if any(h < 0 or h > 23 for h in hours):
                raise ValueError(
                    "home_hours_of_day must be a subset of range(24), "
                    f"got {sorted(set(hours))}"
                )
            object.__setattr__(self, "home_hours_of_day", hours)
        if self.electricity_price_eur_per_kwh < 0:
            raise ValueError(
                "electricity_price_eur_per_kwh must be >= 0, "
                f"got {self.electricity_price_eur_per_kwh}"
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ThermalVariantResult:
    """
    Aggregated comparison result for a single house variant.

    Scalar KPIs are aggregated across the Monte Carlo paths; the daily series
    are folded onto a single calendar-aligned **typical year** (mean across
    paths and years) for charting.

    Attributes:
        label: The variant's label (echoes :attr:`HouseVariant.label`).
        ua_kw_per_c: Derived envelope U·A in kW/°C (the headline insulation
            number).
        hvac_kwh_annual_mean: Mean annual HVAC electric energy (kWh/yr).
        hvac_kwh_annual_p05: 5th percentile across paths (kWh/yr).
        hvac_kwh_annual_p95: 95th percentile across paths (kWh/yr).
        heating_kwh_annual_mean: Mean annual electric energy spent in
            *heating* mode (kWh/yr). The hour is classed as heating when the
            outdoor temperature is below the effective heating setpoint —
            exact in steady-state mode, an approximation in the dynamic RC
            mode (where the controller decides the mode on the free-running
            temperature, not directly on ``T_out``).
        cooling_kwh_annual_mean: Mean annual electric energy spent in
            *cooling* mode (kWh/yr). Same caveat as
            ``heating_kwh_annual_mean``. In steady-state,
            ``heating + cooling == hvac_kwh_annual``.
        annual_cost_eur_mean: Mean annual HVAC cost
            (= ``hvac_kwh_annual × price``), €/yr.
        annual_cost_eur_p05: 5th percentile annual cost (€/yr).
        annual_cost_eur_p95: 95th percentile annual cost (€/yr).
        comfort_breach_hours_per_year_mean: Mean yearly hours the heat pump
            was capped and could not hold the setpoint.
        p_elec_hvac_peak_kw_mean: Mean peak instantaneous HVAC draw (kW).
        t_in_min_c: Worst-case (coldest) indoor temperature across paths
            (°C). Equals the heating setpoint in steady-state mode.
        t_in_max_c: Worst-case (hottest) indoor temperature across paths
            (°C). Equals the cooling setpoint in steady-state mode.
        daily_hvac_kwh: Typical-year daily HVAC energy (kWh/day), shape
            ``(365,)``, index 0 = January 1.
        daily_indoor_min_c: Typical-year daily minimum indoor temperature
            from a representative path (°C), shape ``(365,)``. ``None`` in
            steady-state mode (no trajectory).
        daily_indoor_max_c: Typical-year daily maximum indoor temperature
            from a representative path (°C), shape ``(365,)``. ``None`` in
            steady-state mode.
        worst_heating_day_index: Day-of-year (0-based) with the largest HVAC
            energy among heating-dominated days, or ``None`` if no heating
            day occurred.
        worst_cooling_day_index: Day-of-year (0-based) with the largest HVAC
            energy among cooling-dominated days, or ``None`` if none.
    """

    label: str
    ua_kw_per_c: float
    hvac_kwh_annual_mean: float
    hvac_kwh_annual_p05: float
    hvac_kwh_annual_p95: float
    heating_kwh_annual_mean: float
    cooling_kwh_annual_mean: float
    annual_cost_eur_mean: float
    annual_cost_eur_p05: float
    annual_cost_eur_p95: float
    comfort_breach_hours_per_year_mean: float
    p_elec_hvac_peak_kw_mean: float
    t_in_min_c: float
    t_in_max_c: float
    daily_hvac_kwh: np.ndarray
    daily_indoor_min_c: Optional[np.ndarray]
    daily_indoor_max_c: Optional[np.ndarray]
    worst_heating_day_index: Optional[int]
    worst_cooling_day_index: Optional[int]


@dataclass
class ThermalLabResult:
    """
    Top-level result of :func:`compare_house_variants`.

    Attributes:
        days: Day-of-year indices ``0..364`` (x-axis of the daily charts).
        daily_outdoor_mean_c: Typical-year daily mean outdoor temperature
            (°C), shape ``(365,)``, shared by all variants (same climate).
        variants: One :class:`ThermalVariantResult` per compared house, in
            the same order as :attr:`ThermalLabConfig.house_variants`.
        n_paths: Number of Monte Carlo paths run.
        n_years: Horizon per path (years).
    """

    days: np.ndarray
    daily_outdoor_mean_c: np.ndarray
    variants: list[ThermalVariantResult]
    n_paths: int
    n_years: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_at_home_hourly(
    n_hours: int, home_hours_of_day: Optional[Sequence[int]]
) -> np.ndarray:
    """
    Build the hourly occupancy mask from a per-hour-of-day home schedule.

    Args:
        n_hours: Total number of hours in the horizon.
        home_hours_of_day: Hours of the day (0..23) the user is home, applied
            to every day. ``None`` → always home.

    Returns:
        Boolean array of shape ``(n_hours,)``; entry ``h`` is ``True`` when
        ``h % 24`` is a home hour.
    """
    if home_hours_of_day is None:
        return np.ones(n_hours, dtype=bool)
    home_set = set(int(h) for h in home_hours_of_day)
    hour_of_day = np.arange(n_hours) % 24
    return np.isin(hour_of_day, list(home_set))


def _fold_to_typical_year(daily: np.ndarray, n_years: int) -> np.ndarray:
    """
    Fold an ``(n_years*365,)`` daily series onto a single 365-day mean year.

    Args:
        daily: Daily values, length ``n_years * 365``.
        n_years: Number of whole years in ``daily``.

    Returns:
        Array of shape ``(365,)`` — the per-day mean across the years.
    """
    return daily.reshape(n_years, DAYS_PER_YEAR).mean(axis=0)


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------


def compare_house_variants(
    model: ThermalModel,
    config: ThermalLabConfig,
    n_paths: int = 30,
    n_years: int = 1,
    seed: int = 42,
    price_model: Optional[PriceModel] = None,
) -> ThermalLabResult:
    """
    Run a Monte Carlo thermal comparison across the configured house variants.

    For each of ``n_paths`` independent ambient-temperature paths, every
    house variant is run through its own :class:`HvacController` on the *same*
    temperature path (so differences are purely the envelope, not the
    weather). Per-path KPIs are aggregated; daily series are folded onto a
    calendar-aligned typical year for charting.

    Args:
        model: A calibrated :class:`ThermalModel` (the site climate).
        config: The :class:`ThermalLabConfig` (variants + heat pump +
            setpoints + occupancy + price).
        n_paths: Number of independent MC paths. Default 30. Each path uses
            sub-seed ``seed + path_index`` for bit-stable reproducibility.
        n_years: Horizon per path (years). Default 1.
        seed: Master RNG seed.
        price_model: Optional electricity :class:`PriceModel` (Fase 19-bis).
            When provided, the annual HVAC cost of each path is
            ``Σ_day kWh(day)·price(year, month) / n_years`` with the price
            model reset per path on an RNG stream independent of the
            temperature path (so a stochastic price spreads the cost band
            without correlating to the weather). When ``None`` (default) the
            scalar ``config.electricity_price_eur_per_kwh`` is used, which is
            **byte-identical** to the Fase-19 behaviour.

    Returns:
        A :class:`ThermalLabResult` with one entry per variant plus the
        shared outdoor-temperature typical year.

    Raises:
        ValueError: If ``n_paths`` or ``n_years`` ≤ 0.

    Example:
        ```python
        from sim_stochastic_pv.simulation.thermal_load import (
            HeatPumpConfig, HouseThermalConfig, SetpointConfig,
        )
        cfg = ThermalLabConfig(
            house_variants=(
                HouseVariant("poor", HouseThermalConfig(insulation_preset="poor")),
                HouseVariant("good", HouseThermalConfig(insulation_preset="good")),
            ),
            heat_pump=HeatPumpConfig(),
            setpoint=SetpointConfig(),
        )
        result = compare_house_variants(model, cfg, n_paths=20, seed=0)
        # The well-insulated variant uses strictly less energy:
        assert result.variants[1].hvac_kwh_annual_mean < result.variants[0].hvac_kwh_annual_mean
        ```
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    if n_years <= 0:
        raise ValueError(f"n_years must be > 0, got {n_years}")

    n_days = n_years * DAYS_PER_YEAR
    n_hours = n_days * 24
    at_home_hourly = _build_at_home_hourly(n_hours, config.home_hours_of_day)
    price = config.electricity_price_eur_per_kwh

    # Per-day (year, month) indices, used both for the price model lookup and
    # nowhere else; computed once.
    day_idx = np.arange(n_days)
    year_of_day = day_idx // DAYS_PER_YEAR
    month_of_day = np.asarray(month_of_year(day_idx % DAYS_PER_YEAR), dtype=int)
    # Constant scalar price-by-day (used when no price model is supplied).
    scalar_price_by_day = np.full(n_days, price, dtype=float)

    # Build one controller per variant (reused across paths — the controller
    # is stateless between calls except for the cached indoor series).
    controllers: list[HvacController] = []
    for variant in config.house_variants:
        load_config = ThermalLoadConfig(
            enabled=True,
            house=variant.house,
            heat_pump=config.heat_pump,
            setpoint=config.setpoint,
            dynamic=config.dynamic,
        )
        controllers.append(HvacController(load_config))

    # Effective per-hour setpoints are identical across paths and variants
    # (same occupancy + setpoint config), so derive the heating/cooling
    # classification arrays once.
    t_set_heating, t_set_cooling = controllers[0].setpoint_arrays(at_home_hourly)

    n_variants = len(config.house_variants)
    # Per-variant accumulators.
    annual_kwh = [np.empty(n_paths) for _ in range(n_variants)]
    annual_cost = [np.empty(n_paths) for _ in range(n_variants)]
    heating_kwh = [np.empty(n_paths) for _ in range(n_variants)]
    cooling_kwh = [np.empty(n_paths) for _ in range(n_variants)]
    breach_hpy = [np.empty(n_paths) for _ in range(n_variants)]
    peak_kw = [np.empty(n_paths) for _ in range(n_variants)]
    t_in_min = [np.full(n_paths, np.inf) for _ in range(n_variants)]
    t_in_max = [np.full(n_paths, -np.inf) for _ in range(n_variants)]
    daily_kwh_sum = [np.zeros(n_days) for _ in range(n_variants)]
    # Representative-path daily indoor extremes (path 0) for the trajectory
    # chart; only populated in dynamic mode.
    rep_indoor_min = [None for _ in range(n_variants)]  # type: list[Optional[np.ndarray]]
    rep_indoor_max = [None for _ in range(n_variants)]  # type: list[Optional[np.ndarray]]

    outdoor_daily_sum = np.zeros(n_days)

    for p in range(n_paths):
        rng = np.random.default_rng(seed + p)
        daily_means = model.simulate_daily_means(n_days, rng)
        outdoor_daily_sum += daily_means
        t_ambient_hourly = model.to_hourly(daily_means)  # (n_hours,)

        # Per-path electricity price by day. With a stochastic price model the
        # trajectory varies per path (independent RNG stream); the scalar case
        # is the constant array (so cost == annual_kwh × price exactly).
        if price_model is not None:
            price_rng = np.random.default_rng(seed + _PRICE_SEED_OFFSET + p)
            price_model.reset_for_run(rng=price_rng, n_years=n_years)
            monthly_price = np.array(
                [[price_model.get_price(y, m) for m in range(12)] for y in range(n_years)]
            )
            price_by_day = monthly_price[year_of_day, month_of_day]
        else:
            price_by_day = scalar_price_by_day

        # Heating vs cooling classification for this path's weather.
        heat_mask = t_ambient_hourly < t_set_heating
        cool_mask = t_ambient_hourly > t_set_cooling

        for v, controller in enumerate(controllers):
            p_elec, kpis = controller.compute_hourly_p_elec_kw(
                t_ambient_hourly, at_home_hourly
            )
            annual_kwh[v][p] = kpis.hvac_kwh_annual
            breach_hpy[v][p] = kpis.comfort_breach_hours_per_year
            peak_kw[v][p] = kpis.p_elec_hvac_peak_kw
            t_in_min[v][p] = kpis.t_in_min_c
            t_in_max[v][p] = kpis.t_in_max_c
            daily_kwh = p_elec.reshape(n_days, 24).sum(axis=1)
            daily_kwh_sum[v] += daily_kwh
            annual_cost[v][p] = float((daily_kwh * price_by_day).sum() / n_years)
            heating_kwh[v][p] = float(p_elec[heat_mask].sum() / n_years)
            cooling_kwh[v][p] = float(p_elec[cool_mask].sum() / n_years)

            if p == 0 and controller.last_indoor_temp_c is not None:
                indoor_2d = controller.last_indoor_temp_c.reshape(n_days, 24)
                rep_indoor_min[v] = _fold_to_typical_year(
                    indoor_2d.min(axis=1), n_years
                )
                rep_indoor_max[v] = _fold_to_typical_year(
                    indoor_2d.max(axis=1), n_years
                )

    outdoor_typical = _fold_to_typical_year(outdoor_daily_sum / n_paths, n_years)
    # Midpoint of the dead-band: classifies a day as heating- vs
    # cooling-dominated by its mean outdoor temperature.
    sp = config.setpoint
    deadband_mid = 0.5 * (sp.t_setpoint_heating_c + sp.t_setpoint_cooling_c)
    heating_day_mask = outdoor_typical < deadband_mid
    cooling_day_mask = ~heating_day_mask

    variants: list[ThermalVariantResult] = []
    for v, variant in enumerate(config.house_variants):
        daily_typical = _fold_to_typical_year(daily_kwh_sum[v] / n_paths, n_years)
        worst_heating = _worst_day(daily_typical, heating_day_mask)
        worst_cooling = _worst_day(daily_typical, cooling_day_mask)
        variants.append(
            ThermalVariantResult(
                label=variant.label,
                ua_kw_per_c=variant.house.ua_kw_per_c,
                hvac_kwh_annual_mean=float(np.mean(annual_kwh[v])),
                hvac_kwh_annual_p05=float(np.percentile(annual_kwh[v], 5)),
                hvac_kwh_annual_p95=float(np.percentile(annual_kwh[v], 95)),
                heating_kwh_annual_mean=float(np.mean(heating_kwh[v])),
                cooling_kwh_annual_mean=float(np.mean(cooling_kwh[v])),
                annual_cost_eur_mean=float(np.mean(annual_cost[v])),
                annual_cost_eur_p05=float(np.percentile(annual_cost[v], 5)),
                annual_cost_eur_p95=float(np.percentile(annual_cost[v], 95)),
                comfort_breach_hours_per_year_mean=float(np.mean(breach_hpy[v])),
                p_elec_hvac_peak_kw_mean=float(np.mean(peak_kw[v])),
                t_in_min_c=float(np.min(t_in_min[v])),
                t_in_max_c=float(np.max(t_in_max[v])),
                daily_hvac_kwh=daily_typical,
                daily_indoor_min_c=rep_indoor_min[v],
                daily_indoor_max_c=rep_indoor_max[v],
                worst_heating_day_index=worst_heating,
                worst_cooling_day_index=worst_cooling,
            )
        )

    return ThermalLabResult(
        days=np.arange(DAYS_PER_YEAR),
        daily_outdoor_mean_c=outdoor_typical,
        variants=variants,
        n_paths=n_paths,
        n_years=n_years,
    )


def _worst_day(daily_kwh: np.ndarray, day_mask: np.ndarray) -> Optional[int]:
    """
    Index of the day with the largest HVAC energy among the masked days.

    Args:
        daily_kwh: Typical-year daily HVAC energy, shape ``(365,)``.
        day_mask: Boolean mask selecting candidate days (heating or cooling).

    Returns:
        0-based day index, or ``None`` when the mask selects no day with a
        positive HVAC draw (e.g. no cooling ever needed in a cold climate).
    """
    if not day_mask.any():
        return None
    candidates = np.where(day_mask, daily_kwh, -np.inf)
    best = int(np.argmax(candidates))
    if candidates[best] <= 0.0:
        return None
    return best


# ---------------------------------------------------------------------------
# Single-config hourly timeseries (preview)
# ---------------------------------------------------------------------------


@dataclass
class ThermalTimeseriesResult:
    """
    Hourly timeseries for a single house config over a short horizon.

    Powers the "setpoint vs indoor temperature" preview chart: the outdoor
    temperature drives the heat pump, which (in dynamic mode) produces the
    indoor-temperature trajectory.

    Attributes:
        hours: Hour indices ``0..n_hours-1`` (x-axis).
        t_outdoor_c: Ambient temperature per hour (°C).
        t_indoor_c: Indoor temperature per hour (°C) in dynamic mode, else
            ``None`` (steady-state has no trajectory).
        p_elec_hvac_kw: Electric HVAC draw per hour (kW).
        t_set_heating_c: Effective heating setpoint per hour (°C); ``-inf``
            on away hours without a setback setpoint.
        t_set_cooling_c: Effective cooling setpoint per hour (°C); ``+inf``
            on away hours without a setback setpoint.
    """

    hours: np.ndarray
    t_outdoor_c: np.ndarray
    t_indoor_c: Optional[np.ndarray]
    p_elec_hvac_kw: np.ndarray
    t_set_heating_c: np.ndarray
    t_set_cooling_c: np.ndarray


def simulate_thermal_timeseries(
    model: ThermalModel,
    house: HouseThermalConfig,
    heat_pump: HeatPumpConfig,
    setpoint: SetpointConfig,
    dynamic: bool = True,
    home_hours_of_day: Optional[Sequence[int]] = None,
    n_days: int = 14,
    seed: int = 42,
    start_day: int = 0,
) -> ThermalTimeseriesResult:
    """
    Simulate one ambient-temperature path and the resulting HVAC response.

    Returns the full hourly arrays (outdoor temperature, indoor temperature,
    electric draw, effective setpoints) for a single house configuration over
    a short horizon — the data behind the Phase-19 timeseries preview chart.

    Args:
        model: Calibrated :class:`ThermalModel`.
        house: The :class:`HouseThermalConfig` to preview.
        heat_pump: Heat-pump characterisation.
        setpoint: Comfort setpoints (single or scheduled).
        dynamic: Run the dynamic RC integrator (default ``True``) so an
            indoor-temperature trajectory is produced. ``False`` runs the
            steady-state path (``t_indoor_c`` is ``None``).
        home_hours_of_day: Occupancy hours of day, or ``None`` for always
            home.
        n_days: Horizon length in days. Default 14 (two weeks — enough to see
            the diurnal pattern without a heavy payload).
        seed: RNG seed for the single path.
        start_day: Day-of-year offset (0..364) of the first previewed day
            (Fase 19-bis). Default 0 (January 1). Use e.g. 181 to inspect a
            summer window. The model is simulated from January 1 up to
            ``start_day + n_days`` and the tail ``n_days`` are returned, so the
            AR(1) residual + seasonal mean are warmed up to the right
            day-of-year and the diurnal amplitude matches the season.

    Returns:
        A :class:`ThermalTimeseriesResult`.

    Raises:
        ValueError: If ``n_days`` ≤ 0 or ``start_day`` is outside ``[0, 364]``.
    """
    if n_days <= 0:
        raise ValueError(f"n_days must be > 0, got {n_days}")
    if not 0 <= start_day <= 364:
        raise ValueError(f"start_day must be in [0, 364], got {start_day}")

    n_hours = n_days * 24
    rng = np.random.default_rng(seed)
    total_days = start_day + n_days
    daily_means_full = model.simulate_daily_means(total_days, rng)
    daily_means = daily_means_full[start_day:]
    # Lift to hourly with the correct day-of-year offset so the per-month
    # diurnal amplitude matches the previewed season.
    t_ambient_hourly = model.to_hourly(daily_means, start_day_of_year=start_day)

    controller = HvacController(
        ThermalLoadConfig(
            enabled=True,
            house=house,
            heat_pump=heat_pump,
            setpoint=setpoint,
            dynamic=dynamic,
        )
    )
    at_home_hourly = _build_at_home_hourly(n_hours, home_hours_of_day)
    p_elec, _ = controller.compute_hourly_p_elec_kw(t_ambient_hourly, at_home_hourly)
    t_set_heating, t_set_cooling = controller.setpoint_arrays(at_home_hourly)

    return ThermalTimeseriesResult(
        hours=np.arange(n_hours),
        t_outdoor_c=t_ambient_hourly,
        t_indoor_c=controller.last_indoor_temp_c,
        p_elec_hvac_kw=p_elec,
        t_set_heating_c=t_set_heating,
        t_set_cooling_c=t_set_cooling,
    )
