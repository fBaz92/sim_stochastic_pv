"""
Orchestrator for the "Electricity Market" lab section.

This module sits on top of the standalone market engine
(:mod:`sim_stochastic_pv.market`) and turns a user-edited market configuration
(generation mix + capacity trends + fuel/CO2 scenarios) into the handful of
aggregates the web lab visualises:

* a **price heatmap** of the wholesale price by calendar-month × hour-of-day
  for a chosen horizon year;
* an **annual price fan chart** — mean and p05/p95 wholesale price per year over
  the horizon, across market trajectories;
* a **price duration curve** — the year's hourly prices sorted descending;
* a **"who sets the price" heatmap** — the technology most often marginal in
  each month × hour bucket;
* the **installed-capacity trajectory** per technology over the horizon (the
  data behind the trend editor's stacked-area chart).

The price-based views (heatmap, fan, duration) are derived from a cached
:class:`~sim_stochastic_pv.market.PriceSurface`; the "who sets the price" view
comes from a separate, single-year Monte Carlo of the display-year mix (the
surface keeps only prices, not the marginal-technology breakdown). Both are
driven by the same :class:`MixTrend`, so they describe the same market.

The orchestrator follows the thermal-lab pattern: small frozen dataclasses for
the configuration and the result, module-level functions for the behaviour, and
numpy arrays inside the result that the API layer flattens to lists.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ..market import MixTrend, PriceSurface, TechCapacityTrend, build_price_surface
from ..market.config import (
    CO2_SCENARIOS,
    COAL_SCENARIOS,
    GAS_SCENARIOS,
    ITALIAN_MIX,
)
from ..market.simulation import run_monte_carlo
from .market_pricing import MarketPriceProvider


# Days per calendar month (non-leap year), used to weight month×hour price
# buckets into an hours-of-the-year duration curve.
_DAYS_IN_MONTH: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Maximum number of representative years actually dispatched when building the
# price surface; intermediate years are linearly interpolated. Keeps an
# interactive lab run to a couple of seconds even over a 20-year horizon.
_MAX_REPRESENTATIVE_YEARS: int = 4


@dataclass(frozen=True)
class TechTrendSpec:
    """
    Capacity-evolution spec for one generation technology over the horizon.

    Mirrors :class:`~sim_stochastic_pv.market.TechCapacityTrend` but without the
    base capacity (which is taken from the configured mix), so the lab UI can
    edit growth and one-off step changes independently of the starting fleet.

    Attributes:
        annual_growth_pct: Compound annual capacity growth, percent/year
            (e.g. ``5.0`` = +5%/year). May be negative for a phase-out.
        step_year: Optional horizon year (0-based) at which a one-off capacity
            level is imposed (e.g. "nuclear comes online in year 9").
        step_capacity_gw: The installed capacity (GW) imposed from ``step_year``
            onwards; growth then resumes from that level. Required when
            ``step_year`` is set.
    """

    annual_growth_pct: float = 0.0
    step_year: int | None = None
    step_capacity_gw: float | None = None


@dataclass(frozen=True)
class MarketLabConfig:
    """
    Full configuration of a market-lab run.

    Captures everything needed to (re)build a :class:`MixTrend` and the
    derived price surface: per-technology starting capacities, their trends,
    the fuel/CO2 scenario presets and drifts, and the Monte Carlo sizing. The
    starting mix is the packaged ``ITALIAN_MIX`` with the capacities in
    ``capacities_gw`` overridden (other technology parameters — efficiency,
    inertia, emission factor — are kept from the packaged mix).

    Attributes:
        capacities_gw: Override of the year-0 installed capacity (GW) per
            technology. Keys must be a subset of the packaged mix technologies
            (``gas``, ``solar``, ``wind``, ``nuclear``, ``coal``,
            ``hydro_mustrun``). Missing technologies keep their packaged value.
        capacity_trends: Per-technology :class:`TechTrendSpec`. Technologies not
            present default to a flat (no-growth) trend.
        gas_scenario: Key into :data:`GAS_SCENARIOS` (``base``/``tension``/
            ``crisis``) selecting the year-0 gas-price O-U scenario.
        co2_scenario: Key into :data:`CO2_SCENARIOS` (``base``/``low``/``high``)
            or ``None`` to use the engine default.
        coal_scenario: Key into :data:`COAL_SCENARIOS` or ``None`` for the
            engine default.
        gas_mu_drift_annual: Annual drift applied to the gas-price mean over the
            horizon (decimal, e.g. ``0.03`` = +3%/year).
        co2_mu_drift_annual: Annual drift applied to the CO2-price mean.
        n_years: Horizon length in years (≥ 1).
        n_trajectories: Number of independent market trajectories for the price
            surface (the fan-chart band width).
        n_runs: Monte Carlo runs for the single-year "who sets the price"
            breakdown (≥ 1).
        seed: Master RNG seed for reproducibility.
        display_year: Horizon year (0-based) the heatmap / duration curve /
            price-setter view represent.

    Notes:
        - Validation lives at the API boundary (Pydantic) and in the market
          engine; this dataclass trusts its inputs (CLAUDE.md §2.4).
    """

    capacities_gw: Mapping[str, float] = field(default_factory=dict)
    capacity_trends: Mapping[str, TechTrendSpec] = field(default_factory=dict)
    gas_scenario: str = "base"
    co2_scenario: str | None = None
    coal_scenario: str | None = None
    gas_mu_drift_annual: float = 0.0
    co2_mu_drift_annual: float = 0.0
    n_years: int = 20
    n_trajectories: int = 8
    n_runs: int = 6
    seed: int = 42
    display_year: int = 0


@dataclass(frozen=True)
class MarketLabResult:
    """
    Bundle of market-lab aggregates ready for the web visualisations.

    All EUR figures are wholesale prices in EUR/kWh (the engine works in
    EUR/MWh; the surface and this result are EUR/kWh). Arrays are kept as numpy
    here and flattened to lists at the API boundary.

    Attributes:
        techs: Generation technologies in the mix, in display order.
        years: Horizon years (0-based), length ``n_years``.
        capacity_by_year_gw: ``tech -> (n_years,)`` installed capacity (GW).
        display_year: The horizon year the month×hour views describe.
        price_heatmap_eur_per_kwh: Mean wholesale price for the display year,
            shape ``(12, 24)`` (month × hour-of-day).
        annual_price_mean_eur_per_kwh: Across-trajectory mean annual price,
            shape ``(n_years,)``.
        annual_price_p05_eur_per_kwh: 5th-percentile annual price band.
        annual_price_p95_eur_per_kwh: 95th-percentile annual price band.
        duration_curve_x: Fraction-of-year axis (0..1), shape ``(288,)``.
        duration_curve_price_eur_per_kwh: Display-year prices sorted descending,
            aligned with ``duration_curve_x``.
        price_setter_techs: Technologies that can set the price (mix techs plus
            the ``import`` pseudo-technology), in display order.
        price_setter_dominant: For the display year, the index into
            ``price_setter_techs`` of the technology most often marginal in each
            month × hour bucket, shape ``(12, 24)``. ``-1`` marks a bucket where
            no unit ever set a non-zero price.
        price_setter_share_year: ``tech -> fraction`` of the year that
            technology set the marginal price (mean across MC runs).
        mean_price_eur_per_kwh: Mean wholesale price over the display year.
        n_trajectories: Number of price-surface trajectories used.
        n_runs: Number of price-setter Monte Carlo runs used.
    """

    techs: list[str]
    years: list[int]
    capacity_by_year_gw: dict[str, np.ndarray]
    display_year: int
    price_heatmap_eur_per_kwh: np.ndarray
    annual_price_mean_eur_per_kwh: np.ndarray
    annual_price_p05_eur_per_kwh: np.ndarray
    annual_price_p95_eur_per_kwh: np.ndarray
    duration_curve_x: np.ndarray
    duration_curve_price_eur_per_kwh: np.ndarray
    price_setter_techs: list[str]
    price_setter_dominant: np.ndarray
    price_setter_share_year: dict[str, float]
    mean_price_eur_per_kwh: float
    n_trajectories: int
    n_runs: int


def _representative_years(n_years: int, max_points: int = _MAX_REPRESENTATIVE_YEARS) -> list[int]:
    """
    Pick an ascending set of horizon years to actually dispatch.

    Always includes year 0 and the final year; intermediate years are evenly
    spaced. The rest of the horizon is linearly interpolated by
    :func:`build_price_surface`.

    Args:
        n_years: Horizon length (≥ 1).
        max_points: Maximum number of representative years.

    Returns:
        Sorted list of unique 0-based year indices.
    """
    if n_years <= 1:
        return [0]
    k = min(max_points, n_years)
    raw = {int(round(i * (n_years - 1) / (k - 1))) for i in range(k)}
    raw.add(0)
    raw.add(n_years - 1)
    return sorted(raw)


def _build_mix_trend(config: MarketLabConfig) -> MixTrend:
    """
    Assemble a :class:`MixTrend` from the lab configuration.

    Starts from the packaged Italian mix, overrides the year-0 capacities, and
    attaches a :class:`TechCapacityTrend` per technology (flat where the user
    gave no trend).

    Args:
        config: The lab configuration.

    Returns:
        A :class:`MixTrend` ready to feed :func:`build_price_surface` and the
        single-year price-setter Monte Carlo.
    """
    base_mix = copy.deepcopy(dict(ITALIAN_MIX))
    for tech, capacity in config.capacities_gw.items():
        if tech in base_mix:
            base_mix[tech] = {**base_mix[tech], "capacity_gw": float(capacity)}

    trends: dict[str, TechCapacityTrend] = {}
    for tech, params in base_mix.items():
        spec = config.capacity_trends.get(tech)
        base_capacity = float(params.get("capacity_gw", 0.0))
        if spec is None:
            trends[tech] = TechCapacityTrend(base_capacity_gw=base_capacity)
        else:
            trends[tech] = TechCapacityTrend(
                base_capacity_gw=base_capacity,
                annual_growth_pct=spec.annual_growth_pct,
                step_year=spec.step_year,
                step_capacity_gw=spec.step_capacity_gw,
            )

    return MixTrend(
        base_mix=base_mix,
        capacity_trends=trends,
        gas_mu_drift_annual=config.gas_mu_drift_annual,
        co2_mu_drift_annual=config.co2_mu_drift_annual,
    )


def _coal_scenario(config: MarketLabConfig) -> dict | None:
    """Resolve the coal O-U scenario dict (or ``None`` for the engine default)."""
    if config.coal_scenario is None:
        return None
    return dict(COAL_SCENARIOS[config.coal_scenario])


def _co2_scenario(config: MarketLabConfig) -> dict | None:
    """Resolve the CO2 O-U scenario dict (or ``None`` for the engine default)."""
    if config.co2_scenario is None:
        return None
    return dict(CO2_SCENARIOS[config.co2_scenario])


def _build_surface(config: MarketLabConfig, trend: MixTrend) -> PriceSurface:
    """
    Build the cached wholesale :class:`PriceSurface` for the configuration.

    Args:
        config: The lab configuration (sizing + scenarios).
        trend: The mix trend assembled by :func:`_build_mix_trend`.

    Returns:
        A :class:`PriceSurface` of shape ``(n_trajectories, n_years, 12, 24)``
        in EUR/kWh.
    """
    return build_price_surface(
        trend,
        GAS_SCENARIOS[config.gas_scenario],
        n_years=config.n_years,
        n_trajectories=config.n_trajectories,
        coal_scenario=_coal_scenario(config),
        co2_scenario=_co2_scenario(config),
        representative_years=_representative_years(config.n_years),
        seed=config.seed,
    )


def _duration_curve(month_hour_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a price duration curve from a month×hour price grid.

    Each ``(month, hour)`` bucket is weighted by the number of days in that
    month (one hour per day over the month), giving a typical-year hours
    distribution; buckets are then sorted by price descending and a cumulative
    fraction-of-year axis is produced.

    Args:
        month_hour_grid: Wholesale prices, shape ``(12, 24)`` EUR/kWh.

    Returns:
        ``(x_fraction, price_sorted)`` each shape ``(288,)``: ``x_fraction`` is
        the cumulative share of the year (0..1) at the *centre* of each bucket,
        ``price_sorted`` the bucket prices in descending order.
    """
    prices = month_hour_grid.reshape(-1)  # (288,)
    weights = np.repeat(np.asarray(_DAYS_IN_MONTH, dtype=float), 24)  # (288,)
    order = np.argsort(prices)[::-1]
    price_sorted = prices[order]
    weight_sorted = weights[order]
    total = weight_sorted.sum()
    # Cumulative fraction at the centre of each weighted bucket.
    cum = np.cumsum(weight_sorted) - 0.5 * weight_sorted
    x_fraction = cum / total
    return x_fraction, price_sorted


def _price_setter_views(
    config: MarketLabConfig, trend: MixTrend, techs: list[str]
) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """
    Compute the "who sets the price" views via a single-year Monte Carlo.

    Runs the market engine on the display-year mix and aggregates the
    marginal-technology breakdown into (a) the dominant technology per month ×
    hour bucket and (b) the share of the whole year each technology was
    marginal.

    Args:
        config: The lab configuration.
        trend: The assembled mix trend.
        techs: The mix technologies (display order), used to order the legend
            ahead of the ``import`` pseudo-technology.

    Returns:
        ``(setter_techs, dominant_grid, share_year)``:
            - ``setter_techs``: technologies that can set the price, mix order
              then any extra (e.g. ``import``), filtered to those that ever set
              the price.
            - ``dominant_grid``: ``(12, 24)`` indices into ``setter_techs`` of
              the most-often-marginal technology per bucket (``-1`` if none).
            - ``share_year``: ``tech -> fraction`` of the year marginal (mean
              across runs).
    """
    year = min(max(config.display_year, 0), config.n_years - 1)
    mix_year = trend.build_mix_for_year(year)
    gas_year = trend.gas_scenario_for_year(GAS_SCENARIOS[config.gas_scenario], year)
    co2_base = _co2_scenario(config)
    co2_year = (
        trend.co2_scenario_for_year(co2_base, year) if co2_base is not None else None
    )
    mc = run_monte_carlo(
        mix_config=mix_year,
        gas_scenario=gas_year,
        coal_scenario=_coal_scenario(config),
        co2_scenario=co2_year,
        n_runs=config.n_runs,
        seed=config.seed,
    )

    by_month_hour = mc.price_setter_by_month_hour  # tech -> (n, 12, 24)
    pct_by_tech = mc.price_setter_pct_by_tech  # tech -> (n,)

    # Order: mix techs first (in display order), then any extra setter techs
    # (e.g. the ``import`` pseudo-technology), keeping only those that ever set
    # the price.
    extra = [t for t in by_month_hour if t not in techs]
    ordered = [t for t in techs if t in by_month_hour] + sorted(extra)
    setter_techs = [t for t in ordered if float(np.sum(by_month_hour[t])) > 0.0]

    if not setter_techs:
        return [], np.full((12, 24), -1, dtype=int), {}

    # Stack mean-over-runs grids → (n_setter, 12, 24); argmax over tech axis.
    stacked = np.stack(
        [by_month_hour[t].mean(axis=0) for t in setter_techs], axis=0
    )
    total = stacked.sum(axis=0)  # (12, 24)
    dominant = np.argmax(stacked, axis=0).astype(int)
    dominant[total <= 0.0] = -1

    share_year = {
        t: float(np.mean(pct_by_tech[t])) for t in setter_techs if t in pct_by_tech
    }
    return setter_techs, dominant, share_year


def run_market_lab(config: MarketLabConfig) -> MarketLabResult:
    """
    Run the market lab and return every aggregate the web view needs.

    Builds the mix trend, the cached price surface (heatmap / fan / duration)
    and a single-year price-setter Monte Carlo, then packages the results.

    Args:
        config: The validated :class:`MarketLabConfig`.

    Returns:
        A :class:`MarketLabResult` with numpy arrays ready for the API layer.

    Example:
        ```python
        from sim_stochastic_pv.simulation.market_lab import (
            MarketLabConfig, run_market_lab,
        )

        result = run_market_lab(MarketLabConfig(n_years=5, n_trajectories=4, n_runs=4))
        assert result.price_heatmap_eur_per_kwh.shape == (12, 24)
        ```

    Notes:
        - The price views and the "who sets the price" view come from two
          independent dispatch passes of the *same* mix, so they describe the
          same market but are not path-coupled (a documented modelling limit
          consistent with the rest of the market layer).
    """
    trend = _build_mix_trend(config)
    techs = list(trend.base_mix.keys())
    surface = _build_surface(config, trend)

    mean_grid = surface.mean_grid()  # (n_years, 12, 24)
    p05_grid = surface.percentile_grid(5.0)
    p95_grid = surface.percentile_grid(95.0)

    display_year = min(max(config.display_year, 0), config.n_years - 1)
    heatmap = mean_grid[display_year]  # (12, 24)

    # Annual fan: per-trajectory annual mean, then mean / p05 / p95 across them.
    per_traj_annual = surface.price_eur_per_kwh.mean(axis=(2, 3))  # (K, n_years)
    annual_mean = per_traj_annual.mean(axis=0)
    annual_p05 = np.percentile(per_traj_annual, 5, axis=0)
    annual_p95 = np.percentile(per_traj_annual, 95, axis=0)

    dur_x, dur_price = _duration_curve(heatmap)

    capacity_by_year = {
        tech: np.array(
            [trend.capacity_trends[tech].capacity_for_year(y) for y in range(config.n_years)],
            dtype=float,
        )
        for tech in techs
    }

    setter_techs, dominant, share_year = _price_setter_views(config, trend, techs)

    return MarketLabResult(
        techs=techs,
        years=list(range(config.n_years)),
        capacity_by_year_gw=capacity_by_year,
        display_year=display_year,
        price_heatmap_eur_per_kwh=heatmap,
        annual_price_mean_eur_per_kwh=annual_mean,
        annual_price_p05_eur_per_kwh=annual_p05,
        annual_price_p95_eur_per_kwh=annual_p95,
        duration_curve_x=dur_x,
        duration_curve_price_eur_per_kwh=dur_price,
        price_setter_techs=setter_techs,
        price_setter_dominant=dominant,
        price_setter_share_year=share_year,
        mean_price_eur_per_kwh=float(heatmap.mean()),
        n_trajectories=config.n_trajectories,
        n_runs=config.n_runs,
    )


def build_market_provider(
    config: MarketLabConfig,
    *,
    pmg_base_eur_per_kwh: float = 0.0,
    retail_markup_fraction: float | None = None,
    retail_fixed_components_eur_per_kwh: float = 0.0,
) -> MarketPriceProvider:
    """
    Build a :class:`MarketPriceProvider` from a lab configuration.

    Re-builds the price surface for ``config`` and wraps it with the supplied
    dedicated-withdrawal (PMG) and optional retail parameters. Used by the
    "save profile" path so the lab can persist a reusable market profile.

    Args:
        config: The lab configuration to materialise.
        pmg_base_eur_per_kwh: Guaranteed minimum export price at year 0
            (EUR/kWh, ≥ 0).
        retail_markup_fraction: Optional retail markup on the wholesale price.
        retail_fixed_components_eur_per_kwh: Flat per-kWh retail add-on.

    Returns:
        A :class:`MarketPriceProvider` ready to value PV export, or to be
        serialised via :meth:`MarketPriceProvider.to_config_dict`.

    Notes:
        - Rebuilds the surface (a few seconds); saving a profile is a rare
          action, so this is acceptable and keeps the orchestrator stateless.
    """
    trend = _build_mix_trend(config)
    surface = _build_surface(config, trend)
    return MarketPriceProvider(
        surface,
        pmg_base_eur_per_kwh=pmg_base_eur_per_kwh,
        retail_markup_fraction=retail_markup_fraction,
        retail_fixed_components_eur_per_kwh=retail_fixed_components_eur_per_kwh,
    )
