"""
Global configuration and default parameters.

This module centralizes all constants, default scenarios, and system
coefficients used across the simulator. Parameters are organized by category:
time resolution, system base, Monte Carlo settings, dispatch constraints,
load/solar/wind profiles, gas scenarios, and the Italian generation mix.

All prices are in EUR/MWh (electrical) unless noted as EUR/MWh_th (thermal).
All powers are in per-unit of P_BASE (60 GW) internally; GW in config dicts.
Time resolution: quarter-hour (0.25 h). Index 0 = Jan 1 00:00.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Time resolution
# ---------------------------------------------------------------------------
QUARTERS_PER_HOUR: int = 4
"""Number of quarter-hour intervals per hour."""

HOURS_PER_DAY: int = 24
"""Number of hours per day."""

QUARTERS_PER_DAY: int = QUARTERS_PER_HOUR * HOURS_PER_DAY  # 96
"""Number of quarter-hour intervals per day (96)."""

DAYS_PER_YEAR: int = 365
"""Number of days per simulated year (non-leap)."""

QUARTERS_PER_YEAR: int = QUARTERS_PER_DAY * DAYS_PER_YEAR  # 35040
"""Total quarter-hour intervals per year (35 040)."""

# ---------------------------------------------------------------------------
# System base
# ---------------------------------------------------------------------------
P_PEAK_GW: float = 60.0
"""Italian peak load in GW, used as the per-unit base for all power values."""

P_BASE: float = P_PEAK_GW
"""Per-unit base power (alias for P_PEAK_GW)."""

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
N_MC_RUNS: int = 100
"""Default number of Monte Carlo runs for a full simulation."""

RANDOM_SEED: int = 42
"""Base random seed for reproducibility. Each MC run uses seed + run_index."""

# ---------------------------------------------------------------------------
# Dispatch constraints
# ---------------------------------------------------------------------------
H_MIN_SECONDS: float = 3.5
"""Minimum system inertia constant in seconds."""

RESERVE_FRACTION: float = 0.05
"""Spinning reserve requirement as fraction of load (5%)."""

CONTINGENCY_MW_PU: float = 1.8 / P_PEAK_GW
"""Largest credible generation loss (~1.8 GW) expressed in per-unit."""

# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------
CO2_PRICE_DEFAULT: float = 65.0
"""Default EU ETS carbon price in EUR per ton of CO2."""

DISCOUNT_RATE: float = 0.07
"""Discount rate used for capital recovery factor (CRF) calculations."""

# ---------------------------------------------------------------------------
# Load profile factors
# ---------------------------------------------------------------------------
MONTHLY_LOAD_FACTORS: dict[int, float] = {
    1: 0.88, 2: 0.85, 3: 0.82, 4: 0.75, 5: 0.78, 6: 0.90,
    7: 1.00, 8: 0.95, 9: 0.88, 10: 0.80, 11: 0.85, 12: 0.90,
}
"""Monthly load factors (1.0 = peak month = July). Keys are 1-indexed months."""

HOURLY_LOAD_FACTORS: dict[int, float] = {
    0: 0.58, 1: 0.55, 2: 0.53, 3: 0.52, 4: 0.53, 5: 0.56,
    6: 0.62, 7: 0.72, 8: 0.82, 9: 0.90, 10: 0.95, 11: 0.97,
    12: 0.95, 13: 0.93, 14: 0.92, 15: 0.91, 16: 0.90, 17: 0.92,
    18: 0.96, 19: 1.00, 20: 0.98, 21: 0.93, 22: 0.82, 23: 0.70,
}
"""Hourly load factors (fraction of daily peak, 0-indexed hours)."""

WEEKDAY_LOAD_FACTORS: dict[int, float] = {
    0: 1.00,  # Monday
    1: 1.00,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.00,  # Thursday
    4: 1.00,  # Friday
    5: 0.85,  # Saturday
    6: 0.75,  # Sunday
}
"""Day-of-week load multipliers (0=Monday, 6=Sunday).

Weekend demand is lower than weekday demand. Saturday sees partial commercial
activity (0.85), Sunday is the lowest (0.75). Applied multiplicatively on top
of monthly and hourly factors.
"""

HOLIDAY_LOAD_FACTOR: float = 0.80
"""Load multiplier for public holidays.

Applied on top of all other factors (monthly, hourly, weekday). A holiday
falling on Sunday compounds: 0.75 * 0.80 = 0.60 of peak, which reflects
the near-total shutdown of industrial and commercial loads.
"""

ITALIAN_HOLIDAYS_DOY: list[int] = [
    0,    # 1 January — New Year's Day
    5,    # 6 January — Epiphany
    114,  # 25 April — Liberation Day
    120,  # 1 May — Labour Day
    152,  # 2 June — Republic Day
    226,  # 15 August — Ferragosto
    304,  # 1 November — All Saints' Day
    341,  # 8 December — Immaculate Conception
    358,  # 25 December — Christmas Day
    359,  # 26 December — St. Stephen's Day
]
"""Italian public holidays as day-of-year indices (0-indexed, 0 = Jan 1).

Only fixed-date holidays are included. Easter and Easter Monday are excluded
because the simulated year is generic (no specific calendar year), and their
floating dates would require an arbitrary choice. Their omission has negligible
impact on annual statistics (~2 days out of 365).
"""

DEFAULT_LOAD_NOISE_SIGMA: float = 0.04
"""Default standard deviation for multiplicative Gaussian load noise.

Increased from the original 0.02 to better capture intra-day demand variability
(weather-driven HVAC swings, random industrial load). The noise is clipped to
[0.5, 1.5] to prevent unrealistic extremes.
"""

# ---------------------------------------------------------------------------
# Solar profile parameters
# ---------------------------------------------------------------------------
MONTHLY_SOLAR_FACTORS: dict[int, float] = {
    1: 0.30, 2: 0.40, 3: 0.60, 4: 0.75, 5: 0.90, 6: 0.95,
    7: 1.00, 8: 0.95, 9: 0.75, 10: 0.55, 11: 0.35, 12: 0.28,
}
"""Monthly solar irradiance factors (1.0 = July). Keys are 1-indexed months."""


def _solar_envelope() -> dict[int, float]:
    """Compute hourly solar irradiance envelope using a Gaussian centered at 13:00.

    The envelope approximates Italian solar noon (~13:00 local time) with a
    standard deviation of 2.8 hours. Night hours (0-5 and 21-23) are hard-zeroed.

    Returns:
        dict[int, float]: Mapping from hour (0-23) to normalized irradiance
            factor (0.0 to 1.0).
    """
    hours = np.arange(24)
    envelope = np.exp(-0.5 * ((hours - 13.0) / 2.8) ** 2)
    envelope[0:6] = 0.0
    envelope[21:] = 0.0
    envelope /= envelope.max()
    return {h: float(envelope[h]) for h in range(24)}


HOURLY_SOLAR_ENVELOPE: dict[int, float] = _solar_envelope()
"""Hourly solar irradiance envelope (Gaussian, peak at 13:00, zero at night)."""

# ---------------------------------------------------------------------------
# Wind profile parameters
# ---------------------------------------------------------------------------
MONTHLY_WIND_LAMBDA: dict[int, float] = {
    1: 8.5, 2: 8.2, 3: 7.8, 4: 7.5, 5: 6.8, 6: 6.0,
    7: 5.5, 8: 5.8, 9: 6.5, 10: 7.0, 11: 7.8, 12: 8.3,
}
"""Monthly Weibull scale parameter (m/s) for Italian average onshore wind."""

WIND_WEIBULL_K: float = 2.0
"""Weibull shape parameter for wind speed distribution."""

WIND_CUT_IN: float = 3.0
"""Turbine cut-in wind speed (m/s)."""

WIND_RATED: float = 12.0
"""Turbine rated wind speed (m/s) at which full power is reached."""

WIND_CUT_OUT: float = 25.0
"""Turbine cut-out wind speed (m/s) above which turbine shuts down."""

# ---------------------------------------------------------------------------
# Cloud Markov chain
# ---------------------------------------------------------------------------
CLOUD_TRANSITION: dict[int, tuple[float, float]] = {
    1:  (0.40, 0.35),  2: (0.38, 0.37),  3: (0.30, 0.40),
    4:  (0.25, 0.45),  5: (0.20, 0.50),  6: (0.15, 0.55),
    7:  (0.10, 0.60),  8: (0.12, 0.58),  9: (0.20, 0.50),
    10: (0.30, 0.40), 11: (0.38, 0.35), 12: (0.42, 0.33),
}
"""Monthly cloud state transition probabilities.

Each entry is ``(P(cloudy|sunny), P(sunny|cloudy))`` for the two-state
daily Markov chain used by :class:`~sim_stochastic_pv.market.generators.SolarAvailability`.
"""

# ---------------------------------------------------------------------------
# Gas price scenarios (TTF EUR/MWh_th)
# ---------------------------------------------------------------------------
GAS_SCENARIOS: dict[str, dict[str, float]] = {
    'base':    {'mu': 35.0, 'sigma': 8.0,  'theta': 0.1},
    'tension': {'mu': 55.0, 'sigma': 15.0, 'theta': 0.1},
    'crisis':  {'mu': 90.0, 'sigma': 25.0, 'theta': 0.1},
}
"""Gas price scenario parameters for the Ornstein-Uhlenbeck fuel price model.

Keys:
    mu: Long-run mean price (EUR/MWh_th).
    sigma: Volatility.
    theta: Mean-reversion speed.
"""

# ---------------------------------------------------------------------------
# Coal price scenarios (EUR/MWh_th)
# ---------------------------------------------------------------------------
COAL_SCENARIOS: dict[str, dict[str, float]] = {
    'base':    {'mu': 12.0, 'sigma': 3.0, 'theta': 0.05},
    'tension': {'mu': 18.0, 'sigma': 5.0, 'theta': 0.05},
    'crisis':  {'mu': 25.0, 'sigma': 8.0, 'theta': 0.05},
}
"""Coal price scenario parameters for the Ornstein-Uhlenbeck fuel price model.

Coal is cheaper than gas per MWh_th but has lower volatility and slower
mean-reversion. With high CO₂ prices (>60 EUR/ton), coal SRMC can exceed
gas SRMC ("fuel switching").

Keys:
    mu: Long-run mean price (EUR/MWh_th).
    sigma: Volatility.
    theta: Mean-reversion speed.
"""

# ---------------------------------------------------------------------------
# CO2 price scenarios (EUR/ton)
# ---------------------------------------------------------------------------
CO2_SCENARIOS: dict[str, dict[str, float]] = {
    'base':    {'mu': 65.0, 'sigma': 10.0, 'theta': 0.05},
    'low':     {'mu': 40.0, 'sigma': 8.0,  'theta': 0.05},
    'high':    {'mu': 100.0, 'sigma': 15.0, 'theta': 0.05},
}
"""CO2 ETS price scenario parameters for the Ornstein-Uhlenbeck carbon price model.

Slower mean-reversion than gas (theta=0.05) reflects the ETS market's
structural inertia. A volatile CO₂ price creates timesteps where coal is
cheaper than gas and vice versa, producing realistic fuel-switching behavior.

Keys:
    mu: Long-run mean CO2 price (EUR/ton).
    sigma: Volatility.
    theta: Mean-reversion speed.
"""

# ---------------------------------------------------------------------------
# Italian generation mix defaults
# ---------------------------------------------------------------------------
ITALIAN_MIX: dict[str, dict] = {
    'gas': {
        'capacity_gw': 45.0,
        'capex_per_kw': 900,
        'lifetime_years': 27,
        'vom_eur_mwh': 3.0,
        'fom_eur_kw_yr': 20.0,
        'efficiency': 0.58,
        'emission_factor': 0.20,
        'h_inertia': 4.5,
        'min_stable_pct': 0.40,
        'ramp_rate_pct_per_min': 0.06,
        'startup_cost_eur_mw': 50.0,
    },
    'solar': {
        'capacity_gw': 30.0,
        'capex_per_kw': 550,
        'lifetime_years': 28,
        'vom_eur_mwh': 0.5,
        'fom_eur_kw_yr': 10.0,
        'efficiency': 1.0,
        'emission_factor': 0.0,
        'h_inertia': 0.0,
        'min_stable_pct': 0.0,
        'ramp_rate_pct_per_min': 1.0,
        'startup_cost_eur_mw': 0.0,
    },
    'wind': {
        'capacity_gw': 13.0,
        'capex_per_kw': 1250,
        'lifetime_years': 22,
        'vom_eur_mwh': 1.5,
        'fom_eur_kw_yr': 32.0,
        'efficiency': 1.0,
        'emission_factor': 0.0,
        'h_inertia': 0.0,
        'min_stable_pct': 0.0,
        'ramp_rate_pct_per_min': 1.0,
        'startup_cost_eur_mw': 0.0,
    },
    'nuclear': {
        'capacity_gw': 0.0,
        'capex_per_kw': 5500,
        'lifetime_years': 60,
        'vom_eur_mwh': 2.5,
        'fom_eur_kw_yr': 80.0,
        'efficiency': 0.33,
        'emission_factor': 0.0,
        'h_inertia': 6.0,
        'min_stable_pct': 0.50,
        'ramp_rate_pct_per_min': 0.03,
        'startup_cost_eur_mw': 200.0,
        'fuel_cost_eur_mwh_th': 3.0,
    },
    'coal': {
        'capacity_gw': 0.0,
        'capex_per_kw': 1500,
        'lifetime_years': 40,
        'vom_eur_mwh': 4.0,
        'fom_eur_kw_yr': 35.0,
        'efficiency': 0.40,
        'emission_factor': 0.34,
        'h_inertia': 5.0,
        'min_stable_pct': 0.45,
        'ramp_rate_pct_per_min': 0.02,
        'startup_cost_eur_mw': 80.0,
    },
    'hydro_mustrun': {
        'capacity_gw': 8.0,
        'capex_per_kw': 0,
        'lifetime_years': 80,
        'vom_eur_mwh': 0.0,
        'fom_eur_kw_yr': 0.0,
        'efficiency': 1.0,
        'emission_factor': 0.0,
        'h_inertia': 3.5,
        'min_stable_pct': 1.0,
        'ramp_rate_pct_per_min': 0.0,
        'startup_cost_eur_mw': 0.0,
    },
}
"""Default Italian generation mix parameters.

Each generator type maps to a dict with:
    capacity_gw (float): Installed capacity in GW.
    capex_per_kw (float): Capital expenditure in EUR per kW.
    lifetime_years (float): Economic lifetime in years.
    vom_eur_mwh (float): Variable O&M cost in EUR/MWh.
    fom_eur_kw_yr (float): Fixed O&M cost in EUR per kW per year.
    efficiency (float): Thermal-to-electric efficiency (1.0 for renewables).
    emission_factor (float): CO2 emissions in tCO2/MWh_th.
    h_inertia (float): Inertia constant H in seconds (0 for non-synchronous).
    min_stable_pct (float): Minimum stable generation as fraction of capacity.
    ramp_rate_pct_per_min (float): Ramp rate as fraction of capacity per minute.
    startup_cost_eur_mw (float): Start-up cost in EUR per MW.
"""

# ---------------------------------------------------------------------------
# Price areas (neighbouring electricity markets)
# ---------------------------------------------------------------------------
PRICE_AREAS: dict[str, dict] = {
    'FR': {
        'mu': 45.0, 'sigma': 15.0, 'theta': 0.05,
        'carbon_intensity_g_per_kwh': 50.0,
    },
    'CH': {
        'mu': 50.0, 'sigma': 12.0, 'theta': 0.05,
        'carbon_intensity_g_per_kwh': 30.0,
    },
    'AT': {
        'mu': 55.0, 'sigma': 18.0, 'theta': 0.05,
        'carbon_intensity_g_per_kwh': 180.0,
    },
    'SI': {
        'mu': 58.0, 'sigma': 16.0, 'theta': 0.05,
        'carbon_intensity_g_per_kwh': 220.0,
    },
    'GR': {
        'mu': 70.0, 'sigma': 22.0, 'theta': 0.05,
        'carbon_intensity_g_per_kwh': 430.0,
    },
}
"""External electricity price areas (neighbouring markets).

Each area is an independent exogenous market modelled as an Ornstein-Uhlenbeck
stochastic process. Areas are linked to the domestic system by one or more
:class:`~sim_stochastic_pv.market.interconnections.Interconnection` objects.

Values are annual averages roughly consistent with 2022-2023 day-ahead
observations and ENTSO-E carbon intensity data. The ``carbon_intensity_g_per_kwh``
is used for consumption-based emission accounting (emissions embedded in imports).

Keys:
    mu (float): Long-run mean day-ahead price (EUR/MWh).
    sigma (float): Volatility of the O-U process.
    theta (float): Mean-reversion speed.
    carbon_intensity_g_per_kwh (float): Average emission intensity of the
        neighbour's generation mix (gCO₂/kWh).
"""

PRICE_AREA_CORRELATIONS: dict[tuple[str, str], float] = {
    ('FR', 'CH'): 0.80,
    ('FR', 'AT'): 0.65,
    ('FR', 'SI'): 0.55,
    ('FR', 'GR'): 0.30,
    ('CH', 'AT'): 0.75,
    ('CH', 'SI'): 0.60,
    ('CH', 'GR'): 0.30,
    ('AT', 'SI'): 0.70,
    ('AT', 'GR'): 0.40,
    ('SI', 'GR'): 0.45,
}
"""Pairwise correlations between price areas (O-U shock correlations).

Only unordered pairs need to be specified; symmetry is enforced in
:class:`~sim_stochastic_pv.market.price_areas.PriceAreaCoupling`. Missing pairs default to 0.
Values are order-of-magnitude empirical correlations of day-ahead price
*increments* (not levels) across the EU coupled markets.

The full correlation matrix must be positive semidefinite — the coupling
class validates this and raises :class:`ValueError` otherwise.
"""

PRICE_AREAS_CORRELATED: bool = True
"""Master switch for price area correlation.

When ``False``, each area's O-U process is simulated independently (useful
for sensitivity checks and for users who do not have correlation data).
When ``True``, shocks are correlated according to :data:`PRICE_AREA_CORRELATIONS`.
"""

# ---------------------------------------------------------------------------
# Interconnections (cross-border transmission links)
# ---------------------------------------------------------------------------
INTERCONNECTIONS: dict[str, dict] = {
    'IT-FR': {
        'price_area': 'FR',
        'ntc_import_gw': 4.5,
        'ntc_export_gw': 2.8,
        'transport_cost_eur_mwh': 3.0,
        'seasonal_ntc_factor_monthly': None,
        'reliability': {
            'type': 'technology',
            'tech': 'hvdc_back_to_back',
        },
    },
    'IT-CH': {
        'price_area': 'CH',
        'ntc_import_gw': 6.2,
        'ntc_export_gw': 3.5,
        'transport_cost_eur_mwh': 2.5,
        'seasonal_ntc_factor_monthly': None,
        'reliability': {
            'type': 'technology',
            'tech': 'overhead_ac',
        },
    },
    'IT-AT': {
        'price_area': 'AT',
        'ntc_import_gw': 0.3,
        'ntc_export_gw': 0.3,
        'transport_cost_eur_mwh': 4.0,
        'seasonal_ntc_factor_monthly': None,
        'reliability': {
            'type': 'technology',
            'tech': 'overhead_ac',
        },
    },
    'IT-SI': {
        'price_area': 'SI',
        'ntc_import_gw': 0.6,
        'ntc_export_gw': 0.45,
        'transport_cost_eur_mwh': 3.5,
        'seasonal_ntc_factor_monthly': None,
        'reliability': {
            'type': 'technology',
            'tech': 'overhead_ac',
        },
    },
    'IT-GR': {
        'price_area': 'GR',
        'ntc_import_gw': 0.5,
        'ntc_export_gw': 0.5,
        'transport_cost_eur_mwh': 5.0,
        'seasonal_ntc_factor_monthly': None,
        'reliability': {
            'type': 'technology',
            'tech': 'submarine_cable_hvdc',
            'mttr_sigma_log': 1.8,
        },
    },
}
"""Default Italian interconnection topology.

Each entry defines a commercial border with a neighbouring price area.
NTC values are reference figures consistent with Terna publications; actual
monthly NTC varies with grid conditions and is not modelled at that resolution.

Keys:
    price_area (str): Name of a key in :data:`PRICE_AREAS`. The foreign
        electricity price for this link is the realised path of that area.
    ntc_import_gw (float): Maximum commercial flow into Italy (GW).
    ntc_export_gw (float): Maximum commercial flow out of Italy (GW).
    transport_cost_eur_mwh (float): Wheeling fee + loss compensation
        (EUR/MWh, applied to both directions).
    seasonal_ntc_factor_monthly (list[float] | None): Optional 12-element
        list of monthly multipliers on the nominal NTC. ``None`` = constant
        NTC year-round.
    reliability (dict): Reliability model specification, routed through
        :func:`~sim_stochastic_pv.market.reliability.build_reliability_model`. Supported
        ``type`` values: ``'perfect'``, ``'explicit'``, ``'availability'``,
        ``'technology'``.
"""

ENABLE_NTC_FAULTS: bool = True
"""Master switch for interconnection stochastic faults.

When ``False``, every interconnection uses
:class:`~sim_stochastic_pv.market.reliability.PerfectReliability`, overriding the per-link
``reliability`` config. Useful for deterministic runs and baseline comparisons.
"""

ENABLE_INTERCONNECTIONS: bool = True
"""Master switch for cross-border exchanges.

When ``False``, no import virtual generators or export adjustments are added
to the dispatch, regardless of :data:`INTERCONNECTIONS` content.
"""

# ---------------------------------------------------------------------------
# Battery storage (utility-scale BESS)
# ---------------------------------------------------------------------------
STORAGE_UNITS: dict[str, dict] = {
    'aggregate_bess': {
        'energy_capacity_gwh': 4.0,
        'power_capacity_gw': 2.0,
        'efficiency_roundtrip': 0.88,
        'soc_min_frac': 0.10,
        'soc_max_frac': 0.90,
        'initial_soc_frac': 0.50,
        'self_discharge_per_day': 0.001,
        'h_synthetic': 4.0,
        'inertia_soc_margin': 0.02,
    },
}
"""Default aggregate battery storage for the Italian system.

A single aggregated unit representing ~2 GW / 4 GWh of grid-scale BESS,
roughly consistent with PNIEC 2030 targets. The round-trip efficiency of
0.88 is typical of current LFP utility-scale systems including inverter
and auxiliary losses. Synthetic inertia of 4 s emulates a gas turbine.

Keys (per unit):
    energy_capacity_gwh (float): Nameplate energy capacity (GWh AC).
    power_capacity_gw (float): Nameplate charge/discharge power (GW AC).
    efficiency_roundtrip (float): AC-AC round-trip efficiency.
    soc_min_frac, soc_max_frac (float): Operational SOC band fractions.
    initial_soc_frac (float): Starting SOC at t=0.
    self_discharge_per_day (float): Idle energy loss per 24 h.
    h_synthetic (float): Emulated inertia constant in seconds.
    inertia_soc_margin (float): SOC headroom required for inertia support.
"""

STORAGE_PERCENTILE_WINDOW_QH: int = 672
"""Rolling window length (quarter-hours) for the percentile arbitrage.

672 qh = 7 days — captures a full weekly price cycle, long enough to
estimate stable quantiles but short enough to track slow price drifts
(gas shocks, seasonal transitions).
"""

STORAGE_CHARGE_PERCENTILE: float = 25.0
"""Charge when current marginal price is below this percentile of the window."""

STORAGE_DISCHARGE_PERCENTILE: float = 75.0
"""Discharge when current marginal price is above this percentile of the window."""

ENABLE_STORAGE: bool = True
"""Master switch for battery storage in the dispatch pipeline.

When ``False``, the dispatch behaves exactly as in previous phases
(merit order + inertia fix + export adjustment), and no storage arrays
are populated on :class:`~sim_stochastic_pv.market.dispatch.DispatchResult`. Useful
for baseline runs and for isolating the marginal contribution of storage
to prices, curtailment and emissions.
"""
