"""
Representative-week preview of a load profile's consumption personality.

This is the building block of the load-profile **detail page**: the user
picks a profile, a calendar month, and (optionally) a climate, and immediately
sees what a typical week of consumption looks like — the mean hourly load with
a p05–p95 band, split into baseline / discrete appliances / HVAC, plus the
annual kWh totals and (when a climate is selected) the weekly temperature.

No PV, battery, or inverter is involved: this previews the *demand* side only,
so it stays cheap and runs synchronously behind a single API call. It reuses
exactly the same building blocks the full simulator uses — the home/away
sub-profiles, the :class:`StochasticLoadProfile` decorator, the
:class:`EventBasedApplianceProfile`, the :class:`HvacController`, and the
:class:`ThermalModel` — so the preview is faithful to what a scenario would
actually consume.

Two regimes are previewed independently (the user's "scenario misto"):

- ``"home"`` — the full personality: baseline home pattern + daily variability
  + discrete appliances + HVAC.
- ``"away"`` — the semi-constant away pattern + optional daily variability
  only (appliances and HVAC are home-only by construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..calendar_utils import build_calendar
from .load_profiles.appliances import ApplianceProfileConfig, EventBasedApplianceProfile
from .load_profiles.base import LoadProfile
from .load_profiles.stochastic import StochasticLoadConfig, StochasticLoadProfile
from .thermal import ThermalModel
from .thermal_load import HvacController, ThermalLoadConfig

# Hours in the representative week (7 days × 24 hours). Index ``wd*24 + h`` is
# hour ``h`` of weekday ``wd`` (0=Monday … 6=Sunday).
HOURS_PER_WEEK: int = 7 * 24

# Default number of Monte Carlo paths for the preview. Small on purpose: the
# preview is interactive (re-runs as the user tweaks variability), and the
# bands stabilise quickly because the per-hour noise is averaged over the days
# of the month before percentiles are taken.
DEFAULT_PREVIEW_PATHS: int = 80


@dataclass(frozen=True)
class LoadPreviewResult:
    """
    Aggregated representative-week preview of one load-profile regime.

    All weekly arrays are length :data:`HOURS_PER_WEEK` (168), laid out
    weekday-major: index ``wd*24 + h`` is hour ``h`` of weekday ``wd``
    (0=Monday … 6=Sunday). Power is in kW; energy in kWh; temperature in °C.

    Attributes:
        regime: ``"home"`` or ``"away"`` — which sub-profile was previewed.
        month: Calendar month previewed (0=January … 11=December).
        n_paths: Number of Monte Carlo paths aggregated.
        week_hours: ``list(range(168))`` — the x-axis for the weekly charts.
        total_kw_mean / _p05 / _p95: Mean and 5th/95th percentile of the total
            household load across paths, per week-hour.
        baseline_kw_mean: Mean baseline (pattern) contribution per week-hour.
        appliance_kw_mean: Mean discrete-appliance contribution per week-hour.
        hvac_kw_mean: Mean HVAC electric contribution per week-hour.
        annual_kwh_mean: Expected total annual consumption of this regime
            (baseline + appliances + HVAC), kWh/year.
        baseline_kwh_annual: Deterministic annual baseline consumption, kWh/year.
        appliance_kwh_annual: Expected annual appliance consumption, kWh/year.
        hvac_kwh_annual_mean: Mean annual HVAC consumption across paths, kWh/year.
        appliance_kwh_annual_by_name: Expected kWh/year per appliance name.
        has_appliances / has_hvac: Whether the regime included those layers.
        has_thermal: Whether a climate was supplied (enables the temperature
            charts even when HVAC itself is off).
        temp_out_c_mean / _p05 / _p95: Weekly outdoor temperature stats, or
            ``None`` when no climate was supplied.
        temp_in_c_mean: Weekly indoor temperature (dynamic RC HVAC only), or
            ``None`` otherwise.
    """

    regime: str
    month: int
    n_paths: int
    week_hours: list[int]
    total_kw_mean: list[float]
    total_kw_p05: list[float]
    total_kw_p95: list[float]
    baseline_kw_mean: list[float]
    appliance_kw_mean: list[float]
    hvac_kw_mean: list[float]
    annual_kwh_mean: float
    baseline_kwh_annual: float
    appliance_kwh_annual: float
    hvac_kwh_annual_mean: float
    appliance_kwh_annual_by_name: dict[str, float]
    has_appliances: bool
    has_hvac: bool
    has_thermal: bool
    temp_out_c_mean: list[float] | None
    temp_out_c_p05: list[float] | None
    temp_out_c_p95: list[float] | None
    temp_in_c_mean: list[float] | None


def _baseline_annual_kwh(
    base_profile_factory: Callable[[], LoadProfile],
    calendar: tuple[np.ndarray, ...],
    seed: int,
) -> float:
    """
    Deterministic annual baseline consumption (kWh) of the regime profile.

    Sums the *unwrapped* (noise-free) baseline over a full calendar year.
    Because the stochastic multiplier has unit mean (``E[eps]=1``), this equals
    the expected annual baseline, so we avoid a per-path full-year hour loop.

    Args:
        base_profile_factory: Zero-arg factory producing the regime's
            :class:`LoadProfile` (a fresh instance, no decorators).
        calendar: The 5-tuple returned by :func:`build_calendar` for one year.
        seed: Seed used only to reset any inner stochastic state of the base
            profile (deterministic sub-profiles ignore it).

    Returns:
        Annual baseline consumption in kWh (sum of hourly kW over the year).
    """
    (_, month_in_year_for_day, year_index_for_day, day_in_month_for_day, weekday_for_day) = calendar
    base = base_profile_factory()
    base.reset_for_run(rng=np.random.default_rng(seed), n_years=1)
    total = 0.0
    for d in range(len(month_in_year_for_day)):
        miy = int(month_in_year_for_day[d])
        dim = int(day_in_month_for_day[d])
        yi = int(year_index_for_day[d])
        wd = int(weekday_for_day[d])
        for h in range(24):
            total += base.get_hourly_load_kw(yi, miy, dim, h, wd)
    return float(total)


def simulate_load_profile_preview(
    *,
    base_profile_factory: Callable[[], LoadProfile],
    regime: str,
    month: int,
    n_paths: int = DEFAULT_PREVIEW_PATHS,
    seed: int = 42,
    stochastic_config: StochasticLoadConfig | None = None,
    appliance_config: ApplianceProfileConfig | None = None,
    thermal_load_config: ThermalLoadConfig | None = None,
    thermal_model: ThermalModel | None = None,
    solar_hourly_shape: np.ndarray | None = None,
    calendar_start_weekday: int = 0,
) -> LoadPreviewResult:
    """
    Simulate a representative week of one load-profile regime for a month.

    Runs ``n_paths`` independent Monte Carlo paths of the demand side only,
    each refreshing the daily-variability multiplier, the appliance event
    schedule, and (when a climate is supplied) the temperature realisation.
    For the chosen month it averages each day's hourly load into a typical
    week (by weekday), then takes the cross-path mean and p05–p95 band.

    The annual totals are computed cheaply: the baseline analytically (unit-mean
    noise), the appliances from their expected event frequency, and the HVAC as
    the cross-path mean of the per-path annual draw.

    Args:
        base_profile_factory: Zero-arg factory producing a fresh
            :class:`LoadProfile` for the regime (home or away sub-profile).
            Called once per path so each path gets clean stochastic state.
        regime: ``"home"`` or ``"away"`` — recorded in the result; the caller
            is responsible for passing the matching factory/configs (appliances
            and HVAC apply only to the home regime).
        month: Calendar month to preview (0=January … 11=December).
        n_paths: Number of Monte Carlo paths. Default
            :data:`DEFAULT_PREVIEW_PATHS`.
        seed: Master seed for reproducible per-path sub-seeds.
        stochastic_config: Daily-variability config. Applied (wrapping the base
            profile) only when present, ``enabled``, and ``sigma_log > 0``.
        appliance_config: Discrete-appliance config. Applied only when present,
            ``enabled``, and non-empty.
        thermal_load_config: HVAC config. Applied only when present, ``enabled``,
            and a ``thermal_model`` is supplied.
        thermal_model: Calibrated climate model. When supplied, the outdoor
            temperature week is returned even if HVAC is off.
        solar_hourly_shape: 24-length PV shape (sums to 1) used to bias
            ``smart_pv`` appliance starts. Required if any appliance is in
            ``smart_pv`` mode.
        calendar_start_weekday: Weekday of January 1 (0=Monday). Default 0.

    Returns:
        :class:`LoadPreviewResult` with the weekly bands, breakdown, annual
        totals, and (optional) temperature series.

    Raises:
        ValueError: When ``month`` is outside 0–11 or ``n_paths`` is not
            positive.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile
        import numpy as np

        factory = lambda: MonthlyAverageLoadProfile(np.full((12, 24), 300.0))
        res = simulate_load_profile_preview(
            base_profile_factory=factory, regime="home", month=0, n_paths=20
        )
        assert len(res.total_kw_mean) == 168
        assert abs(res.total_kw_mean[12] - 0.3) < 1e-9  # 300 W → 0.3 kW
        ```

    Notes:
        - Deterministic for a fixed ``(seed, n_paths, month, configs)``.
        - The appliance profile mirrors the simulator's 30-day-month internal
          index; the preview drives it through the same calendar coordinates,
          so its contribution matches a real run.
    """
    if not (0 <= month <= 11):
        raise ValueError(f"month must be in 0..11, got {month}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")

    calendar = build_calendar(1, start_weekday=calendar_start_weekday)
    (_, month_in_year_for_day, year_index_for_day, day_in_month_for_day, weekday_for_day) = calendar
    n_days = len(month_in_year_for_day)
    target_days = [d for d in range(n_days) if int(month_in_year_for_day[d]) == month]

    stoch_enabled = (
        stochastic_config is not None
        and stochastic_config.enabled
        and stochastic_config.sigma_log > 0
    )
    appliance_enabled = (
        appliance_config is not None
        and appliance_config.enabled
        and bool(appliance_config.appliances)
    )
    hvac_enabled = (
        thermal_load_config is not None
        and thermal_load_config.enabled
        and thermal_model is not None
    )
    show_temp = thermal_model is not None

    # Number of days of each weekday inside the target month (for averaging the
    # daily curves into one representative week).
    wd_count = np.zeros(7)
    for d in target_days:
        wd_count[int(weekday_for_day[d])] += 1.0
    wd_count_safe = np.where(wd_count > 0, wd_count, 1.0)[:, None]  # (7,1) broadcast

    controller = HvacController(thermal_load_config) if hvac_enabled else None

    # Cross-path accumulators.
    total_paths = np.zeros((n_paths, HOURS_PER_WEEK))
    baseline_acc = np.zeros((7, 24))
    appliance_acc = np.zeros((7, 24))
    hvac_acc = np.zeros((7, 24))
    temp_out_paths = np.zeros((n_paths, HOURS_PER_WEEK)) if show_temp else None
    temp_in_acc = np.zeros((7, 24))
    have_indoor = False
    hvac_kwh_annual_vals: list[float] = []

    master = np.random.default_rng(seed)
    for i in range(n_paths):
        rng = np.random.default_rng(int(master.integers(0, 2_000_000_000)))

        base: LoadProfile = base_profile_factory()
        if stoch_enabled:
            base = StochasticLoadProfile(base, stochastic_config)
        base.reset_for_run(rng=rng, n_years=1)

        appliance: EventBasedApplianceProfile | None = None
        if appliance_enabled:
            appliance = EventBasedApplianceProfile(
                appliance_config.appliances, solar_hourly_shape
            )
            appliance.reset_for_run(rng=rng, n_years=1)

        hvac_hourly: np.ndarray | None = None
        t_amb: np.ndarray | None = None
        indoor: np.ndarray | None = None
        if thermal_model is not None:
            daily_means = thermal_model.simulate_daily_means(n_days, rng)
            t_amb = thermal_model.to_hourly(daily_means)
            if hvac_enabled:
                at_home = np.ones(n_days * 24, dtype=bool)
                hvac_hourly, hvac_kpis = controller.compute_hourly_p_elec_kw(
                    t_amb, at_home
                )
                hvac_kwh_annual_vals.append(float(hvac_kpis.hvac_kwh_annual))
                indoor = controller.last_indoor_temp_c

        total_wd = np.zeros((7, 24))
        base_wd = np.zeros((7, 24))
        app_wd = np.zeros((7, 24))
        hvac_wd = np.zeros((7, 24))
        temp_wd = np.zeros((7, 24))
        tin_wd = np.zeros((7, 24))
        for d in target_days:
            wd = int(weekday_for_day[d])
            miy = int(month_in_year_for_day[d])
            dim = int(day_in_month_for_day[d])
            yi = int(year_index_for_day[d])
            for h in range(24):
                b = base.get_hourly_load_kw(yi, miy, dim, h, wd)
                a = (
                    appliance.get_hourly_load_kw(yi, miy, dim, h, wd)
                    if appliance is not None
                    else 0.0
                )
                hv = float(hvac_hourly[d * 24 + h]) if hvac_hourly is not None else 0.0
                base_wd[wd, h] += b
                app_wd[wd, h] += a
                hvac_wd[wd, h] += hv
                total_wd[wd, h] += b + a + hv
                if t_amb is not None:
                    temp_wd[wd, h] += float(t_amb[d * 24 + h])
                if indoor is not None:
                    tin_wd[wd, h] += float(indoor[d * 24 + h])

        total_wd /= wd_count_safe
        base_wd /= wd_count_safe
        app_wd /= wd_count_safe
        hvac_wd /= wd_count_safe
        temp_wd /= wd_count_safe
        tin_wd /= wd_count_safe

        total_paths[i] = total_wd.reshape(HOURS_PER_WEEK)
        baseline_acc += base_wd
        appliance_acc += app_wd
        hvac_acc += hvac_wd
        if temp_out_paths is not None:
            temp_out_paths[i] = temp_wd.reshape(HOURS_PER_WEEK)
        if indoor is not None:
            temp_in_acc += tin_wd
            have_indoor = True

    total_mean = total_paths.mean(axis=0)
    total_p05 = np.percentile(total_paths, 5, axis=0)
    total_p95 = np.percentile(total_paths, 95, axis=0)
    baseline_mean = (baseline_acc / n_paths).reshape(HOURS_PER_WEEK)
    appliance_mean = (appliance_acc / n_paths).reshape(HOURS_PER_WEEK)
    hvac_mean = (hvac_acc / n_paths).reshape(HOURS_PER_WEEK)

    baseline_kwh_annual = _baseline_annual_kwh(base_profile_factory, calendar, seed)

    appliance_kwh_annual_by_name: dict[str, float] = {}
    appliance_kwh_annual = 0.0
    if appliance_enabled:
        for ev in appliance_config.appliances:
            kwh = float(ev.expected_kwh_annual())
            appliance_kwh_annual_by_name[ev.name] = (
                appliance_kwh_annual_by_name.get(ev.name, 0.0) + kwh
            )
            appliance_kwh_annual += kwh

    hvac_kwh_annual_mean = (
        float(np.mean(hvac_kwh_annual_vals)) if hvac_kwh_annual_vals else 0.0
    )
    annual_kwh_mean = baseline_kwh_annual + appliance_kwh_annual + hvac_kwh_annual_mean

    def _as_list(arr: np.ndarray) -> list[float]:
        return [float(x) for x in arr]

    temp_out_mean = temp_out_p05 = temp_out_p95 = None
    if temp_out_paths is not None:
        temp_out_mean = _as_list(temp_out_paths.mean(axis=0))
        temp_out_p05 = _as_list(np.percentile(temp_out_paths, 5, axis=0))
        temp_out_p95 = _as_list(np.percentile(temp_out_paths, 95, axis=0))
    temp_in_mean = (
        _as_list((temp_in_acc / n_paths).reshape(HOURS_PER_WEEK)) if have_indoor else None
    )

    return LoadPreviewResult(
        regime=regime,
        month=month,
        n_paths=n_paths,
        week_hours=list(range(HOURS_PER_WEEK)),
        total_kw_mean=_as_list(total_mean),
        total_kw_p05=_as_list(total_p05),
        total_kw_p95=_as_list(total_p95),
        baseline_kw_mean=_as_list(baseline_mean),
        appliance_kw_mean=_as_list(appliance_mean),
        hvac_kw_mean=_as_list(hvac_mean),
        annual_kwh_mean=float(annual_kwh_mean),
        baseline_kwh_annual=float(baseline_kwh_annual),
        appliance_kwh_annual=float(appliance_kwh_annual),
        hvac_kwh_annual_mean=float(hvac_kwh_annual_mean),
        appliance_kwh_annual_by_name=appliance_kwh_annual_by_name,
        has_appliances=bool(appliance_enabled),
        has_hvac=bool(hvac_enabled),
        has_thermal=bool(show_temp),
        temp_out_c_mean=temp_out_mean,
        temp_out_c_p05=temp_out_p05,
        temp_out_c_p95=temp_out_p95,
        temp_in_c_mean=temp_in_mean,
    )
