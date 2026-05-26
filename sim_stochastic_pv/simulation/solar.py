"""
Stochastic solar photovoltaic production model.

Provides :class:`SolarMonthParams` to capture seasonal statistics and
:class:`SolarModel` which combines monthly variability, weather stochasticity,
and long-term degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# Empirical azimuth factors for Italian latitudes (Northern Hemisphere, ~35-45°N)
# Based on real PV system measurements accounting for direct + diffuse radiation
AZIMUTH_FACTORS = {
    0: 0.68,    # North - mostly diffuse + reflected light
    45: 0.78,   # Northeast
    90: 0.84,   # East - morning sun
    135: 0.93,  # Southeast - excellent
    180: 1.00,  # South - optimal
    225: 0.93,  # Southwest - excellent
    270: 0.84,  # West - afternoon sun
    315: 0.78,  # Northwest
}


@dataclass
class SolarMonthParams:
    """
    Monthly parameters for stochastic solar PV production modelling.

    Encapsulates the statistical characteristics of solar energy production
    for a single month so the Monte Carlo simulator can draw realistic
    sunny/cloudy day sequences. The model classifies every day as either
    *sunny* or *cloudy* and applies the corresponding production multiplier.

    Day-to-day dependence is governed by a two-state Markov chain whose
    transition probabilities are computed from two user-facing knobs:
    ``p_sunny`` (the long-term marginal probability of a sunny day) and
    ``weather_persistence`` (the strength of day-to-day autocorrelation).
    The transition matrix is built so that ``p_sunny`` is the *stationary
    distribution* of the chain by construction:

        p_ss = p_sunny      + (1 − p_sunny) · persistence
        p_cs = p_sunny      · (1 − persistence)
        p_sc = (1 − p_sunny) · (1 − persistence)
        p_cc = (1 − p_sunny) + p_sunny      · persistence

    Setting ``weather_persistence = 0.0`` recovers the legacy iid Bernoulli
    behaviour (no memory). Setting it to ``1.0`` produces perfect runs
    (whatever state starts the month is held until the next month).

    Attributes:
        avg_daily_kwh_per_kwp: Average daily energy production per installed
            kWp (kWh/kWp/day). Baseline for an "average" day in the month.
            Typical values for Northern Italy:
            - Winter (Dec-Feb): 1.0-2.5
            - Shoulder (Mar-May, Sep-Nov): 3.0-5.0
            - Summer (Jun-Aug): 5.0-6.0
            Annual total typically 1100-1400 kWh/kWp/year.
        p_sunny: Long-term marginal probability of a sunny day in this month
            (0.0 to 1.0). When ``weather_persistence > 0`` this is also the
            stationary distribution of the underlying Markov chain.
        sunny_factor: Production multiplier for sunny days (typically > 1.0,
            range 1.1-1.3). Applied to ``avg_daily_kwh_per_kwp``.
        cloudy_factor: Production multiplier for cloudy days (typically < 1.0,
            range 0.2-0.5). Applied to ``avg_daily_kwh_per_kwp``.
        weather_persistence: Day-to-day persistence parameter (0.0 to 1.0).
            - 0.0 → memoryless (iid Bernoulli, legacy behaviour).
            - 0.2-0.5 → realistic climatological values for Italy.
            - 1.0 → state never flips within a month.
            Defaults to 0.0 so existing call sites that omit this argument
            keep their previous semantics.

    Example:
        ```python
        # Summer month in Northern Italy with realistic persistence
        june_params = SolarMonthParams(
            avg_daily_kwh_per_kwp=5.46,
            p_sunny=0.70,           # 70% sunny on average
            sunny_factor=1.2,
            cloudy_factor=0.3,
            weather_persistence=0.4 # moderate summer persistence
        )

        # Same month modelled as iid (back-compat)
        june_iid = SolarMonthParams(
            avg_daily_kwh_per_kwp=5.46,
            p_sunny=0.70,
            sunny_factor=1.2,
            cloudy_factor=0.3,
            # weather_persistence omitted → 0.0
        )
        ```

    Notes:
        - For balance of the model it is good practice to choose
          ``sunny_factor`` and ``cloudy_factor`` so that
          ``p_sunny·sunny_factor + (1-p_sunny)·cloudy_factor ≈ 1.0``,
          which keeps the long-run mean equal to ``avg_daily_kwh_per_kwp``.
        - This is a *bimodal* model — it does not represent partly cloudy
          conditions or hour-by-hour intra-day variability.
        - Calibrate ``weather_persistence`` from observed lag-1
          autocorrelation of clear-sky-index time series for the location.
    """
    avg_daily_kwh_per_kwp: float
    p_sunny: float
    sunny_factor: float
    cloudy_factor: float
    weather_persistence: float = 0.0


class SolarModel:
    """
    Stochastic photovoltaic energy production model with degradation.

    Simulates realistic solar PV energy generation over multiple years with:
    - Stochastic daily weather variability (sunny/cloudy classification)
    - Monthly seasonal patterns (different production per month)
    - Long-term panel degradation (capacity fade over time)
    - Fixed hourly production shape (Gaussian daylight curve)

    The model operates at two time scales:
    1. Daily: Stochastic production based on weather (via simulate_daily_energy)
    2. Hourly: Deterministic shape applied to daily totals (via daily_profile_kwh)

    This hybrid approach balances realism with computational efficiency,
    capturing the economic impact of weather variability without requiring
    minute-by-minute irradiance simulation.

    Attributes:
        pv_kwp: Installed PV system capacity (kWp).
            Nameplate DC power rating under standard test conditions.
        month_params: List of 12 SolarMonthParams objects (one per month).
            Defines monthly production characteristics and weather probabilities.
        degradation_per_year: Annual degradation rate as fraction (0-1).
            Panel capacity loss per year. Typical: 0.005-0.01 (0.5-1.0%/year).
            Example: 0.007 = 0.7% annual degradation (common assumption).
        hourly_shape: Normalized 24-hour production profile (array, sums to 1.0).
            Gaussian-shaped daylight curve (6am-6pm) for splitting daily totals
            into hourly values. Generated automatically during initialization.

    Example:
        ```python
        from sim_stochastic_pv.simulation import make_default_solar_params_for_pavullo

        # Create solar model for 5 kWp system
        params = make_default_solar_params_for_pavullo()
        solar = SolarModel(
            pv_kwp=5.0,
            month_params=params,
            degradation_per_year=0.007  # 0.7% per year
        )

        # Simulate 20 years of daily production
        import numpy as np
        n_years = 20
        n_days = n_years * 365

        # Create day indices
        month_in_year = np.array([d // 30 % 12 for d in range(n_days)])
        year_index = np.array([d // 365 for d in range(n_days)])

        # Generate stochastic daily production
        rng = np.random.default_rng(seed=42)
        daily_kwh = solar.simulate_daily_energy(n_years, month_in_year, year_index, rng)

        # Convert to hourly for simulation
        hourly_kwh = solar.daily_profile_kwh(daily_kwh[0])  # First day's profile

        print(f"Year 1 avg: {daily_kwh[:365].mean():.1f} kWh/day")
        print(f"Year 20 avg: {daily_kwh[-365:].mean():.1f} kWh/day")
        # Year 20 will be ~13% lower due to degradation
        ```

    Notes:
        - Production varies stochastically day-to-day (weather)
        - Hourly shape is deterministic (same curve every day)
        - Degradation is compounding: (1 - rate)^years
        - After 20 years with 0.7%/year: capacity = 86.9% of original
        - Model assumes no sudden failures or catastrophic degradation
    """

    def __init__(
        self,
        pv_kwp: float,
        month_params: List[SolarMonthParams],
        degradation_per_year: float = 0.007,
        optimal_tilt_degrees: float = 35.0,
        optimal_azimuth_degrees: float = 180.0,
        panel_tilt_degrees: float | None = None,
        panel_azimuth_degrees: float | None = None,
    ) -> None:
        """
        Initialize a stochastic solar PV production model with degradation and orientation.

        Sets up the model with system capacity, monthly weather parameters,
        degradation assumptions, and panel orientation. Automatically applies
        orientation correction factor if panel installation differs from optimal.

        Args:
            pv_kwp: PV system capacity in kilowatt-peak (kWp).
                Installed DC nameplate capacity under STC (1000 W/m², 25°C).
                Typical residential: 3-10 kWp. Commercial: 20-100+ kWp.
            month_params: List of exactly 12 SolarMonthParams objects.
                Must contain one entry per month (January = index 0).
                Defines monthly production baselines and weather probabilities.
                See make_default_solar_params_for_pavullo() for example.
            degradation_per_year: Annual capacity degradation rate (0-1).
                Fractional loss per year. Typical values:
                - 0.005 (0.5%/year): High-quality panels with warranty
                - 0.007 (0.7%/year): Standard assumption (default)
                - 0.010 (1.0%/year): Budget panels or harsh environment
                Defaults to 0.007 (0.7% per year).
            optimal_tilt_degrees: Tilt angle for which production data is calibrated.
                Typically equals latitude for mid-latitudes (35-45° for Italy).
                Defaults to 35.0 degrees.
            optimal_azimuth_degrees: Azimuth for which production data is calibrated.
                180° = south (optimal for Northern Hemisphere).
                Defaults to 180.0 degrees (south).
            panel_tilt_degrees: Actual installed panel tilt angle (0-90°).
                If None, assumes optimal_tilt_degrees (no correction applied).
                If different from optimal, reduces production accordingly.
            panel_azimuth_degrees: Actual installed panel azimuth (0-360°).
                0° = north, 90° = east, 180° = south, 270° = west.
                If None, assumes optimal_azimuth_degrees (no correction applied).
                If different from optimal, reduces production accordingly.

        Raises:
            ValueError: If month_params does not contain exactly 12 entries.

        Example:
            ```python
            # Standard 6 kWp south-facing system at optimal tilt
            params = make_default_solar_params_for_pavullo()
            model = SolarModel(
                pv_kwp=6.0,
                month_params=params,
                degradation_per_year=0.007
            )

            # East-facing rooftop installation (reduced production)
            model_east = SolarModel(
                pv_kwp=6.0,
                month_params=params,
                optimal_tilt_degrees=35.0,  # Data calibrated for 35°
                optimal_azimuth_degrees=180.0,  # Data calibrated for south
                panel_tilt_degrees=35.0,  # Installed at 35° (optimal)
                panel_azimuth_degrees=90.0  # But facing EAST (84% of optimal)
            )

            # Flat roof with suboptimal tilt
            model_flat = SolarModel(
                pv_kwp=6.0,
                month_params=params,
                panel_tilt_degrees=10.0,  # Nearly flat (reduces production)
                panel_azimuth_degrees=180.0  # South-facing (optimal azimuth)
            )
            ```

        Notes:
            - Hourly shape is Gaussian with peak at noon, 6am-6pm window
            - Shape is normalized (sums to 1.0) for energy distribution
            - month_params order matters: [Jan, Feb, ..., Dec]
            - Degradation compounds: year N capacity = (1 - rate)^N
            - Orientation factor automatically applied to pv_kwp if non-optimal
            - See AZIMUTH_FACTORS for empirical orientation derating values
        """
        if len(month_params) != 12:
            raise ValueError("month_params must contain 12 entries")

        self.month_params = month_params
        self.degradation_per_year = degradation_per_year
        self.optimal_tilt_degrees = optimal_tilt_degrees
        self.optimal_azimuth_degrees = optimal_azimuth_degrees

        # Determine actual panel orientation (defaults to optimal)
        actual_tilt = panel_tilt_degrees if panel_tilt_degrees is not None else optimal_tilt_degrees
        actual_azimuth = panel_azimuth_degrees if panel_azimuth_degrees is not None else optimal_azimuth_degrees

        # Compute orientation factor and apply to capacity
        orientation_factor = self.compute_orientation_factor(actual_tilt, actual_azimuth)
        self.pv_kwp = pv_kwp * orientation_factor
        self.orientation_factor = orientation_factor  # Store for reference

        # Generate hourly production shape (Gaussian centered at noon)
        hours = np.arange(24)
        daylight_mask = (hours >= 6) & (hours <= 18)
        x = hours[daylight_mask] - 12.0
        shape = np.exp(-(x ** 2) / (2 * 3.0 ** 2))
        shape /= shape.sum()
        self.hourly_shape = np.zeros(24)
        self.hourly_shape[daylight_mask] = shape

    def _interpolate_azimuth_factor(self, azimuth_degrees: float) -> float:
        """
        Linearly interpolate azimuth factor from empirical lookup table.

        Uses AZIMUTH_FACTORS constant with 8 cardinal/intercardinal directions
        to compute orientation derating. Handles angle wrapping at 360°/0°.

        Args:
            azimuth_degrees: Panel azimuth angle (0-360°).
                0° = north, 90° = east, 180° = south, 270° = west.

        Returns:
            float: Azimuth derating factor (0.68-1.0).
                1.0 = south (optimal), 0.68 = north (worst case).

        Example:
            ```python
            factor_south = model._interpolate_azimuth_factor(180.0)  # 1.00
            factor_se = model._interpolate_azimuth_factor(135.0)     # 0.93
            factor_ese = model._interpolate_azimuth_factor(112.5)    # ~0.885 (interpolated)
            factor_north = model._interpolate_azimuth_factor(0.0)    # 0.68
            ```

        Notes:
            - Linear interpolation between table points
            - Wraps correctly at 360°/0° boundary
            - Based on empirical Italian PV data (35-45°N latitude)
        """
        # Normalize azimuth to [0, 360)
        azimuth = azimuth_degrees % 360

        # Get sorted table angles
        angles = sorted(AZIMUTH_FACTORS.keys())

        # Handle wrapping case (azimuth > 315°)
        if azimuth > 315:
            lower_angle, upper_angle = 315, 360
            lower_factor = AZIMUTH_FACTORS[315]
            upper_factor = AZIMUTH_FACTORS[0]  # Wraps to north
            weight = (azimuth - lower_angle) / (upper_angle - lower_angle)
            return lower_factor + weight * (upper_factor - lower_factor)

        # Find bracketing angles in table
        for i, angle in enumerate(angles):
            if azimuth <= angle:
                if i == 0:
                    return AZIMUTH_FACTORS[angle]
                lower_angle = angles[i - 1]
                upper_angle = angle
                lower_factor = AZIMUTH_FACTORS[lower_angle]
                upper_factor = AZIMUTH_FACTORS[upper_angle]
                weight = (azimuth - lower_angle) / (upper_angle - lower_angle)
                return lower_factor + weight * (upper_factor - lower_factor)

        # Fallback (should not reach here)
        return AZIMUTH_FACTORS[315]

    def _compute_tilt_factor(self, panel_tilt: float) -> float:
        """
        Compute tilt deviation derating factor.

        Calculates production reduction when panel tilt differs from optimal.
        Tilt deviations are more forgiving than azimuth deviations due to
        wide solar elevation range throughout the day.

        Args:
            panel_tilt: Panel tilt from horizontal (0-90°).
                0° = flat/horizontal, 90° = vertical.

        Returns:
            float: Tilt derating factor (0.66-1.0).
                1.0 = at optimal tilt, decreases with deviation.

        Factor ranges by deviation:
            - ±0-15°: 0.98-1.00 (minimal impact)
            - ±15-30°: 0.86-0.98 (moderate impact)
            - ±30°+: 0.66-0.86 (significant impact)

        Example:
            ```python
            # Optimal case
            factor = model._compute_tilt_factor(35.0)  # 1.00 (if optimal=35°)

            # Slightly off
            factor = model._compute_tilt_factor(40.0)  # ~0.99 (5° deviation)

            # Moderately off
            factor = model._compute_tilt_factor(50.0)  # ~0.92 (15° deviation)

            # Significantly off
            factor = model._compute_tilt_factor(70.0)  # ~0.74 (35° deviation)
            ```

        Notes:
            - More forgiving than azimuth (wider acceptable range)
            - ±15° deviation loses only ~2% production
            - Flat roofs (10-20°) typically 90-98% of optimal
        """
        deviation = abs(panel_tilt - self.optimal_tilt_degrees)
        if deviation <= 15:
            return 1.0 - 0.02 * (deviation / 15)  # 98-100%
        elif deviation <= 30:
            return 0.98 - 0.12 * ((deviation - 15) / 15)  # 86-98%
        else:
            return 0.86 - 0.20 * min(1.0, (deviation - 30) / 30)  # 66-86%

    def compute_orientation_factor(
        self,
        panel_tilt_degrees: float,
        panel_azimuth_degrees: float,
    ) -> float:
        """
        Compute combined orientation derating factor using empirical data.

        Calculates production reduction when panel orientation (tilt + azimuth)
        differs from optimal values. Accounts for both direct radiation geometry
        and diffuse radiation that panels receive from all orientations.

        The combined factor is the product of azimuth and tilt derating factors,
        representing the overall reduction in energy capture due to non-optimal
        orientation.

        Args:
            panel_tilt_degrees: Panel tilt from horizontal (0-90°).
                0° = flat, 35° = typical optimal for Italy, 90° = vertical.
            panel_azimuth_degrees: Panel azimuth (0-360°).
                0° = north, 90° = east, 180° = south, 270° = west.

        Returns:
            float: Combined orientation derating factor (0.45-1.0).
                Multiply production values by this factor.
                1.0 = optimal orientation (typically 35° tilt, 180° azimuth).

        Example factors for typical installations:
            - South 35°: 1.00 (100% - optimal)
            - Southeast 35°: 0.93 (93% - excellent)
            - Southwest 35°: 0.93 (93% - excellent)
            - East 35°: 0.84 (84% - good)
            - West 35°: 0.84 (84% - good)
            - Northeast 35°: 0.78 (78% - acceptable)
            - North 35°: 0.68 (68% - poor but viable due to diffuse)
            - South 20° (flat): 0.98 (98% - very good)
            - East 50° (steep): 0.74 (74% - moderate)

        Example:
            ```python
            # Standard south-facing optimal installation
            factor = model.compute_orientation_factor(35.0, 180.0)  # 1.00

            # East-facing rooftop
            factor = model.compute_orientation_factor(35.0, 90.0)   # 0.84
            # Annual production: 6 kWp × 0.84 = 5.04 kWp effective

            # Flat roof south-facing
            factor = model.compute_orientation_factor(15.0, 180.0)  # 0.99
            # Only 1% loss despite 20° tilt deviation

            # North-facing (still produces from diffuse)
            factor = model.compute_orientation_factor(35.0, 0.0)    # 0.68
            # Still 68% production (diffuse + reflected light)
            ```

        Notes:
            - Based on empirical PV data for Italian latitudes (35-45°N)
            - Accounts for ~35% diffuse radiation in total irradiance
            - For other latitudes, adjust AZIMUTH_FACTORS if needed
            - Orientation losses are MULTIPLICATIVE (worst case: both factors low)
        """
        azimuth_factor = self._interpolate_azimuth_factor(panel_azimuth_degrees)
        tilt_factor = self._compute_tilt_factor(panel_tilt_degrees)
        return azimuth_factor * tilt_factor

    def simulate_daily_energy(
        self,
        n_years: int,
        month_in_year_for_day: np.ndarray,
        year_index_for_day: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate stochastic daily PV energy production over multiple years.

        Generates a complete time series of daily energy production values
        incorporating:
        1. Monthly seasonal variation (via month_params)
        2. Random weather (sunny vs cloudy days)
        3. Panel degradation over time
        4. System size scaling (via pv_kwp)

        Each day is independently classified as sunny or cloudy based on the
        month's p_sunny probability, with corresponding production multipliers
        applied. Degradation compounds annually, reducing output over time.

        Args:
            n_years: Number of simulation years (integer).
                Typically 20-25 years for PV system lifetime analysis.
                Used only for validation (actual simulation length is n_days).
            month_in_year_for_day: Array of month indices for each day (0-11).
                Shape: (n_days,). Maps each day to its month for parameter lookup.
                Example: [0, 0, ..., 0, 1, 1, ..., 11] for multi-year simulation.
                January=0, February=1, ..., December=11.
            year_index_for_day: Array of year indices for each day (0-based).
                Shape: (n_days,). Used for calculating degradation factor.
                Example: [0, 0, ..., 0, 1, 1, ..., 19] for 20-year simulation.
                Year 0 = first year, Year 1 = second year, etc.
            rng: NumPy random number generator.
                Used for stochastic sunny/cloudy classification.
                Create with: np.random.default_rng(seed=42) for reproducibility.

        Returns:
            np.ndarray: Daily PV energy production in kWh/day.
                Shape: (n_days,) where n_days = len(month_in_year_for_day).
                Values vary stochastically based on weather draws.
                Generally decreases over time due to degradation.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.solar import (
                SolarModel, make_default_solar_params_for_pavullo
            )

            # Setup
            params = make_default_solar_params_for_pavullo()
            solar = SolarModel(
                pv_kwp=5.0,
                month_params=params,
                degradation_per_year=0.007
            )

            # Simulate 20 years (7300 days)
            n_years = 20
            n_days = n_years * 365

            # Create day indexing arrays
            month_in_year = np.array([d // 30 % 12 for d in range(n_days)])
            year_index = np.array([d // 365 for d in range(n_days)])

            # Generate production
            rng = np.random.default_rng(seed=123)
            daily_production = solar.simulate_daily_energy(
                n_years, month_in_year, year_index, rng
            )

            # Analysis
            print(f"Year 1 total: {daily_production[:365].sum():.0f} kWh/year")
            print(f"Year 20 total: {daily_production[-365:].sum():.0f} kWh/year")
            print(f"Min daily: {daily_production.min():.1f} kWh (cloudy winter)")
            print(f"Max daily: {daily_production.max():.1f} kWh (sunny summer)")

            # Expected output for 5 kWp in Northern Italy:
            # Year 1 total: ~6250 kWh/year
            # Year 20 total: ~5440 kWh/year (13% degradation)
            # Min daily: ~0.4 kWh (cloudy December)
            # Max daily: ~33 kWh (sunny June)
            ```

        Notes:
            - Day-to-day weather is generated by a two-state Markov chain
              whose stationary distribution equals ``p_sunny`` for the
              current month (see :class:`SolarMonthParams`).
            - When ``weather_persistence == 0.0`` for all months the chain
              degenerates to independent Bernoulli draws (legacy behaviour).
            - The Markov state is reset from the stationary distribution at
              every month boundary so the marginal ``p_sunny`` is preserved
              even when transition probabilities change month-to-month.
            - Production = base × weather_factor × degradation_factor × pv_kwp
            - Weather factor is either ``sunny_factor`` or ``cloudy_factor``.
            - Degradation factor = (1 - degradation_per_year)^year_index
            - Return values represent total daily energy, not hourly breakdown
            - Use daily_profile_kwh() to split into hourly values
        """
        n_days = len(month_in_year_for_day)
        pv_daily = np.zeros(n_days, dtype=float)

        # Markov state for the weather (True = sunny, False = cloudy). The
        # state is carried from one day to the next and is re-initialised
        # from the stationary distribution every time we enter a new
        # (year, month) couple — this keeps the marginal P(sunny) equal to
        # the month's ``p_sunny`` even at month boundaries where the
        # transition probabilities change.
        prev_state_is_sunny: bool | None = None
        prev_year: int = -1
        prev_month: int = -1

        for i in range(n_days):
            m = int(month_in_year_for_day[i])
            y = int(year_index_for_day[i])
            params = self.month_params[m]

            base_daily = params.avg_daily_kwh_per_kwp * self.pv_kwp

            persistence = float(getattr(params, "weather_persistence", 0.0) or 0.0)
            persistence = min(1.0, max(0.0, persistence))

            crossed_month_boundary = (m != prev_month) or (y != prev_year)
            if prev_state_is_sunny is None or crossed_month_boundary:
                # First day overall, or first day of this month: draw from the
                # stationary distribution which by construction equals p_sunny.
                is_sunny = rng.random() < params.p_sunny
            else:
                # Transition from the previous state using the Markov chain
                # transition probabilities derived from p_sunny + persistence.
                if prev_state_is_sunny:
                    # P(sunny_t | sunny_{t-1}) = p_sunny + (1-p_sunny) * persistence
                    p_stay_sunny = params.p_sunny + (1.0 - params.p_sunny) * persistence
                    is_sunny = rng.random() < p_stay_sunny
                else:
                    # P(sunny_t | cloudy_{t-1}) = p_sunny * (1 - persistence)
                    p_jump_to_sunny = params.p_sunny * (1.0 - persistence)
                    is_sunny = rng.random() < p_jump_to_sunny

            factor = params.sunny_factor if is_sunny else params.cloudy_factor
            degradation_factor = (1.0 - self.degradation_per_year) ** y
            pv_daily[i] = base_daily * factor * degradation_factor

            prev_state_is_sunny = bool(is_sunny)
            prev_year = y
            prev_month = m

        return pv_daily

    def daily_profile_kwh(self, daily_energy_kwh: float) -> np.ndarray:
        """
        Distribute daily energy into 24 hourly values using fixed production shape.

        Converts a total daily energy value into an hourly time series by
        applying the pre-computed Gaussian hourly shape. This deterministic
        distribution assumes solar production follows a predictable daily
        pattern with peak at solar noon.

        The hourly shape was generated during __init__ as a Gaussian curve
        over daylight hours (6am-6pm), normalized to sum to 1.0. Nighttime
        hours (6pm-6am) have zero production.

        Args:
            daily_energy_kwh: Total energy for the day (kWh).
                This is typically a value from simulate_daily_energy().
                Must be non-negative. Negative values are not validated.

        Returns:
            np.ndarray: Hourly energy distribution (kWh/hour).
                Shape: (24,) representing hours 0-23 (midnight to 11pm).
                Values sum to daily_energy_kwh (within numerical precision).
                Nighttime hours (approx 0-5, 19-23) are near-zero.
                Peak production around hour 12 (noon).

        Example:
            ```python
            from sim_stochastic_pv.simulation.solar import (
                SolarModel, make_default_solar_params_for_pavullo
            )

            # Create model
            params = make_default_solar_params_for_pavullo()
            solar = SolarModel(pv_kwp=5.0, month_params=params)

            # Typical sunny summer day: 30 kWh total
            hourly = solar.daily_profile_kwh(30.0)

            print(f"Shape: {hourly.shape}")  # (24,)
            print(f"Sum: {hourly.sum():.1f} kWh")  # 30.0 kWh
            print(f"Peak hour: {hourly.argmax()}")  # ~12 (noon)
            print(f"Peak value: {hourly.max():.2f} kWh/h")  # ~4-5 kWh/h
            print(f"Night value: {hourly[0]:.4f} kWh/h")  # ~0.0000 kWh/h

            # Typical hourly pattern:
            # 00:00-05:00: ~0.0 kWh/h (night)
            # 06:00-08:00: 0.5-2.0 kWh/h (sunrise)
            # 09:00-15:00: 3.0-5.0 kWh/h (midday peak)
            # 16:00-18:00: 1.0-3.0 kWh/h (sunset)
            # 19:00-23:00: ~0.0 kWh/h (night)
            ```

        Notes:
            - Hourly shape is the same for all days (deterministic)
            - Day-to-day variation comes from daily_energy_kwh input
            - Gaussian shape approximates real solar production curves
            - Does not account for time-of-day weather (e.g., morning fog)
            - Suitable for economic analysis where hourly detail matters
            - For battery sizing, hourly resolution is essential
        """
        return daily_energy_kwh * self.hourly_shape


def make_default_solar_params_for_pavullo() -> List[SolarMonthParams]:
    """
    Create default solar production parameters calibrated for Northern Italy.

    Returns pre-configured monthly parameters representing typical photovoltaic
    production characteristics for Pavullo nel Frignano (Northern Italy, ~44.3°N).
    These parameters approximate ~1250 kWh/kWp/year annual production with
    realistic seasonal variation and weather patterns.

    The parameters are based on historical weather data and solar irradiance
    for the region, suitable for residential PV system analysis in similar
    climates (Northern Mediterranean, mid-latitude, moderate continental).

    Returns:
        List[SolarMonthParams]: List of 12 monthly parameter objects.
            Index 0 = January, Index 11 = December.
            Each month configured with:
            - avg_daily_kwh_per_kwp: Baseline daily production (kWh/kWp/day)
            - p_sunny: Probability of sunny day (higher in summer)
            - sunny_factor: 1.2 (20% above average on sunny days)
            - cloudy_factor: 0.3 (70% below average on cloudy days)

    Example:
        ```python
        from sim_stochastic_pv.simulation.solar import (
            SolarModel, make_default_solar_params_for_pavullo
        )

        # Use default parameters for quick setup
        params = make_default_solar_params_for_pavullo()
        solar = SolarModel(pv_kwp=6.0, month_params=params)

        # Annual production estimate
        # Nominal: 6.0 kWp × 1250 kWh/kWp/year = 7500 kWh/year
        # Actual varies stochastically ±10-15% depending on weather

        # Inspect specific month
        print(f"June avg: {params[5].avg_daily_kwh_per_kwp:.2f} kWh/kWp/day")
        # Output: June avg: 5.46 kWh/kWp/day
        print(f"June sunny prob: {params[5].p_sunny:.1%}")
        # Output: June sunny prob: 70.0%

        # December (winter)
        print(f"Dec avg: {params[11].avg_daily_kwh_per_kwp:.2f} kWh/kWp/day")
        # Output: Dec avg: 1.36 kWh/kWp/day (4× less than June)
        ```

    Monthly Breakdown:
        Month | Daily Avg (kWh/kWp/day) | P(sunny) | Monthly Total (kWh/kWp)
        ------|-------------------------|----------|-------------------------
        Jan   | 1.46                    | 0.40     | ~45
        Feb   | 2.27                    | 0.45     | ~64
        Mar   | 3.18                    | 0.50     | ~99
        Apr   | 4.09                    | 0.55     | ~123
        May   | 5.00                    | 0.60     | ~155
        Jun   | 5.46                    | 0.70     | ~164
        Jul   | 5.46                    | 0.70     | ~169
        Aug   | 4.55                    | 0.65     | ~141
        Sep   | 3.64                    | 0.55     | ~109
        Oct   | 2.73                    | 0.50     | ~85
        Nov   | 1.82                    | 0.45     | ~55
        Dec   | 1.36                    | 0.40     | ~42
        ------|-------------------------|----------|-------------------------
        Total |                         |          | ~1251 kWh/kWp/year

    Notes:
        - Based on Pavullo nel Frignano (44.3°N, 10.8°E, ~700m elevation)
        - Suitable for similar climates: Northern Italy, Southern France, etc.
        - Accounts for alpine/Apennine climate (cooler, cloudier than coast)
        - Assumes optimal panel orientation (South-facing, ~35° tilt)
        - Does not account for shading, soiling, or inverter losses
        - For other locations, create custom parameters or use PVGIS data
        - Weather factors (sunny/cloudy) calibrated to regional statistics
    """
    avg_daily_kwh_per_kwp = [
        1.46,  # Jan
        2.27,  # Feb
        3.18,  # Mar
        4.09,  # Apr
        5.00,  # May
        5.46,  # Jun
        5.46,  # Jul
        4.55,  # Aug
        3.64,  # Sep
        2.73,  # Oct
        1.82,  # Nov
        1.36,  # Dec
    ]

    p_sunny = [
        0.4, 0.45, 0.5, 0.55, 0.6, 0.7,
        0.7, 0.65, 0.55, 0.5, 0.45, 0.4,
    ]

    params = []
    for i in range(12):
        params.append(
            SolarMonthParams(
                avg_daily_kwh_per_kwp=avg_daily_kwh_per_kwp[i],
                p_sunny=p_sunny[i],
                sunny_factor=1.2,
                cloudy_factor=0.3,
            )
        )
    return params
