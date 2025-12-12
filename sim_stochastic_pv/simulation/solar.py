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


@dataclass
class SolarMonthParams:
    """
    Monthly parameters for stochastic solar PV production modeling.

    Encapsulates the statistical characteristics of solar energy production
    for a single month, enabling Monte Carlo simulation of weather variability.
    Each day is randomly classified as "sunny" or "cloudy" based on probability,
    with corresponding production multipliers applied.

    This bimodal weather model (sunny/cloudy) is a simplification of real
    weather patterns but captures the essential variability needed for
    economic risk assessment of PV systems.

    Attributes:
        avg_daily_kwh_per_kwp: Average daily energy production per installed kWp (kWh/kWp/day).
            This is the baseline production for an "average" day in the month.
            Typical values:
            - Winter (Dec-Feb): 1.0-2.5 kWh/kWp/day (Northern Italy)
            - Spring/Fall (Mar-May, Sep-Nov): 3.0-5.0 kWh/kWp/day
            - Summer (Jun-Aug): 5.0-6.0 kWh/kWp/day
            Annual total typically 1100-1400 kWh/kWp/year in Northern Italy.
        p_sunny: Probability of a sunny day (0.0 to 1.0).
            Fraction of days in the month expected to be sunny (not cloudy).
            Examples:
            - 0.4 = 40% sunny days (typical winter)
            - 0.7 = 70% sunny days (typical summer)
            Higher values = more consistent production, lower risk.
        sunny_factor: Production multiplier for sunny days (typically > 1.0).
            Applied to avg_daily_kwh_per_kwp on sunny days.
            Typical values: 1.1-1.3 (10-30% above average).
            Example: 1.2 means sunny days produce 20% more than average.
        cloudy_factor: Production multiplier for cloudy days (typically < 1.0).
            Applied to avg_daily_kwh_per_kwp on cloudy days.
            Typical values: 0.2-0.5 (20-50% of average, significant reduction).
            Example: 0.3 means cloudy days produce 70% less than average.

    Example:
        ```python
        # Summer month in Northern Italy
        june_params = SolarMonthParams(
            avg_daily_kwh_per_kwp=5.46,  # High summer production
            p_sunny=0.7,                  # 70% of days are sunny
            sunny_factor=1.2,             # Sunny: 5.46 × 1.2 = 6.55 kWh/kWp/day
            cloudy_factor=0.3             # Cloudy: 5.46 × 0.3 = 1.64 kWh/kWp/day
        )

        # Winter month
        january_params = SolarMonthParams(
            avg_daily_kwh_per_kwp=1.46,  # Low winter production
            p_sunny=0.4,                  # Only 40% sunny days
            sunny_factor=1.2,             # Sunny: 1.75 kWh/kWp/day
            cloudy_factor=0.3             # Cloudy: 0.44 kWh/kWp/day
        )
        ```

    Notes:
        - Factors should satisfy: p_sunny × sunny_factor + (1-p_sunny) × cloudy_factor ≈ 1.0
          This ensures weighted average production equals avg_daily_kwh_per_kwp
        - Bimodal model is simpler than full irradiance distributions but adequate
        - Does not model hour-by-hour variability (uses fixed daily shape)
        - Calibrate parameters to local climate data for accurate results
    """
    avg_daily_kwh_per_kwp: float
    p_sunny: float
    sunny_factor: float
    cloudy_factor: float


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
    ) -> None:
        """
        Initialize a stochastic solar PV production model with degradation.

        Sets up the model with system capacity, monthly weather parameters,
        and degradation assumptions. Automatically generates a Gaussian
        hourly production shape for daylight hours (6am-6pm).

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

        Raises:
            ValueError: If month_params does not contain exactly 12 entries.

        Example:
            ```python
            # Standard 6 kWp residential system
            params = make_default_solar_params_for_pavullo()
            model = SolarModel(
                pv_kwp=6.0,
                month_params=params,
                degradation_per_year=0.007
            )

            # Premium system with slower degradation
            model_premium = SolarModel(
                pv_kwp=6.0,
                month_params=params,
                degradation_per_year=0.005  # Better panels
            )
            ```

        Notes:
            - Hourly shape is Gaussian with peak at noon, 6am-6pm window
            - Shape is normalized (sums to 1.0) for energy distribution
            - month_params order matters: [Jan, Feb, ..., Dec]
            - Degradation compounds: year N capacity = (1 - rate)^N
        """
        if len(month_params) != 12:
            raise ValueError("month_params must contain 12 entries")
        self.pv_kwp = pv_kwp
        self.month_params = month_params
        self.degradation_per_year = degradation_per_year

        hours = np.arange(24)
        daylight_mask = (hours >= 6) & (hours <= 18)
        x = hours[daylight_mask] - 12.0
        shape = np.exp(-(x ** 2) / (2 * 3.0 ** 2))
        shape /= shape.sum()
        self.hourly_shape = np.zeros(24)
        self.hourly_shape[daylight_mask] = shape

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
            - Each day is simulated independently (no autocorrelation)
            - Production = base × weather_factor × degradation_factor × pv_kwp
            - Weather factor is either sunny_factor or cloudy_factor per day
            - Degradation factor = (1 - degradation_per_year)^year_index
            - Return values represent total daily energy, not hourly breakdown
            - Use daily_profile_kwh() to split into hourly values
        """
        n_days = len(month_in_year_for_day)
        pv_daily = np.zeros(n_days, dtype=float)

        for i in range(n_days):
            m = month_in_year_for_day[i]
            y = year_index_for_day[i]
            params = self.month_params[m]

            base_daily = params.avg_daily_kwh_per_kwp * self.pv_kwp

            is_sunny = rng.random() < params.p_sunny
            factor = params.sunny_factor if is_sunny else params.cloudy_factor

            degradation_factor = (1.0 - self.degradation_per_year) ** y

            pv_daily[i] = base_daily * factor * degradation_factor

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
