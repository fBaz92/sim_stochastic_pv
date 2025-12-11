from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .calendar_utils import MONTH_LENGTHS

# Ogni riga = mese (0=Gennaio..11=Dicembre)
# Colonne: [BL_F1, BL_F2, BL_F3] in Watt
# ARERA tariff bands base load table: array of shape (12, 3) containing base load
# values in Watt for each month (rows) and ARERA tariff band (columns: F1, F2, F3).
BL_TABLE: np.ndarray = np.array([
    [110.6719368,  83.79888268, 157.0512821 ],  # Gen
    [ 68.18181818, 73.17073171,  72.91666667],  # Feb
    [ 99.56709957,113.5135135,   94.51219512],  # Mar
    [144.6280992, 120.6896552,  138.1578947 ],  # Apr
    [ 90.90909091, 89.47368421, 102.5641026 ],  # Mag
    [116.8831169, 130.1775148,  134.375     ],  # Giu
    [130.4347826, 145.2513966,  125.        ],  # Lug
    [251.0822511, 286.4864865,  280.4878049],  # Ago
    [ 95.04132231, 97.70114943,  92.10526316],  # Set
    [169.9604743, 111.7318436, 118.5897436 ],  # Ott
    [131.8181818,  83.33333333, 168.75     ],  # Nov
    [185.770751,  229.0502793,  211.5384615],  # Dic
])


class LoadProfile:
    """Generic interface for hourly load models."""

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """Hook called at the start of each Monte Carlo path (default no-op)."""

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load consumption.
        
        Args:
            year_index: Year index (0-based).
            month_in_year: Month index (0-11).
            day_in_month: Day index within month (0-based).
            hour_in_day: Hour index (0-23).
            weekday: Weekday index (0=Monday, 6=Sunday).
        
        Returns:
            Load consumption in kW.
        """
        raise NotImplementedError


class MonthlyAverageLoadProfile(LoadProfile):
    """
    Monthly hourly load profiles (deterministic 24h pattern repeated every day).
    """

    def __init__(self, monthly_profiles_w: np.ndarray) -> None:
        """
        Initialize monthly average load profile.
        
        Args:
            monthly_profiles_w: Array of shape (12, 24) with hourly load in Watt
                for each month (rows) and hour (columns).
        """
        if monthly_profiles_w.shape != (12, 24):
            raise ValueError("monthly_profiles_w must have shape (12, 24)")
        self.monthly_profiles_kw = monthly_profiles_w / 1000.0

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load from monthly average profile.
        
        Returns the same value for all days in a month at the same hour.
        """
        return float(self.monthly_profiles_kw[month_in_year, hour_in_day])


class AreraLoadProfile(LoadProfile):
    """Load model that applies ARERA bands (F1/F2/F3) using BL table inputs."""

    def __init__(self, bl_table: np.ndarray = BL_TABLE) -> None:
        """
        Initialize ARERA load profile.
        
        Args:
            bl_table: Array of shape (12, 3) with base load values in Watt
                for each month (rows) and ARERA band (columns: F1, F2, F3).
        """
        if bl_table.shape != (12, 3):
            raise ValueError("bl_table must have shape (12, 3)")
        self.bl_table = bl_table

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load based on ARERA tariff bands.
        
        Determines the appropriate tariff band (F1/F2/F3) based on weekday
        and hour, then returns the corresponding base load value.
        """
        hour_in_month = day_in_month * 24 + hour_in_day
        first_weekday = (weekday - day_in_month) % 7
        load_w = get_load_w(
            month_index=month_in_year,
            hour_in_month=hour_in_month,
            first_weekday_of_month=first_weekday,
            bl_table=self.bl_table,
        )
        return load_w / 1000.0


class HomeAwayLoadProfile(LoadProfile):
    """Combines two load profiles with stochastic presence days for the home profile."""

    def __init__(
        self,
        home_profile: LoadProfile,
        away_profile: LoadProfile,
        min_days_home: List[int],
        max_days_home: List[int],
    ) -> None:
        """
        Initialize home/away load profile with stochastic presence.
        
        Args:
            home_profile: Load profile to use when at home.
            away_profile: Load profile to use when away.
            min_days_home: Minimum days at home per month (12 values, one per month).
            max_days_home: Maximum days at home per month (12 values, one per month).
        """
        if len(min_days_home) != 12 or len(max_days_home) != 12:
            raise ValueError("min_days_home and max_days_home must have length 12")

        self.home_profile = home_profile
        self.away_profile = away_profile
        self.min_days_home = np.array(min_days_home, dtype=int)
        self.max_days_home = np.array(max_days_home, dtype=int)

        self._home_days_schedule: Dict[Tuple[int, int], np.ndarray] = {}
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Reset for a new Monte Carlo simulation path.
        
        Clears the cached home days schedule and resets both profiles.
        """
        self._home_days_schedule.clear()
        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng
        self.home_profile.reset_for_run(rng=rng, n_years=n_years)
        self.away_profile.reset_for_run(rng=rng, n_years=n_years)

    def _rng_for_month(self) -> np.random.Generator:
        """
        Get random number generator for stochastic operations.
        
        Returns:
            Random number generator instance.
        """
        if self._rng is None:
            self._rng = self._fallback_rng
        return self._rng

    def _get_month_schedule(
        self,
        year_index: int,
        month_in_year: int,
    ) -> np.ndarray:
        """
        Get or generate the home days schedule for a specific month.
        
        Args:
            year_index: Year index.
            month_in_year: Month index (0-11).
        
        Returns:
            Boolean array indicating which days are home days (True) vs away days (False).
        """
        key = (year_index, month_in_year)
        schedule = self._home_days_schedule.get(key)
        if schedule is not None:
            return schedule

        days_in_month = MONTH_LENGTHS[month_in_year]
        min_days = int(np.clip(self.min_days_home[month_in_year], 0, days_in_month))
        max_days = int(np.clip(self.max_days_home[month_in_year], 0, days_in_month))
        if max_days < min_days:
            max_days = min_days

        rng = self._rng_for_month()
        if max_days == min_days:
            n_home_days = min_days
        else:
            n_home_days = int(rng.integers(min_days, max_days + 1))

        mask = np.zeros(days_in_month, dtype=bool)
        if n_home_days > 0:
            day_indices = rng.choice(days_in_month, size=n_home_days, replace=False)
            mask[day_indices] = True

        self._home_days_schedule[key] = mask
        return mask

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load based on stochastic home/away schedule.
        
        Returns home profile load if the day is scheduled as a home day,
        otherwise returns away profile load.
        """
        schedule = self._get_month_schedule(year_index, month_in_year)
        if 0 <= day_in_month < schedule.size and schedule[day_in_month]:
            return self.home_profile.get_hourly_load_kw(
                year_index=year_index,
                month_in_year=month_in_year,
                day_in_month=day_in_month,
                hour_in_day=hour_in_day,
                weekday=weekday,
            )

        return self.away_profile.get_hourly_load_kw(
            year_index=year_index,
            month_in_year=month_in_year,
            day_in_month=day_in_month,
            hour_in_day=hour_in_day,
            weekday=weekday,
        )


def get_load_w(
    month_index: int,
    hour_in_month: int,
    first_weekday_of_month: int = 0,
    bl_table: np.ndarray = BL_TABLE,
) -> float:
    """
    Restituisce il carico in Watt per una certa ora del mese, usando le fasce ARERA.
    """
    if not (0 <= month_index < 12):
        raise ValueError("month_index deve essere tra 0 e 11")

    days_in_month = MONTH_LENGTHS[month_index]
    max_hours = 24 * days_in_month
    if not (0 <= hour_in_month < max_hours):
        raise ValueError(
            f"hour_in_month deve essere tra 0 e {max_hours-1} per il mese {month_index}"
        )

    day_in_month = hour_in_month // 24
    hour = hour_in_month % 24
    weekday = (first_weekday_of_month + day_in_month) % 7
    band = _get_band_arera(weekday, hour)

    if band == "F1":
        bl = bl_table[month_index, 0]
    elif band == "F2":
        bl = bl_table[month_index, 1]
    else:
        bl = bl_table[month_index, 2]

    return bl


def _get_band_arera(weekday: int, hour: int) -> str:
    """
    Determine ARERA tariff band (F1/F2/F3) based on weekday and hour.
    
    Args:
        weekday: Weekday index (0=Monday, 6=Sunday).
        hour: Hour of day (0-23).
    
    Returns:
        Tariff band string: "F1", "F2", or "F3".
    """
    if weekday == 6:
        return "F3"

    if weekday in range(0, 5):
        if 8 <= hour <= 18:
            return "F1"
        elif hour == 7 or (19 <= hour <= 22):
            return "F2"
        else:
            return "F3"
    else:
        if 7 <= hour <= 22:
            return "F2"
        else:
            return "F3"


def make_flat_monthly_load_profiles(base_load_w: float = 140.0) -> np.ndarray:
    """
    Returns a (12,24) array with constant load base_load_w for all months/hours.
    """
    return np.full((12, 24), base_load_w, dtype=float)


@dataclass
class LoadScenarioBlueprint:
    """
    Helper config to assemble load profile objects.

    - If both factories are provided, returns a HomeAwayLoadProfile (requires min/max day arrays).
    - If only one factory is provided, returns that profile directly.
    """

    home_profile_factory: Callable[[], LoadProfile] | None = None
    away_profile_factory: Callable[[], LoadProfile] | None = None
    min_days_home: List[int] | None = None
    max_days_home: List[int] | None = None

    def build_load_profile(self) -> LoadProfile:
        """
        Build a load profile based on configured factories.
        
        Returns:
            LoadProfile instance. If both factories are provided, returns
            a HomeAwayLoadProfile. Otherwise returns the single profile.
        
        Raises:
            ValueError: If configuration is invalid or no factory is provided.
        """
        if self.home_profile_factory and self.away_profile_factory:
            if self.min_days_home is None or self.max_days_home is None:
                raise ValueError("min_days_home and max_days_home are required for home/away scenarios")
            home_profile = self.home_profile_factory()
            away_profile = self.away_profile_factory()
            return HomeAwayLoadProfile(
                home_profile=home_profile,
                away_profile=away_profile,
                min_days_home=self.min_days_home,
                max_days_home=self.max_days_home,
            )

        if self.home_profile_factory:
            return self.home_profile_factory()

        if self.away_profile_factory:
            return self.away_profile_factory()

        raise ValueError("At least one profile factory must be provided")
