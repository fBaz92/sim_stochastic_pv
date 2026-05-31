"""
Fuel prices, availability models, and generator classes.

This module contains three layers of abstraction:

1. **Price models** (:class:`FuelPriceModel`, :class:`ConstantFuelPrice`,
   :class:`CarbonPriceModel`) — generate stochastic or constant price paths
   for fuel and CO2 over a simulated year.

2. **Availability models** (:class:`DispatchableAvailability`,
   :class:`MustRunAvailability`, :class:`SolarAvailability`,
   :class:`WindAvailability`) — generate capacity-factor profiles that
   determine how much of a generator's installed capacity is available at
   each quarter-hour.

3. **Generator** (:class:`Generator`) and factory (:func:`build_generators`)
   — combine price and availability models with cost/technical parameters
   into dispatchable units.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from sim_stochastic_pv.market.config import (
    QUARTERS_PER_YEAR, QUARTERS_PER_HOUR, QUARTERS_PER_DAY,
    CO2_PRICE_DEFAULT, DISCOUNT_RATE, P_PEAK_GW,
    MONTHLY_SOLAR_FACTORS, HOURLY_SOLAR_ENVELOPE,
    MONTHLY_WIND_LAMBDA, WIND_WEIBULL_K,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT,
    CLOUD_TRANSITION, COAL_SCENARIOS, CO2_SCENARIOS,
)
from sim_stochastic_pv.market.grid import TimeGrid


# ── Price models ──────────────────────────────────────────────────────────


class FuelPriceModel:
    """Ornstein-Uhlenbeck mean-reverting fuel price process.

    Models fuel prices (e.g. natural gas) as a continuous mean-reverting
    stochastic process discretized at quarter-hour resolution.

    Attributes:
        mu (float): Long-run mean price (EUR/MWh_th).
        sigma (float): Volatility parameter.
        theta (float): Mean-reversion speed (higher = faster reversion).
    """

    def __init__(self, mu: float, sigma: float, theta: float = 0.1) -> None:
        """Initialize the O-U fuel price model.

        Args:
            mu: Long-run mean price in EUR/MWh_th.
            sigma: Volatility of the price process.
            theta: Mean-reversion speed. Defaults to 0.1.
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def generate_path(self, n_steps: int, rng: np.random.Generator,
                      dt: float = 0.25 / 24 / 365) -> np.ndarray:
        """Generate a fuel price path for one simulated year.

        Uses Euler-Maruyama discretization of the O-U SDE:
        ``dP = theta * (mu - P) * dt + sigma * dW``.

        Shocks are drawn iid from the standard normal distribution. For
        correlated multi-area simulations, use :meth:`generate_path_from_shocks`
        directly with externally-supplied correlated shocks.

        Args:
            n_steps: Number of time steps (typically 35 040).
            rng: NumPy random generator instance.
            dt: Time step in years. Defaults to one quarter-hour
                (0.25 / 24 / 365).

        Returns:
            np.ndarray: Price path of shape ``(n_steps,)`` in EUR/MWh_th,
                floored at 1.0 EUR.
        """
        shocks = rng.standard_normal(n_steps)
        return self.generate_path_from_shocks(shocks, dt=dt)

    def generate_path_from_shocks(self, shocks: np.ndarray,
                                   dt: float = 0.25 / 24 / 365) -> np.ndarray:
        """Integrate the O-U SDE driven by externally-supplied Gaussian shocks.

        Decouples the stochastic source from the numerical integrator, which
        allows the same O-U path machinery to be reused for correlated
        multi-area simulations (see
        :class:`~sim_stochastic_pv.market.price_areas.PriceAreaCoupling`). The iid path
        produced by :meth:`generate_path` is mathematically identical to a
        call to this method with ``shocks = rng.standard_normal(n_steps)``.

        Args:
            shocks: Array of Gaussian shocks of shape ``(n_steps,)``. Must
                be marginally standard normal. Correlation with other
                processes (if any) lives in the cross-process covariance.
            dt: Time step in years. Defaults to one quarter-hour
                (0.25 / 24 / 365).

        Returns:
            np.ndarray: Price path of shape ``(n_steps,)`` in EUR/MWh_th,
                floored at 1.0 EUR. First element is ``self.mu``.
        """
        n_steps = shocks.shape[0]
        prices = np.empty(n_steps)
        prices[0] = self.mu
        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_steps):
            dp = (self.theta * (self.mu - prices[t - 1]) * dt
                  + self.sigma * sqrt_dt * shocks[t])
            prices[t] = max(prices[t - 1] + dp, 1.0)
        return prices


class ConstantFuelPrice:
    """Constant fuel price model (e.g. uranium).

    Attributes:
        price (float): Fixed fuel price in EUR/MWh_th.
    """

    def __init__(self, price: float) -> None:
        """Initialize with a fixed price.

        Args:
            price: Constant fuel price in EUR/MWh_th.
        """
        self.price = price

    def generate_path(self, n_steps: int, rng: np.random.Generator = None,
                      dt: float = None) -> np.ndarray:
        """Generate a constant price path.

        Args:
            n_steps: Number of time steps.
            rng: Unused. Accepted for interface compatibility.
            dt: Unused. Accepted for interface compatibility.

        Returns:
            np.ndarray: Constant array of shape ``(n_steps,)`` filled with
                ``self.price``.
        """
        return np.full(n_steps, self.price)


class CarbonPriceModel:
    """CO2 ETS price model using Ornstein-Uhlenbeck mean-reverting process.

    Models EU ETS carbon prices as a continuous mean-reverting stochastic
    process, with slower mean-reversion than fuel prices (reflecting ETS
    market structural inertia). A volatile CO₂ price produces realistic
    fuel-switching dynamics between coal and gas.

    Attributes:
        mu (float): Long-run mean CO2 price (EUR/ton).
        sigma (float): Volatility parameter.
        theta (float): Mean-reversion speed (lower = slower reversion).
    """

    def __init__(self, mu: float = CO2_PRICE_DEFAULT,
                 sigma: float = 10.0, theta: float = 0.05) -> None:
        """Initialize the carbon price model.

        Args:
            mu: Long-run mean CO2 price in EUR/ton.
                Defaults to ``CO2_PRICE_DEFAULT`` (65).
            sigma: Volatility of the price process. Defaults to 10.0.
            theta: Mean-reversion speed. Defaults to 0.05 (slower than
                fuel prices, reflecting ETS market inertia).
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def generate_path(self, n_steps: int, rng: np.random.Generator,
                      dt: float = 0.25 / 24 / 365) -> np.ndarray:
        """Generate a stochastic CO2 price path for one simulated year.

        Uses Euler-Maruyama discretization of the O-U SDE:
        ``dP = theta * (mu - P) * dt + sigma * dW``.

        Args:
            n_steps: Number of time steps (typically 35 040).
            rng: NumPy random generator instance.
            dt: Time step in years. Defaults to one quarter-hour
                (0.25 / 24 / 365).

        Returns:
            np.ndarray: CO2 price path of shape ``(n_steps,)`` in EUR/ton,
                floored at 1.0 EUR.
        """
        prices = np.empty(n_steps)
        prices[0] = self.mu
        sqrt_dt = np.sqrt(dt)
        noise = rng.standard_normal(n_steps)
        for t in range(1, n_steps):
            dp = self.theta * (self.mu - prices[t - 1]) * dt + self.sigma * sqrt_dt * noise[t]
            prices[t] = max(prices[t - 1] + dp, 1.0)
        return prices


# ── Availability models ──────────────────────────────────────────────────


class DispatchableAvailability:
    """Always-available model for dispatchable generators (gas, nuclear).

    Returns a flat capacity factor of 1.0 at every time step.
    """

    def generate_profile(self, time_grid: TimeGrid,
                         rng: np.random.Generator = None) -> np.ndarray:
        """Generate a flat availability profile.

        Args:
            time_grid: Temporal backbone (used only for its length).
            rng: Unused. Accepted for interface compatibility.

        Returns:
            np.ndarray: Ones array of shape ``(time_grid.n,)``.
        """
        return np.ones(time_grid.n)


class MustRunAvailability:
    """Constant must-run availability model (e.g. hydro base, nuclear high-CF).

    Attributes:
        cf (float): Constant capacity factor (0.0 to 1.0).
    """

    def __init__(self, cf: float = 1.0) -> None:
        """Initialize with a fixed capacity factor.

        Args:
            cf: Capacity factor applied at every time step. Defaults to 1.0.
        """
        self.cf = cf

    def generate_profile(self, time_grid: TimeGrid,
                         rng: np.random.Generator = None) -> np.ndarray:
        """Generate a constant availability profile.

        Args:
            time_grid: Temporal backbone (used only for its length).
            rng: Unused. Accepted for interface compatibility.

        Returns:
            np.ndarray: Constant array of shape ``(time_grid.n,)`` filled
                with ``self.cf``.
        """
        return np.full(time_grid.n, self.cf)


class SolarAvailability:
    """Solar availability with monthly/hourly envelope and Markov cloud model.

    Combines a deterministic irradiance envelope (month x hour Gaussian) with
    a stochastic two-state daily Markov chain for cloud cover. Night hours
    are hard-zeroed via the envelope.
    """

    def generate_profile(self, time_grid: TimeGrid,
                         rng: np.random.Generator) -> np.ndarray:
        """Generate a stochastic solar availability profile for one year.

        Steps:
        1. Build deterministic envelope from monthly and hourly solar factors.
        2. Simulate a two-state (sunny/cloudy) daily Markov chain.
        3. Apply cloud attenuation factor to the envelope.

        Args:
            time_grid: Temporal backbone providing month and hour arrays.
            rng: NumPy random generator for cloud state simulation.

        Returns:
            np.ndarray: Solar capacity factor profile of shape ``(time_grid.n,)``
                with values in [0.0, 1.0]. Night hours are 0.0.
        """
        n = time_grid.n

        # Deterministic envelope
        k_month = np.array([MONTHLY_SOLAR_FACTORS.get(m, 0.5) for m in time_grid.month])
        k_hour = np.array([HOURLY_SOLAR_ENVELOPE.get(h, 0.0) for h in time_grid.hour])
        envelope = k_month * k_hour

        # Cloud Markov chain (state per day)
        n_days = 365
        cloudy = np.zeros(n_days, dtype=bool)
        cloudy[0] = rng.random() < 0.4
        for d in range(1, n_days):
            month = time_grid.month[d * QUARTERS_PER_DAY]
            p_s2c, p_c2s = CLOUD_TRANSITION.get(month, (0.25, 0.45))
            if cloudy[d - 1]:
                cloudy[d] = rng.random() > p_c2s
            else:
                cloudy[d] = rng.random() < p_s2c

        # Expand to quarter-hour resolution
        cloudy_qh = np.repeat(cloudy, QUARTERS_PER_DAY)[:n]

        # Cloud attenuation factor
        k_cloud = np.where(
            cloudy_qh,
            rng.uniform(0.15, 0.40, size=n),
            rng.uniform(0.85, 1.00, size=n),
        )

        profile = envelope * k_cloud
        return profile


class WindAvailability:
    """Wind availability using Weibull distribution with AR(1) autocorrelation.

    Models wind speed as an AR(1) process in Gaussian space, then transforms
    to Weibull-distributed wind speeds with month-dependent scale parameter,
    and finally applies a cubic turbine power curve.

    Attributes:
        k (float): Weibull shape parameter.
        cut_in (float): Turbine cut-in speed (m/s).
        rated (float): Turbine rated speed (m/s).
        cut_out (float): Turbine cut-out speed (m/s).
    """

    def __init__(self, weibull_k: float = WIND_WEIBULL_K,
                 cut_in: float = WIND_CUT_IN,
                 rated: float = WIND_RATED,
                 cut_out: float = WIND_CUT_OUT) -> None:
        """Initialize the wind availability model.

        Args:
            weibull_k: Weibull shape parameter. Defaults to ``WIND_WEIBULL_K``.
            cut_in: Turbine cut-in wind speed in m/s. Defaults to ``WIND_CUT_IN``.
            rated: Turbine rated wind speed in m/s. Defaults to ``WIND_RATED``.
            cut_out: Turbine cut-out wind speed in m/s. Defaults to ``WIND_CUT_OUT``.
        """
        self.k = weibull_k
        self.cut_in = cut_in
        self.rated = rated
        self.cut_out = cut_out

    def _power_curve(self, v: np.ndarray) -> np.ndarray:
        """Compute normalized power output from wind speed.

        Applies a cubic power curve between cut-in and rated speed,
        full power between rated and cut-out, and zero outside.

        Args:
            v: Wind speed array in m/s, any shape.

        Returns:
            np.ndarray: Normalized power output in [0.0, 1.0], same shape as ``v``.
        """
        p = np.zeros_like(v)
        mask_partial = (v >= self.cut_in) & (v < self.rated)
        p[mask_partial] = ((v[mask_partial] - self.cut_in) / (self.rated - self.cut_in)) ** 3
        mask_full = (v >= self.rated) & (v <= self.cut_out)
        p[mask_full] = 1.0
        return p

    def generate_profile(self, time_grid: TimeGrid,
                         rng: np.random.Generator) -> np.ndarray:
        """Generate a stochastic wind availability profile for one year.

        Steps:
        1. Simulate AR(1) process in Gaussian space (high autocorrelation).
        2. Transform to uniform via Gaussian CDF.
        3. Apply month-dependent inverse Weibull CDF to get wind speeds.
        4. Apply turbine power curve to get capacity factors.

        Args:
            time_grid: Temporal backbone providing month array for
                seasonally-varying Weibull scale.
            rng: NumPy random generator for the AR(1) noise.

        Returns:
            np.ndarray: Wind capacity factor profile of shape ``(time_grid.n,)``
                with values in [0.0, 1.0].
        """
        n = time_grid.n

        # AR(1) in Gaussian space
        ar_coeff = 0.995
        z = np.empty(n)
        z[0] = rng.standard_normal()
        noise = rng.standard_normal(n)
        for t in range(1, n):
            z[t] = ar_coeff * z[t - 1] + np.sqrt(1 - ar_coeff ** 2) * noise[t]

        # Transform to uniform via Gaussian CDF
        u = norm.cdf(z)
        u = np.clip(u, 1e-6, 1 - 1e-6)

        # Month-dependent Weibull scale
        lambdas = np.array([MONTHLY_WIND_LAMBDA.get(m, 7.0) for m in time_grid.month])

        # Inverse Weibull CDF: v = lambda * (-ln(1-u))^(1/k)
        v = lambdas * (-np.log(1 - u)) ** (1.0 / self.k)

        profile = self._power_curve(v)
        return profile


# ── Generator ─────────────────────────────────────────────────────────────


class Generator:
    """A generation unit with cost model, fuel/CO2 pricing, and availability.

    Combines technical parameters (capacity, efficiency, inertia, ramp rates)
    with economic parameters (CAPEX, OPEX, fuel model) and an availability
    model to represent a dispatchable or non-dispatchable power plant.

    Attributes:
        name (str): Human-readable generator name.
        gen_type (str): Technology type (``'gas'``, ``'solar'``, ``'wind'``,
            ``'nuclear'``, ``'hydro_mustrun'``).
        capacity_pu (float): Installed capacity in per-unit of system base.
        capacity_gw (float): Installed capacity in GW.
        capex_per_kw (float): Capital expenditure in EUR/kW.
        lifetime (float): Economic lifetime in years.
        vom (float): Variable O&M cost in EUR/MWh.
        fom (float): Fixed O&M cost in EUR/kW/year.
        efficiency (float): Thermal-to-electric efficiency (1.0 for renewables).
        emission_factor (float): CO2 emissions in tCO2/MWh_th.
        h_inertia (float): Inertia constant H in seconds (0 for non-synchronous).
        min_stable_pct (float): Minimum stable generation as fraction of capacity.
        ramp_rate (float): Ramp rate as fraction of capacity per minute.
        startup_cost (float): Start-up cost in EUR/MW.
        fuel_model: Fuel price model instance (or ``None`` for zero-fuel-cost).
        availability: Availability model instance.
        decom_eur_kw (float): Decommissioning cost in EUR/kW.
    """

    def __init__(self, name: str, gen_type: str, capacity_gw: float,
                 capex_per_kw: float, lifetime_years: float,
                 vom_eur_mwh: float, fom_eur_kw_yr: float,
                 efficiency: float, emission_factor: float,
                 h_inertia: float, min_stable_pct: float,
                 ramp_rate_pct_per_min: float,
                 startup_cost_eur_mw: float,
                 fuel_model=None, availability_model=None,
                 decom_eur_kw: float = 0.0) -> None:
        """Initialize a generator with all technical and economic parameters.

        Args:
            name: Human-readable name for this generator.
            gen_type: Technology type identifier.
            capacity_gw: Installed capacity in GW.
            capex_per_kw: Capital expenditure in EUR per kW.
            lifetime_years: Economic lifetime in years.
            vom_eur_mwh: Variable O&M in EUR/MWh.
            fom_eur_kw_yr: Fixed O&M in EUR per kW per year.
            efficiency: Thermal-to-electric conversion efficiency.
            emission_factor: CO2 emission factor in tCO2/MWh_th.
            h_inertia: Inertia constant H in seconds.
            min_stable_pct: Minimum stable generation as fraction of capacity.
            ramp_rate_pct_per_min: Ramp rate (fraction of capacity per minute).
            startup_cost_eur_mw: Start-up cost in EUR/MW.
            fuel_model: Instance of a fuel price model (``FuelPriceModel``,
                ``ConstantFuelPrice``, or ``None``).
            availability_model: Instance of an availability model. Defaults
                to :class:`DispatchableAvailability`.
            decom_eur_kw: Decommissioning cost in EUR/kW. Defaults to 0.
        """
        self.name = name
        self.gen_type = gen_type
        self.capacity_pu = capacity_gw / P_PEAK_GW
        self.capacity_gw = capacity_gw
        self.capex_per_kw = capex_per_kw
        self.lifetime = lifetime_years
        self.vom = vom_eur_mwh
        self.fom = fom_eur_kw_yr
        self.efficiency = efficiency
        self.emission_factor = emission_factor
        self.h_inertia = h_inertia
        self.min_stable_pct = min_stable_pct
        self.ramp_rate = ramp_rate_pct_per_min
        self.startup_cost = startup_cost_eur_mw
        self.fuel_model = fuel_model
        self.availability = availability_model or DispatchableAvailability()
        self.decom_eur_kw = decom_eur_kw

        # Populated per MC run by prepare_run()
        self._fuel_path: np.ndarray | None = None
        self._co2_path: np.ndarray | None = None
        self._avail_profile: np.ndarray | None = None

    @property
    def is_synchronous(self) -> bool:
        """Whether this generator provides rotational inertia (H > 0)."""
        return self.h_inertia > 0

    def prepare_run(self, time_grid: TimeGrid, rng: np.random.Generator,
                    co2_model: CarbonPriceModel) -> None:
        """Generate stochastic paths for one Monte Carlo year.

        Must be called before :meth:`srmc` or :meth:`available_power_pu`.

        Args:
            time_grid: Temporal backbone for the simulated year.
            rng: NumPy random generator for this MC run.
            co2_model: Carbon price model to generate CO2 price path.
        """
        if self.fuel_model is not None:
            self._fuel_path = self.fuel_model.generate_path(time_grid.n, rng)
        else:
            self._fuel_path = np.zeros(time_grid.n)
        self._co2_path = co2_model.generate_path(time_grid.n, rng)
        self._avail_profile = self.availability.generate_profile(time_grid, rng)

    def srmc(self) -> np.ndarray:
        """Compute the short-run marginal cost vector.

        SRMC = fuel_price / efficiency + CO2_price * emission_factor / efficiency + VOM.

        Returns:
            np.ndarray: SRMC array of shape ``(35040,)`` in EUR/MWh_e.
        """
        fuel_component = self._fuel_path / self.efficiency if self.efficiency > 0 else 0.0
        co2_component = (self._co2_path * self.emission_factor / self.efficiency
                         if self.efficiency > 0 else 0.0)
        return fuel_component + co2_component + self.vom

    def available_power_pu(self) -> np.ndarray:
        """Compute available power at each time step.

        Returns:
            np.ndarray: Available power array of shape ``(35040,)`` in per-unit
                of system base.
        """
        return self.capacity_pu * self._avail_profile

    def min_stable_power_pu(self) -> float:
        """Compute minimum stable generation level.

        Returns:
            float: Minimum stable generation in per-unit of system base.
        """
        return self.capacity_pu * self.min_stable_pct

    def lcoe(self) -> float:
        """Compute levelized cost of energy (approximate).

        Uses the capital recovery factor method with estimated average
        capacity factors by technology type.

        Returns:
            float: LCOE in EUR/MWh.
        """
        r = DISCOUNT_RATE
        n = self.lifetime
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        avg_cf = {'nuclear': 0.85, 'gas': 0.50, 'coal': 0.55,
                  'solar': 0.15, 'wind': 0.25}.get(self.gen_type, 0.90)
        fixed = (self.capex_per_kw * crf + self.fom) / (avg_cf * 8760) * 1000
        variable = self.vom
        fuel_avg = (self._fuel_path.mean() / self.efficiency
                    if self._fuel_path is not None and self.efficiency < 1.0
                    else 0)
        return fixed + variable + fuel_avg


# ── Factory ───────────────────────────────────────────────────────────────


def build_generators(mix_config: dict, gas_scenario: dict,
                     coal_scenario: dict | None = None) -> list[Generator]:
    """Build a list of Generator objects from configuration dicts.

    Routes each technology type to the appropriate fuel price model and
    availability model based on ``gen_type``. Generators with zero capacity
    are skipped.

    Args:
        mix_config: Generation mix dictionary mapping technology type strings
            to parameter dicts (see :data:`~sim_stochastic_pv.market.config.ITALIAN_MIX`
            for format).
        gas_scenario: Gas price scenario parameters (keys: ``mu``, ``sigma``,
            ``theta``) passed to :class:`FuelPriceModel`.
        coal_scenario: Coal price scenario parameters (keys: ``mu``, ``sigma``,
            ``theta``). If ``None``, defaults to ``COAL_SCENARIOS['base']``.

    Returns:
        list[Generator]: List of initialized Generator objects (one per
            technology with non-zero capacity).
    """
    generators = []
    coal_params = coal_scenario or COAL_SCENARIOS['base']

    for gen_type, params in mix_config.items():
        if params['capacity_gw'] <= 0:
            continue

        # Fuel model selection
        if gen_type == 'gas':
            fuel_model = FuelPriceModel(**gas_scenario)
        elif gen_type == 'coal':
            fuel_model = FuelPriceModel(**coal_params)
        elif gen_type == 'nuclear':
            fuel_model = ConstantFuelPrice(params.get('fuel_cost_eur_mwh_th', 3.0))
        else:
            fuel_model = None

        # Availability model selection
        if gen_type == 'solar':
            avail = SolarAvailability()
        elif gen_type == 'wind':
            avail = WindAvailability()
        elif gen_type == 'hydro_mustrun':
            avail = MustRunAvailability(cf=1.0)
        elif gen_type == 'nuclear':
            avail = MustRunAvailability(cf=0.90)
        else:
            avail = DispatchableAvailability()

        g = Generator(
            name=gen_type,
            gen_type=gen_type,
            capacity_gw=params['capacity_gw'],
            capex_per_kw=params['capex_per_kw'],
            lifetime_years=params['lifetime_years'],
            vom_eur_mwh=params['vom_eur_mwh'],
            fom_eur_kw_yr=params['fom_eur_kw_yr'],
            efficiency=params['efficiency'],
            emission_factor=params['emission_factor'],
            h_inertia=params['h_inertia'],
            min_stable_pct=params['min_stable_pct'],
            ramp_rate_pct_per_min=params['ramp_rate_pct_per_min'],
            startup_cost_eur_mw=params['startup_cost_eur_mw'],
            fuel_model=fuel_model,
            availability_model=avail,
        )
        generators.append(g)

    return generators
