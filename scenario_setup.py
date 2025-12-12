from __future__ import annotations

import numpy as np

from sim_stochastic_pv import (
    AreraLoadProfile,
    BatteryOption,
    BatterySpecs,
    EconomicConfig,
    EnergySystemConfig,
    EscalatingPriceModel,
    InverterOption,
    LoadProfile,
    LoadScenarioBlueprint,
    MonthlyAverageLoadProfile,
    OptimizationRequest,
    PanelOption,
    SolarModel,
    make_default_solar_params_for_pavullo,
)
from sim_stochastic_pv.load_profiles import get_load_w


def build_default_load_profile() -> LoadProfile:
    home_profiles_w = np.zeros((12, 24))
    cold_months = {9, 10, 11, 0, 1}
    hot_months = {5, 6, 7, 8}
    other_months = sorted(set(range(12)) - cold_months - hot_months)

    for m in cold_months:
        home_profiles_w[m, :] = 700.0
    for m in hot_months:
        home_profiles_w[m, :] = 500.0
    for m in other_months:
        for h in range(24):
            home_profiles_w[m, h] = get_load_w(
                month_index=m,
                hour_in_month=h,
                first_weekday_of_month=0,
            )

    min_days_home = [2, 2, 3, 4, 5, 7, 10, 10, 6, 4, 3, 2]
    max_days_home = [6, 6, 8, 10, 12, 15, 20, 20, 12, 8, 6, 5]

    load_blueprint = LoadScenarioBlueprint(
        home_profile_factory=lambda: MonthlyAverageLoadProfile(home_profiles_w),
        away_profile_factory=lambda: AreraLoadProfile(),
        min_days_home=min_days_home,
        max_days_home=max_days_home,
        home_variation_percentiles=(-0.35, 0.35),
        away_variation_percentiles=(-0.10, 0.10),
    )
    return load_blueprint.build_load_profile()


def build_default_solar_model() -> SolarModel:
    solar_params = make_default_solar_params_for_pavullo()
    return SolarModel(pv_kwp=2.0, month_params=solar_params, degradation_per_year=0.007)


def build_default_energy_config() -> EnergySystemConfig:
    battery_specs = BatterySpecs(capacity_kwh=1.92, cycles_life=6000)
    return EnergySystemConfig(
        n_years=25,
        pv_kwp=2.0,
        battery_specs=battery_specs,
        n_batteries=1,
        inverter_p_ac_max_kw=0.8,
    )


def build_default_price_model(
    use_stochastic_price: bool = True,
    escalation_percentiles: tuple[float, float] = (-0.05, 0.05),
) -> EscalatingPriceModel:
    return EscalatingPriceModel(
        base_price_eur_per_kwh=0.19,
        annual_escalation=0.04,
        use_stochastic_escalation=use_stochastic_price,
        escalation_variation_percentiles=escalation_percentiles,
    )


def build_default_economic_config(n_mc: int = 100) -> EconomicConfig:
    return EconomicConfig(investment_eur=2500.0, n_mc=n_mc)


def build_default_optimization_request() -> OptimizationRequest:
    inverter_options = [

        InverterOption(
            name="Huawei 3kW SUN2000-3KTL-LB0",
            p_ac_max_kw=3000,
            p_dc_max_kw=3000,
            price_eur=752.21,
            install_cost_eur=2000.0,
        ),
        InverterOption(
            name="SolarFlow 800 Pro",
            p_ac_max_kw=1.0,
            p_dc_max_kw=2.640,
            price_eur=199.0,
            install_cost_eur=800.0,
            integrated_battery_specs=BatterySpecs(capacity_kwh=1.92, cycles_life=6000),
            integrated_battery_price_eur=499.0,
            integrated_battery_count_options=[1, 2, 3, 4, 5],
        ),
    ]

    panel_options = [
        PanelOption(name="P500", power_w=500.0, price_eur=79.0),
        PanelOption(name="P440", power_w=440.0, price_eur=63.0),
    ]

    battery_options = [
        BatteryOption(name="Huawei Luna LUNA2000-5KW-E0", specs=BatterySpecs(capacity_kwh=5, cycles_life=6000), price_eur=1986.88),
    ]

    return OptimizationRequest(
        scenario_name="home_away_default",
        inverter_options=inverter_options,
        panel_options=panel_options,
        panel_count_options=[2, 3, 4, 5, 6],
        battery_options=battery_options,
        battery_count_options=[1],
        include_no_battery=False,
    )
