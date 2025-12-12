from __future__ import annotations

import numpy as np

import json
from pathlib import Path
from typing import Any, Mapping

from sim_stochastic_pv.simulation import (
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
DEFAULT_SCENARIO_PATH = Path(__file__).resolve().parent / "examples" / "home_away_default.json"


def load_scenario_data(source: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Load scenario data from JSON or return the provided mapping.

    Args:
        source: Path to a JSON file, mapping, or None for default example.

    Returns:
        Dictionary containing scenario configuration.
    """
    if source is None:
        path = DEFAULT_SCENARIO_PATH
        return json.loads(path.read_text(encoding="utf-8"))
    if isinstance(source, (str, Path)):
        path = Path(source)
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(source)


def build_default_load_profile(scenario_data: Mapping[str, Any] | str | Path | None = None) -> LoadProfile:
    data = load_scenario_data(scenario_data)
    load_cfg = data["load_profile"]
    min_days_home = load_cfg.get("min_days_home", [15] * 12)  # Default fallback if missing
    max_days_home = load_cfg.get("max_days_home", [15] * 12)

    # Home Profile Factory
    home_type = load_cfg.get("home_profile_type", "custom").lower()

    def home_factory() -> LoadProfile:
        if home_type == "arera":
            return AreraLoadProfile()
        # Default custom
        if "home_profiles_w" in load_cfg:
            home_profiles_w = np.array(load_cfg["home_profiles_w"], dtype=float)
            if home_profiles_w.ndim == 1 and home_profiles_w.size == 12:
                # Expand (12,) to (12, 24)
                home_profiles_w = np.tile(home_profiles_w[:, np.newaxis], (1, 24))
            elif home_profiles_w.ndim == 1 and home_profiles_w.size == 24:
                 # Expand (24,) to (12, 24) - Single 24h profile for all months
                home_profiles_w = np.tile(home_profiles_w, (12, 1))
            return MonthlyAverageLoadProfile(home_profiles_w)
        raise ValueError("Missing 'home_profiles_w' for custom home profile.")

    # Away Profile Factory
    # 'away_profile' key can be "arera", "custom", or omitted (defaulting to arera in old logic)
    away_val = load_cfg.get("away_profile", "arera")
    # If it's a string, normalize it. If it's a list/array, treat as custom data?
    # Let's assume configuration uses 'away_profile': 'custom' and 'away_profiles_w': [...]
    # Or 'away_profile': 'arera'

    def away_factory() -> LoadProfile:
        if isinstance(away_val, str) and away_val.lower() == "arera":
            return AreraLoadProfile()
        if (isinstance(away_val, str) and away_val.lower() == "custom") or "away_profiles_w" in load_cfg:
            if "away_profiles_w" not in load_cfg:
                raise ValueError("Missing 'away_profiles_w' for custom away profile.")
            away_profiles_w = np.array(load_cfg["away_profiles_w"], dtype=float)
            if away_profiles_w.ndim == 1 and away_profiles_w.size == 12:
                away_profiles_w = np.tile(away_profiles_w[:, np.newaxis], (1, 24))
            elif away_profiles_w.ndim == 1 and away_profiles_w.size == 24:
                away_profiles_w = np.tile(away_profiles_w, (12, 1))
            return MonthlyAverageLoadProfile(away_profiles_w)
        # Fallback to Arera if unrecognized, or raise error?
        # Original code raised ValueError for unsupported types.
        raise ValueError(f"Unsupported away profile type: {away_val}")

    load_blueprint = LoadScenarioBlueprint(
        home_profile_factory=home_factory,
        away_profile_factory=away_factory,
        min_days_home=min_days_home,
        max_days_home=max_days_home,
        home_variation_percentiles=tuple(load_cfg.get("home_variation_percentiles", (-0.1, 0.1))),
        away_variation_percentiles=tuple(load_cfg.get("away_variation_percentiles", (-0.05, 0.05))),
    )
    return load_blueprint.build_load_profile()


def build_default_solar_model(scenario_data: Mapping[str, Any] | str | Path | None = None) -> SolarModel:
    data = load_scenario_data(scenario_data)
    solar_cfg = data["solar"]
    month_params_raw = solar_cfg.get("month_params")
    default_params = make_default_solar_params_for_pavullo()
    param_cls = type(default_params[0])
    if month_params_raw is None:
        month_params = default_params
    else:
        month_params = [
            param_cls(
                avg_daily_kwh_per_kwp=entry["avg_daily_kwh_per_kwp"],
                p_sunny=entry["p_sunny"],
                sunny_factor=entry["sunny_factor"],
                cloudy_factor=entry["cloudy_factor"],
            )
            for entry in month_params_raw
        ]
    return SolarModel(
        pv_kwp=solar_cfg.get("pv_kwp", 2.0),
        month_params=month_params,
        degradation_per_year=solar_cfg.get("degradation_per_year", 0.007),
    )


def build_default_energy_config(scenario_data: Mapping[str, Any] | str | Path | None = None) -> EnergySystemConfig:
    data = load_scenario_data(scenario_data)
    energy_cfg = data["energy"]
    battery_specs_data = energy_cfg.get("battery_specs", {"capacity_kwh": 0.0, "cycles_life": 0})
    battery_specs = BatterySpecs(
        capacity_kwh=battery_specs_data.get("capacity_kwh", 0.0),
        cycles_life=battery_specs_data.get("cycles_life", 0),
    )
    return EnergySystemConfig(
        n_years=energy_cfg.get("n_years", 20),
        pv_kwp=energy_cfg.get("pv_kwp", 2.0),
        battery_specs=battery_specs,
        n_batteries=energy_cfg.get("n_batteries", 0),
        inverter_p_ac_max_kw=energy_cfg.get("inverter_p_ac_max_kw", 1.0),
    )


def build_default_price_model(
    use_stochastic_price: bool | None = None,
    escalation_percentiles: tuple[float, float] | None = None,
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> EscalatingPriceModel:
    data = load_scenario_data(scenario_data)
    price_cfg = data["price"]
    return EscalatingPriceModel(
        base_price_eur_per_kwh=price_cfg.get("base_price_eur_per_kwh", 0.20),
        annual_escalation=price_cfg.get("annual_escalation", 0.02),
        use_stochastic_escalation=price_cfg.get(
            "use_stochastic_escalation",
            True if use_stochastic_price is None else use_stochastic_price,
        ),
        escalation_variation_percentiles=tuple(
            price_cfg.get("escalation_variation_percentiles", escalation_percentiles or (-0.05, 0.05))
        ),
    )


def build_default_economic_config(
    n_mc: int | None = None,
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> EconomicConfig:
    data = load_scenario_data(scenario_data)
    econ_cfg = data["economic"]
    return EconomicConfig(
        investment_eur=econ_cfg.get("investment_eur", 0.0),
        n_mc=n_mc or econ_cfg.get("n_mc", 100),
    )


def _build_inverter(option: Mapping[str, Any]) -> InverterOption:
    battery_specs = option.get("integrated_battery_specs")
    specs_obj = (
        BatterySpecs(
            capacity_kwh=battery_specs.get("capacity_kwh", 0.0),
            cycles_life=battery_specs.get("cycles_life", 0),
        )
        if battery_specs
        else None
    )
    return InverterOption(
        name=option["name"],
        p_ac_max_kw=option["p_ac_max_kw"],
        p_dc_max_kw=option.get("p_dc_max_kw"),
        price_eur=option.get("price_eur", 0.0),
        install_cost_eur=option.get("install_cost_eur"),
        integrated_battery_specs=specs_obj,
        integrated_battery_price_eur=option.get("integrated_battery_price_eur"),
        integrated_battery_count_options=option.get("integrated_battery_count_options"),
        manufacturer=option.get("manufacturer"),
        model_number=option.get("model_number"),
        datasheet=option.get("datasheet"),
    )


def build_default_optimization_request(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> OptimizationRequest:
    data = load_scenario_data(scenario_data)
    opt_cfg = data["optimization"]

    inverter_options = [_build_inverter(opt) for opt in opt_cfg.get("inverter_options", [])]

    panel_options = [
        PanelOption(
            name=opt["name"],
            power_w=opt.get("power_w", 0.0),
            price_eur=opt.get("price_eur", 0.0),
            manufacturer=opt.get("manufacturer"),
            model_number=opt.get("model_number"),
            datasheet=opt.get("datasheet"),
        )
        for opt in opt_cfg.get("panel_options", [])
    ]

    battery_options = [
        BatteryOption(
            name=opt["name"],
            specs=BatterySpecs(
                capacity_kwh=opt["specs"].get("capacity_kwh", 0.0),
                cycles_life=opt["specs"].get("cycles_life", 0),
            ),
            price_eur=opt.get("price_eur", 0.0),
            manufacturer=opt.get("manufacturer"),
            model_number=opt.get("model_number"),
            datasheet=opt.get("datasheet"),
        )
        for opt in opt_cfg.get("battery_options", [])
    ]

    return OptimizationRequest(
        scenario_name=data.get("scenario_name", "custom_scenario"),
        inverter_options=inverter_options,
        panel_options=panel_options,
        panel_count_options=opt_cfg.get("panel_count_options", [1]),
        battery_options=battery_options,
        battery_count_options=opt_cfg.get("battery_count_options", [0]),
        include_no_battery=opt_cfg.get("include_no_battery", True),
    )
