from __future__ import annotations

from sim_stochastic_pv import (
    EnergySystemSimulator,
    MonteCarloSimulator,
    generate_report,
)

from scenario_setup import (
    build_default_economic_config,
    build_default_energy_config,
    build_default_load_profile,
    build_default_price_model,
    build_default_solar_model,
)


def main() -> None:
    load_profile = build_default_load_profile()
    solar_model = build_default_solar_model()
    energy_cfg = build_default_energy_config()
    price_model = build_default_price_model()
    econ_cfg = build_default_economic_config(n_mc=100)

    energy_sim = EnergySystemSimulator(
        config=energy_cfg,
        solar_model=solar_model,
        load_profile=load_profile,
    )

    mc = MonteCarloSimulator(
        energy_simulator=energy_sim,
        price_model=price_model,
        economic_config=econ_cfg,
    )

    results = mc.run(seed=123)
    output_dir = generate_report(
        scenario_name="home_away_default",
        results=results,
        energy_config=energy_cfg,
        economic_config=econ_cfg,
        price_model=price_model,
    )
    print(f"Report salvato in: {output_dir}")


if __name__ == "__main__":
    main()
