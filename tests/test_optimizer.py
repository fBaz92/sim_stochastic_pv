"""Tests for scenario optimizer module (reorganized)."""

from __future__ import annotations

import pytest

from sim_stochastic_pv.simulation.optimizer import (
    InverterOption,
    PanelOption,
    BatteryOption,
    ScenarioDefinition,
    ScenarioEvaluation,
    OptimizationRequest,
    ScenarioOptimizer,
)
from sim_stochastic_pv.simulation.battery import BatterySpecs
from sim_stochastic_pv.simulation.monte_carlo import EconomicConfig
from sim_stochastic_pv.simulation.energy_simulator import EnergySystemConfig
from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile
from sim_stochastic_pv.simulation.solar import SolarModel, SolarMonthParams
from sim_stochastic_pv.simulation.prices import EscalatingPriceModel


class TestHardwareOptions:
    """Tests for hardware option dataclasses."""

    def test_inverter_option(self):
        """InverterOption can be created."""
        inverter = InverterOption(
            name="Test Inverter",
            p_ac_max_kw=5.0,
            p_dc_max_kw=5.5,
            price_eur=1500.0,
            install_cost_eur=300.0,
        )

        assert inverter.name == "Test Inverter"
        assert inverter.p_ac_max_kw == 5.0
        assert inverter.total_cost_eur == 1800.0

    def test_panel_option(self):
        """PanelOption can be created."""
        panel = PanelOption(
            name="Test Panel",
            power_w=400.0,
            price_eur=120.0,
        )

        assert panel.name == "Test Panel"
        assert panel.power_w == 400.0
        assert panel.price_eur == 120.0

    def test_battery_option(self):
        """BatteryOption can be created."""
        battery = BatteryOption(
            name="Test Battery",
            specs=BatterySpecs(capacity_kwh=5.0, cycles_life=6000),
            price_eur=2000.0,
        )

        assert battery.name == "Test Battery"
        assert battery.specs.capacity_kwh == 5.0
        assert battery.price_eur == 2000.0


class TestScenarioDefinition:
    """Tests for ScenarioDefinition dataclass."""

    def test_creation(self):
        """ScenarioDefinition can be created."""
        inverter = InverterOption(
            name="Inv5kW", p_ac_max_kw=5.0, price_eur=1500.0
        )
        panel = PanelOption(name="Panel400W", power_w=400.0, price_eur=120.0)

        scenario = ScenarioDefinition(
            inverter=inverter,
            panel=panel,
            panel_count=10,
            battery_option=None,
            battery_count=0,
        )

        assert scenario.inverter.name == "Inv5kW"
        assert scenario.panel_count == 10

    def test_pv_kwp_calculation(self):
        """pv_kwp is calculated from panel power and count."""
        inverter = InverterOption(name="Inv5kW", p_ac_max_kw=5.0, price_eur=1500.0)
        panel = PanelOption(name="Panel400W", power_w=400.0, price_eur=120.0)

        scenario = ScenarioDefinition(
            inverter=inverter,
            panel=panel,
            panel_count=10,
            battery_option=None,
            battery_count=0,
        )

        # 10 panels × 400W = 4000W = 4.0 kWp
        assert scenario.pv_kwp == 4.0

    def test_investment_calculation_no_battery(self):
        """Investment is calculated correctly without battery."""
        inverter = InverterOption(
            name="Inv5kW",
            p_ac_max_kw=5.0,
            price_eur=1500.0,
            install_cost_eur=300.0,
        )
        panel = PanelOption(name="Panel400W", power_w=400.0, price_eur=120.0)

        scenario = ScenarioDefinition(
            inverter=inverter,
            panel=panel,
            panel_count=10,
            battery_option=None,
            battery_count=0,
        )

        # Inverter: 1500 + 300 = 1800
        # Panels: 10 × 120 = 1200
        # Total: 3000
        assert scenario.investment_eur == 3000.0

    def test_investment_calculation_with_battery(self):
        """Investment is calculated correctly with battery."""
        inverter = InverterOption(
            name="Inv5kW", p_ac_max_kw=5.0, price_eur=1500.0
        )
        panel = PanelOption(name="Panel400W", power_w=400.0, price_eur=120.0)
        battery = BatteryOption(
            name="Battery5kWh",
            specs=BatterySpecs(capacity_kwh=5.0, cycles_life=6000),
            price_eur=2000.0,
        )

        scenario = ScenarioDefinition(
            inverter=inverter,
            panel=panel,
            panel_count=10,
            battery_option=battery,
            battery_count=2,
        )

        # Inverter: 1500
        # Panels: 10 × 120 = 1200
        # Batteries: 2 × 2000 = 4000
        # Total: 6700
        assert scenario.investment_eur == 6700.0

    def test_describe(self):
        """describe() generates readable scenario description."""
        inverter = InverterOption(name="Inv5kW", p_ac_max_kw=5.0, price_eur=1500.0)
        panel = PanelOption(name="Panel400W", power_w=400.0, price_eur=120.0)

        scenario = ScenarioDefinition(
            inverter=inverter,
            panel=panel,
            panel_count=10,
            battery_option=None,
            battery_count=0,
        )

        description = scenario.describe()
        assert "Inv5kW" in description
        assert "Panel400W" in description
        assert "10" in description  # panel count


class TestOptimizationRequest:
    """Tests for OptimizationRequest dataclass."""

    def test_creation(self):
        """OptimizationRequest can be created."""
        request = OptimizationRequest(
            scenario_name="Test Optimization",
            inverter_options=[
                InverterOption(name="Inv5kW", p_ac_max_kw=5.0, price_eur=1500.0)
            ],
            panel_options=[
                PanelOption(name="Panel400W", power_w=400.0, price_eur=120.0)
            ],
            panel_count_options=[10, 12],
            battery_options=[],
            battery_count_options=[0],
            include_no_battery=True,
        )

        assert request.scenario_name == "Test Optimization"
        assert len(request.inverter_options) == 1
        assert len(request.panel_count_options) == 2


class TestScenarioOptimizer:
    """Tests for ScenarioOptimizer."""

    @pytest.fixture
    def minimal_optimizer(self):
        """Create a minimal optimizer for testing."""
        # Hardware options
        request = OptimizationRequest(
            scenario_name="Test Optimization",
            inverter_options=[
                InverterOption(name="Inv1kW", p_ac_max_kw=1.0, price_eur=500.0)
            ],
            panel_options=[
                PanelOption(name="Panel400W", power_w=400.0, price_eur=100.0)
            ],
            panel_count_options=[1, 2],  # 2 scenarios
            battery_options=[],
            battery_count_options=[0],
            include_no_battery=True,
        )

        # Base energy config
        base_energy_config = EnergySystemConfig(
            n_years=2,
            pv_kwp=0.0,  # Will be overridden
            battery_specs=BatterySpecs(capacity_kwh=1.0, cycles_life=1000),
            n_batteries=0,
            inverter_p_ac_max_kw=1.0,
        )

        # Economic config with minimal MC paths
        economic_config = EconomicConfig(
            investment_eur=0.0,  # Will be overridden
            n_mc=3,  # Minimal for speed
            inflation_rate=0.025,
        )

        # Price model
        price_model = EscalatingPriceModel(
            base_price_eur_per_kwh=0.2,
            annual_escalation=0.01,
        )

        # Load profile factory
        def load_profile_factory():
            return MonthlyAverageLoadProfile(monthly_avg_kwh=[200.0] * 12)

        # Solar model
        month_params = [
            SolarMonthParams(
                avg_daily_kwh_per_kwp=2.0,
                p_sunny=0.5,
                sunny_factor=1.1,
                cloudy_factor=0.4,
            )
            for _ in range(12)
        ]
        solar_model = SolarModel(
            pv_kwp=1.0,
            degradation_per_year=0.0,
            month_params=month_params,
        )

        return ScenarioOptimizer(
            request=request,
            base_energy_config=base_energy_config,
            economic_config_template=economic_config,
            price_model=price_model,
            load_profile_factory=load_profile_factory,
            solar_model=solar_model,
        )

    def test_initialization(self, minimal_optimizer):
        """Optimizer can be initialized."""
        assert minimal_optimizer.request.scenario_name == "Test Optimization"
        assert minimal_optimizer.economic_config_template.n_mc == 3

    def test_run_returns_evaluations(self, minimal_optimizer):
        """Run method returns list of ScenarioEvaluations."""
        evaluations = minimal_optimizer.run(seed=42, show_progress=False)

        assert isinstance(evaluations, list)
        assert len(evaluations) > 0
        assert all(isinstance(e, ScenarioEvaluation) for e in evaluations)

    def test_run_generates_correct_number_of_scenarios(self, minimal_optimizer):
        """Run should generate expected number of scenarios."""
        # 1 inverter × 1 panel × 2 panel_counts = 2 scenarios
        evaluations = minimal_optimizer.run(seed=42, show_progress=False)
        assert len(evaluations) == 2

    def test_run_with_seed_is_deterministic(self, minimal_optimizer):
        """Same seed produces identical results."""
        evaluations1 = minimal_optimizer.run(seed=42, show_progress=False)
        evaluations2 = minimal_optimizer.run(seed=42, show_progress=False)

        # Should produce same number of scenarios
        assert len(evaluations1) == len(evaluations2)

        # Final gains should be identical
        for e1, e2 in zip(evaluations1, evaluations2):
            assert abs(e1.final_gain_eur - e2.final_gain_eur) < 0.01

    def test_evaluations_have_results(self, minimal_optimizer):
        """Each evaluation should have Monte Carlo results."""
        evaluations = minimal_optimizer.run(seed=42, show_progress=False)

        for evaluation in evaluations:
            assert evaluation.results is not None
            assert evaluation.results.df_profit is not None
            assert evaluation.final_gain_eur is not None

    def test_evaluations_have_definitions(self, minimal_optimizer):
        """Each evaluation should have scenario definition."""
        evaluations = minimal_optimizer.run(seed=42, show_progress=False)

        for evaluation in evaluations:
            assert isinstance(evaluation.definition, ScenarioDefinition)
            assert evaluation.definition.pv_kwp > 0

    def test_show_progress_parameter(self, minimal_optimizer):
        """Show progress parameter controls progress display."""
        # Should run without errors regardless of show_progress value
        evaluations1 = minimal_optimizer.run(seed=42, show_progress=True)
        evaluations2 = minimal_optimizer.run(seed=42, show_progress=False)

        # Both should produce valid results
        assert len(evaluations1) == len(evaluations2)

    def test_break_even_calculation(self, minimal_optimizer):
        """Break-even month should be calculated."""
        evaluations = minimal_optimizer.run(seed=42, show_progress=False)

        for evaluation in evaluations:
            # break_even_month can be None if never breaks even
            # or an integer if it does
            assert evaluation.break_even_month is None or isinstance(
                evaluation.break_even_month, int
            )


class TestBackwardCompatibility:
    """Tests for backward compatibility of optimizer imports."""

    def test_all_exports_available(self):
        """All expected classes should be importable from main module."""
        from sim_stochastic_pv.simulation.optimizer import (
            InverterOption,
            PanelOption,
            BatteryOption,
            ScenarioDefinition,
            ScenarioEvaluation,
            OptimizationRequest,
            ScenarioOptimizer,
        )

        assert InverterOption is not None
        assert PanelOption is not None
        assert BatteryOption is not None
        assert ScenarioDefinition is not None
        assert ScenarioEvaluation is not None
        assert OptimizationRequest is not None
        assert ScenarioOptimizer is not None

    def test_submodule_imports(self):
        """Hardware and scenario classes should be in submodules."""
        from sim_stochastic_pv.simulation.optimizer.hardware import (
            InverterOption,
            PanelOption,
            BatteryOption,
        )
        from sim_stochastic_pv.simulation.optimizer.scenarios import (
            ScenarioDefinition,
            ScenarioEvaluation,
            OptimizationRequest,
        )

        # Should be importable from submodules
        assert InverterOption is not None
        assert ScenarioDefinition is not None
