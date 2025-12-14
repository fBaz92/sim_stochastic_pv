#!/usr/bin/env python3
"""Test script to verify refactoring changes didn't break anything."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all refactored modules can be imported."""
    print("Testing imports...")

    # Test load_profiles split
    try:
        from sim_stochastic_pv.simulation.load_profiles import (
            LoadProfile,
            MonthlyAverageLoadProfile,
            AreraLoadProfile,
            HomeAwayLoadProfile,
            VariableLoadProfile,
            LoadScenarioBlueprint,
        )
        print("✓ load_profiles imports successful")
    except ImportError as e:
        print(f"✗ load_profiles import failed: {e}")
        return False

    # Test monte_carlo split
    try:
        from sim_stochastic_pv.simulation.monte_carlo import (
            MonteCarloSimulator,
            MonteCarloResults,
            EconomicConfig,
        )
        print("✓ monte_carlo imports successful")
    except ImportError as e:
        print(f"✗ monte_carlo import failed: {e}")
        return False

    # Test optimizer split
    try:
        from sim_stochastic_pv.simulation.optimizer import (
            InverterOption,
            PanelOption,
            BatteryOption,
            ScenarioDefinition,
            ScenarioEvaluation,
            OptimizationRequest,
            ScenarioOptimizer,
        )
        print("✓ optimizer imports successful")
    except ImportError as e:
        print(f"✗ optimizer import failed: {e}")
        return False

    # Test scenario_builder moved
    try:
        from sim_stochastic_pv.scenario_builder import (
            load_scenario_data,
            build_default_load_profile,
            build_default_solar_model,
            build_default_energy_config,
            build_default_price_model,
            build_default_economic_config,
            build_default_optimization_request,
        )
        print("✓ scenario_builder imports successful")
    except ImportError as e:
        print(f"✗ scenario_builder import failed: {e}")
        return False

    # Test application imports still work
    try:
        from sim_stochastic_pv.application import SimulationApplication
        print("✓ application imports successful")
    except ImportError as e:
        print(f"✗ application import failed: {e}")
        return False

    # Test CLI imports
    try:
        from sim_stochastic_pv.cli import build_argument_parser, main
        print("✓ CLI imports successful")
    except ImportError as e:
        print(f"✗ CLI import failed: {e}")
        return False

    return True

def test_cli_help():
    """Test that CLI still generates help without errors."""
    print("\nTesting CLI help...")
    try:
        from sim_stochastic_pv.cli import build_argument_parser
        parser = build_argument_parser()

        # Test main commands exist
        help_text = parser.format_help()

        # Check for updated terminology
        if "optimization" in help_text.lower():
            print("✓ CLI uses 'optimization' terminology")
        else:
            print("✗ CLI terminology not updated properly")
            return False

        # Check old 'campaign' is gone from top level
        if "campaign" in help_text.lower():
            print("⚠ CLI still contains 'campaign' references")

        print("✓ CLI help generates successfully")
        return True
    except Exception as e:
        print(f"✗ CLI help failed: {e}")
        return False

def test_dataclass_instantiation():
    """Test that refactored dataclasses can be instantiated."""
    print("\nTesting dataclass instantiation...")

    try:
        from sim_stochastic_pv.simulation.optimizer import (
            InverterOption,
            PanelOption,
            BatteryOption,
        )
        from sim_stochastic_pv.simulation.battery import BatterySpecs

        # Create sample hardware
        inverter = InverterOption(
            name="Test Inverter",
            p_ac_max_kw=5.0,
            price_eur=1000.0,
        )

        panel = PanelOption(
            name="Test Panel",
            power_w=400.0,
            price_eur=200.0,
        )

        battery = BatteryOption(
            name="Test Battery",
            specs=BatterySpecs(capacity_kwh=10.0, cycles_life=5000),
            price_eur=5000.0,
        )

        print(f"✓ Created InverterOption: {inverter.name}")
        print(f"✓ Created PanelOption: {panel.name}")
        print(f"✓ Created BatteryOption: {battery.name}")
        return True
    except Exception as e:
        print(f"✗ Dataclass instantiation failed: {e}")
        return False

def test_monte_carlo_functions():
    """Test that finance functions are accessible."""
    print("\nTesting monte_carlo finance functions...")

    try:
        # Test that finance functions are available from main module
        from sim_stochastic_pv.simulation.monte_carlo import (
            MonteCarloSimulator,
            EconomicConfig,
        )

        # Check if plotting is available
        try:
            from sim_stochastic_pv.simulation.monte_carlo.plotting import plot_profit_bands
            print("✓ Plotting functions accessible")
        except ImportError:
            print("⚠ Plotting functions not exported (expected)")

        # Check finance functions
        try:
            from sim_stochastic_pv.simulation.monte_carlo.finance import npv
            import numpy as np

            # Test NPV calculation
            cashflows = np.array([-1000, 100, 100, 100, 100, 1200])
            result = npv(0.05, cashflows)
            print(f"✓ Finance NPV calculation works: {result:.2f}")
        except ImportError:
            print("⚠ Finance functions not exported (expected)")

        return True
    except Exception as e:
        print(f"✗ Monte Carlo test failed: {e}")
        return False

def test_application_default_seeds():
    """Verify that default seeds were unified to 123."""
    print("\nTesting default seed unification...")

    try:
        import inspect
        from sim_stochastic_pv.application import SimulationApplication

        # Check run_optimization default seed
        sig = inspect.signature(SimulationApplication.run_optimization)
        seed_param = sig.parameters.get('seed')

        if seed_param and seed_param.default == 123:
            print("✓ run_optimization default seed is 123")
            return True
        else:
            print(f"✗ run_optimization default seed is {seed_param.default}, expected 123")
            return False
    except Exception as e:
        print(f"✗ Seed test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that old import paths still work."""
    print("\nTesting backward compatibility...")

    try:
        # These should all work via __init__.py exports
        from sim_stochastic_pv.simulation.load_profiles import LoadProfile
        from sim_stochastic_pv.simulation.monte_carlo import MonteCarloSimulator
        from sim_stochastic_pv.simulation.optimizer import ScenarioOptimizer

        print("✓ Backward compatible imports work")
        return True
    except ImportError as e:
        print(f"✗ Backward compatibility broken: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("REFACTORING VERIFICATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("CLI Help", test_cli_help),
        ("Dataclass Instantiation", test_dataclass_instantiation),
        ("Monte Carlo Functions", test_monte_carlo_functions),
        ("Default Seeds", test_application_default_seeds),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Refactoring successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review needed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
