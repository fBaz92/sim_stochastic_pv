"""Integration tests for CLI with validation and new features."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from sim_stochastic_pv.cli import main


class TestCLIValidation:
    """Tests for CLI validation integration."""

    def test_scenario_run_with_invalid_config_exits(self, simple_scenario_data):
        """Scenario run with invalid config should exit with error."""
        # Remove required section
        invalid_config = {**simple_scenario_data}
        del invalid_config["load_profile"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(invalid_config, f)
            config_file = f.name

        try:
            # Should exit with error
            with pytest.raises(SystemExit) as exc_info:
                with mock.patch.object(sys, "argv", [
                    "cli",
                    "analyze",
                    "--file",
                    config_file,
                    "--no-save",
                ]):
                    main()

            assert exc_info.value.code == 1
        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_scenario_run_with_valid_config_succeeds(self, simple_scenario_data):
        """Scenario run with valid config should succeed."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(simple_scenario_data, f)
            config_file = f.name

        try:
            # Should succeed
            with mock.patch.object(sys, "argv", [
                "cli",
                "analyze",
                "--file",
                config_file,
                "--no-save",
            ]):
                # Should not raise
                main()
        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_optimization_run_with_invalid_config_exits(self, simple_scenario_data):
        """Optimization run with invalid config should exit with error."""
        # Remove optimization section
        invalid_config = {**simple_scenario_data}
        del invalid_config["optimization"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(invalid_config, f)
            config_file = f.name

        try:
            with pytest.raises(SystemExit) as exc_info:
                with mock.patch.object(sys, "argv", [
                    "cli",
                    "optimize",
                    "--file",
                    config_file,
                    "--no-save",
                ]):
                    main()

            assert exc_info.value.code == 1
        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_optimization_run_with_empty_options_exits(self, simple_scenario_data):
        """Optimization run with empty hardware options should exit."""
        invalid_config = {**simple_scenario_data}
        # Empty all hardware options
        invalid_config["optimization"] = {
            "inverter_options": [],
            "panel_options": [],
            "battery_options": [],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(invalid_config, f)
            config_file = f.name

        try:
            with pytest.raises(SystemExit) as exc_info:
                with mock.patch.object(sys, "argv", [
                    "cli",
                    "optimize",
                    "--file",
                    config_file,
                    "--no-save",
                ]):
                    main()

            assert exc_info.value.code == 1
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestCLITerminology:
    """Tests for CLI terminology standardization."""

    def test_optimize_command_exists(self):
        """Optimize command should exist (not campaign)."""
        # This test verifies the command exists by checking if help works
        with mock.patch.object(sys, "argv", ["cli", "optimize", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Help should exit with code 0
            assert exc_info.value.code == 0

    def test_hardware_commands_exist(self):
        """Hardware management commands should exist."""
        with mock.patch.object(sys, "argv", ["cli", "hardware", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_campaign_alias_dispatches_like_optimization(self):
        """Phase 13: ``campaign`` is an alias of ``optimization`` (UI glossary
        change), and must dispatch to the same subparser group. Help must
        succeed; the subcommand grammar (list/save/run) must be available."""
        with mock.patch.object(sys, "argv", ["cli", "campaign", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0

    def test_design_alias_dispatches_like_optimization(self):
        """Phase 13: ``design`` is the UI-facing alias of ``optimization``
        introduced after the Phase 11 rename. Help must succeed."""
        with mock.patch.object(sys, "argv", ["cli", "design", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0

    def test_optimization_legacy_command_still_works(self):
        """Phase 13: backward compat — the historical ``optimization`` command
        must keep working unchanged for any external script."""
        with mock.patch.object(sys, "argv", ["cli", "optimization", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0


class TestCLIDefaultSeeds:
    """Tests for unified default seeds."""

    def test_analyze_default_seed_is_123(self, simple_scenario_data):
        """Analyze command should use seed 123 by default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(simple_scenario_data, f)
            config_file = f.name

        try:
            # Patch run_analysis to capture the seed
            with mock.patch(
                "sim_stochastic_pv.cli.SimulationApplication.run_analysis"
            ) as mock_run:
                mock_run.return_value = {
                    "scenario": "test",
                    "final_gain_mean_eur": 1000.0,
                    "final_gain_real_mean_eur": 900.0,
                    "prob_gain": 0.8,
                    "output_dir": None,
                }

                with mock.patch.object(sys, "argv", [
                    "cli",
                    "analyze",
                    "--file",
                    config_file,
                    "--no-save",
                ]):
                    main()

                # Check that seed was 123
                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["seed"] == 123
        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_optimize_default_seed_is_123(self, simple_scenario_data):
        """Optimize command should use seed 123 by default."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(simple_scenario_data, f)
            config_file = f.name

        try:
            with mock.patch(
                "sim_stochastic_pv.cli.SimulationApplication.run_optimization"
            ) as mock_run:
                mock_run.return_value = {
                    "evaluations": 2,
                    "output_dir": None,
                }

                with mock.patch.object(sys, "argv", [
                    "cli",
                    "optimize",
                    "--file",
                    config_file,
                    "--no-save",
                ]):
                    main()

                # Check that seed was 123
                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["seed"] == 123
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestCLIDeduplication:
    """Tests for CLI argument parsing deduplication."""

    def test_analyze_accepts_common_arguments(self, simple_scenario_data):
        """Analyze command should accept all common execution arguments."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(simple_scenario_data, f)
            config_file = f.name

        try:
            with mock.patch(
                "sim_stochastic_pv.cli.SimulationApplication.run_analysis"
            ) as mock_run:
                mock_run.return_value = {
                    "scenario": "test",
                    "final_gain_mean_eur": 1000.0,
                    "final_gain_real_mean_eur": 900.0,
                    "prob_gain": 0.8,
                    "output_dir": None,
                }

                with mock.patch.object(sys, "argv", [
                    "cli",
                    "analyze",
                    "--file",
                    config_file,
                    "--seed",
                    "42",
                    "--n-mc",
                    "100",
                    "--no-save",
                ]):
                    main()

                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["seed"] == 42
                assert call_kwargs["n_mc"] == 100
        finally:
            Path(config_file).unlink(missing_ok=True)

    def test_optimize_accepts_common_arguments(self, simple_scenario_data):
        """Optimize command should accept all common execution arguments."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(simple_scenario_data, f)
            config_file = f.name

        try:
            with mock.patch(
                "sim_stochastic_pv.cli.SimulationApplication.run_optimization"
            ) as mock_run:
                mock_run.return_value = {
                    "evaluations": 2,
                    "output_dir": None,
                }

                with mock.patch.object(sys, "argv", [
                    "cli",
                    "optimize",
                    "--file",
                    config_file,
                    "--seed",
                    "42",
                    "--n-mc",
                    "100",
                    "--no-save",
                ]):
                    main()

                call_kwargs = mock_run.call_args.kwargs
                assert call_kwargs["seed"] == 42
                assert call_kwargs["n_mc"] == 100
        finally:
            Path(config_file).unlink(missing_ok=True)
