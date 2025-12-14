# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Major Refactoring (Breaking Changes)

This release includes a comprehensive refactoring to improve code organization, fix inconsistencies, and enhance workflow clarity.

#### Breaking Changes

##### Terminology Standardization
- **CLI**: `pv-sim campaign` → `pv-sim optimize` (all campaign commands renamed to optimization)
- **API**: `/api/campaigns/{id}/run` → `/api/optimizations/{id}/run`
- **Database**: `config_type = "campaign"` → `config_type = "optimization"` in configuration records
- **Migration**: Database is fresh start; no migration needed if database is empty

##### Default Seed Unification
- **Optimization seed**: Changed from `321` to `123` for consistency with analysis
- All workflows (analysis, optimization) now use `seed=123` by default
- **Impact**: Results from previous runs with default seed will differ slightly

##### File Structure Reorganization
- **Module splits**: Several large files split into logical submodules (backward compatible via `__init__.py` re-exports)
  - `simulation/load_profiles.py` (1806 lines) → `simulation/load_profiles/` (8 files)
  - `simulation/monte_carlo.py` (1062 lines) → `simulation/monte_carlo/` (3 files)
  - `simulation/optimizer.py` (775 lines) → `simulation/optimizer/` (4 files)
  - `persistence.py` (753 lines) → `persistence/` (5 files with repository pattern)
- **Module merges**: `result_builder.py` + `reporting.py` → `output/` (6 files)
- **Moved files**: `scenario_setup.py` (root) → `scenario_builder.py` (in package)

##### Internal API Changes
- **Persistence**: Split into repository classes (`HardwareRepository`, `ConfigurationRepository`, `ExecutionRepository`)
  - Main `PersistenceService` maintains backward compatibility via delegation
  - New methods: `hydrate_scenario()` and `hydrate_optimization()` (replace `hydrate_scenario_from_ids()` for new code)
- **Import paths**: Some internal imports changed (e.g., `from sim_stochastic_pv.output import ResultBuilder`)

##### Legacy Script Deprecation
- `main.py`, `main_analysis.py`, `main_optimization.py` now show deprecation warnings
- Users should migrate to CLI: `pv-sim analyze` or `pv-sim optimize`

### Added

#### Validation System
- **Post-hydration validation**: All configurations validated after hydration with clear error messages
- **Validation functions**: `validate_scenario()` and `validate_optimization()` in `validation.py`
- **Automatic checks**:
  - Required sections (load_profile, solar, energy, economic, price)
  - Type validation (integers, floats, dicts)
  - Range validation (positive values for pv_kwp, n_mc, n_years)
  - Optimization-specific checks (at least one hardware option list)
- **Integration**: Validation runs automatically in CLI and API before execution
- **Error reporting**: Clear, actionable error messages (e.g., "energy.pv_kwp must be positive")

#### Progress Indicators
- **tqdm integration**: Monte Carlo simulations and optimizations show real-time progress bars
- **Auto-detect TTY**: Progress bars automatically hidden in non-interactive environments
- **Dual-level progress**: Optimizations show both scenario-level and MC-level progress
- **Example output**:
  ```
  Monte Carlo simulation: 100%|████████████████| 200/200 [00:45<00:00, 4.42path/s]
  Evaluating scenarios: 100%|████████████████| 36/36 [02:30<00:00, 4.17s/scenario]
  ```

### Changed

#### Performance Optimizations
- **Hardware persistence**: Batch upsert for unique hardware in optimizations (300 calls → ~10-20 calls)
  - Optimizations with 100 scenarios now ~15x faster for database writes

#### Code Quality
- **CLI deduplication**: Extracted `_add_execution_arguments()` helper to eliminate ~30 lines of duplicate code
- **Type safety**: Added missing imports and type hints across refactored modules
- **Separation of concerns**: Clear module boundaries with single responsibilities
  - Load profiles: 8 specialized files (base, monthly, arera, home_away, variable, blueprint, helpers)
  - Monte Carlo: 3 files (core, finance, plotting)
  - Optimizer: 4 files (hardware, scenarios, core, generators)
  - Persistence: 5 files (repositories + hydration + facade)
  - Output: 6 files (result_builder, plots, formatters, report, utils)

#### Documentation
- **README**: Updated all terminology from "campaign" to "optimization"
- **Import examples**: Added examples showing reorganized module imports
- **Validation section**: New section explaining configuration validation and common errors
- **Progress section**: Documented progress indicator behavior

### Fixed

- **Inconsistent terminology**: "campaign" vs "optimization" confusion resolved
- **Different default seeds**: Analysis (123) and optimization (321) now unified to 123
- **Duplicate CLI code**: Scenario and optimization run parsers now share common helper
- **Missing imports**: Added missing imports in optimizer/core.py (product, replace, time, sys, Callable)
- **Inefficient persistence**: Hardware no longer upserted repeatedly in optimization loops

### Deprecated

- `main.py`: Use `pv-sim analyze` or `pv-sim optimize` instead
- `main_analysis.py`: Use `pv-sim analyze` instead
- `main_optimization.py`: Use `pv-sim optimize` instead
- `hydrate_scenario_from_ids()`: Use `hydrate_scenario()` or `hydrate_optimization()` for new code (backward compatibility maintained)

### Internal

- **Repository pattern**: Persistence layer now uses repository pattern with facade for backward compatibility
- **Module organization**: Large monolithic files split into focused submodules
- **Hydration split**: Separate functions for scenario vs optimization hydration
- **Progress callbacks**: MonteCarloSimulator and ScenarioOptimizer support `show_progress` parameter

---

## How to Migrate

### For CLI Users

```bash
# Old (deprecated, shows warning)
python -m sim_stochastic_pv.cli campaign run --name my_campaign

# New
python -m sim_stochastic_pv.cli optimize run --name my_optimization
```

### For API Users

```bash
# Old endpoint
POST /api/campaigns/5/run

# New endpoint
POST /api/optimizations/5/run
```

### For Library Users

Most imports remain backward compatible via `__init__.py` re-exports:

```python
# Still works (backward compatible)
from sim_stochastic_pv.simulation.load_profiles import HomeAwayLoadProfile
from sim_stochastic_pv.simulation.monte_carlo import MonteCarloSimulator
from sim_stochastic_pv.persistence import PersistenceService

# New imports (preferred for new code)
from sim_stochastic_pv.output import ResultBuilder  # was result_builder
from sim_stochastic_pv.scenario_builder import load_scenario_data  # was scenario_setup
```

### Database Migration

If you have an existing database with "campaign" config_type records:

```sql
-- Manual migration (if needed)
UPDATE configuration SET config_type = 'optimization' WHERE config_type = 'campaign';
```

**Note**: If your database is empty (as expected), no migration needed.

---

## Contributors

- Francesco Bazzani
- Claude Code (AI-assisted refactoring)
