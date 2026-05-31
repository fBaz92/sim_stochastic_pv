## Simulatore Stocastico PV

Monte Carlo toolkit for residential photovoltaic + battery systems, packaged
as a Python library, CLI, and full-stack web app (FastAPI + Svelte). It
answers a concrete economic question:

> *"Is it worth investing X k€ in a PV + battery system, given my location,
> my consumption profile, and the uncertainty of electricity prices over the
> next 20 years?"*

The simulator combines a stochastic PV production model (two-state Markov
weather chain, per-location seasonal parameters), one of three price models
(deterministic escalation, Geometric Brownian Motion, or Ornstein–Uhlenbeck
mean reversion), and several configurable load profiles. Outputs include NPV,
IRR, break-even month distribution, and probability of break-even within the
investment horizon — exposed through an interactive dashboard with fan charts
for profit, price, energy, and inflation.

### Glossary

The codebase uses a few terms consistently — keep them aligned in
documentation, API schemas, and UI copy.

| Term | Meaning | Where it lives |
|---|---|---|
| **Scenario** | Economic analysis of *one* hardware configuration (single inverter, battery, kWp, load profile, price model). Stochasticity comes only from the Monte Carlo draws (weather / load / price). | DB: `SavedConfigurationModel.config_type = "scenario"`. UI: "Scenario". |
| **Campaign** (a.k.a. *Design*) | Design exploration: sweep over alternative inverters / panel counts / batteries to find the best configuration. | DB: `SavedConfigurationModel.config_type = "optimization"` (legacy DB name kept for backward compat). UI: "Campagna" / "Design". |
| **Run** | One execution of the Monte Carlo for a scenario or campaign. Persisted as `RunResultRecord`. | `result_type ∈ {"analysis", "optimization"}`. |
| **Profile** | Reusable DB-stored object referenced by ID from scenarios / campaigns (load / price / solar). | `LoadProfileModel`, `PriceProfileModel`, `SolarProfileModel`. |
| **Hardware** | Inverters, panels, batteries with full specs in the DB catalog. | `InverterModel`, `PanelModel`, `BatteryModel`. |

> The full glossary, design principles, and contribution rules live in
> [CLAUDE.md](CLAUDE.md). Always read it before working on the codebase.
> The implementation roadmap (current and future phases) lives in
> [ROADMAP.md](ROADMAP.md).

---

## Quick start

### Local (Python only)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api_main:app --reload          # backend on http://localhost:8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev                             # UI on http://localhost:5173
```

### Containerized (Docker)

```bash
docker compose up --build               # backend :8000, frontend :4173
```

### Run the test suite

```bash
source venv/bin/activate
pytest tests/ -q                        # 246 tests, runs in ~60 s
```

---

## Repository layout

| Path | Description |
|------|-------------|
| `sim_stochastic_pv/simulation/` | Core physical/economic models (solar, prices, battery, inverter dispatch, Monte Carlo loop, optimizer, load profiles). |
| `sim_stochastic_pv/application.py` | High-level orchestrator shared across CLI and API. |
| `sim_stochastic_pv/scenario_builder.py` | Translates JSON / DB payloads into runtime model objects. |
| `sim_stochastic_pv/cli.py` | CLI entry point (`python -m sim_stochastic_pv.cli ...`). |
| `sim_stochastic_pv/api/` | FastAPI app, routes, Pydantic schemas, background-job queue. |
| `sim_stochastic_pv/persistence/` | Repository-pattern CRUD helpers (hardware, configurations, execution, profiles). |
| `sim_stochastic_pv/db/` | SQLAlchemy models, session, and seeding. |
| `sim_stochastic_pv/output/` | Result builder, CSV/plot writers, Excel and PDF exporters. |
| `sim_stochastic_pv/validation.py` | Post-hydration scenario / campaign validation. |
| `sim_stochastic_pv/seed_data/` | JSON defaults (solar profiles, ARERA load profiles, etc.). |
| `frontend/` | Vite + Svelte UI (6-step wizard, dashboard with multiple chart tabs). |
| `tests/` | Pytest suite (in-memory SQLite, deterministic seeds). |
| `docs/` | Design notes (e.g. `electrical_simplifications.md`). |
| `Dockerfile.backend`, `frontend/Dockerfile`, `docker-compose.yml` | Container images and orchestration. |

---

## Requirements

- Python 3.11+ (developed on 3.13)
- pip + virtualenv
- SQLite (bundled) or PostgreSQL (optional, set `POSTGRES_DSN`)
- Docker (optional, for the containerized stack)
- WeasyPrint native libs (only for the Phase-11 PDF export):
  - macOS: `brew install pango cairo gdk-pixbuf libffi`
  - Debian/Ubuntu: `apt-get install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info fonts-dejavu-core`
  - Already baked into `Dockerfile.backend`.

`requirements.txt` lists runtime dependencies (FastAPI, NumPy, Pandas,
Matplotlib, SciPy, SQLAlchemy, Pydantic, tqdm, openpyxl, WeasyPrint, Jinja2).
Install dev tools (pytest, ruff, mypy, …) as needed.

---

## Local setup notes

Create an optional `.env` to override defaults:

```bash
POSTGRES_DSN=postgresql+psycopg://user:pass@host:5432/database   # enables Postgres
SIM_PV_DB_PATH=/absolute/path/to/sim_pv.db                       # custom SQLite path
```

Without `POSTGRES_DSN` the app materializes `sim_pv.db` in the repo root.

Matplotlib font-cache warnings on read-only environments:

```bash
export MPLCONFIGDIR=$(pwd)/.cache/matplotlib
export XDG_CACHE_HOME=$(pwd)/.cache
```

Containerized setup volumes mount the local database and source tree for
live reload (`./sim_pv.db`, `./sim_stochastic_pv`, `./api_main.py`).

---

## Usage modes

### Python library

```python
from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.output import ResultBuilder
from sim_stochastic_pv.persistence import PersistenceService

persistence = PersistenceService()
builder = ResultBuilder()
app = SimulationApplication(save_outputs=False, persistence=persistence, result_builder=builder)
summary = app.run_analysis(n_mc=100)
```

Core models can be imported directly:

```python
# Load profiles (per CLAUDE.md §Glossario — "Profilo")
from sim_stochastic_pv.simulation.load_profiles import (
    HomeAwayLoadProfile, AreraLoadProfile, WeeklyPatternLoadProfile,
)

# Monte Carlo + economics
from sim_stochastic_pv.simulation.monte_carlo import (
    MonteCarloSimulator, EconomicConfig, TaxBonusConfig, InflationConfig,
)

# Campaign (a.k.a. optimization) search
from sim_stochastic_pv.simulation.optimizer import ScenarioOptimizer, InverterOption

# Price models
from sim_stochastic_pv.simulation.prices import (
    EscalatingPriceModel, GBMPriceModel, MeanRevertingPriceModel,
)

# Solar with weather Markov chain
from sim_stochastic_pv.simulation.solar import SolarModel, SolarMonthParams

# Validation
from sim_stochastic_pv.validation import validate_scenario, validate_optimization
```

### CLI

Two layers of commands: on-the-fly execution (scenarios fed inline from JSON
files) and database-backed management of saved hardware / scenarios /
campaigns.

#### On-the-fly execution

```bash
python -m sim_stochastic_pv.cli analyze --n-mc 200
python -m sim_stochastic_pv.cli analyze --scenario-file examples/custom_scenario.json
python -m sim_stochastic_pv.cli optimize --n-mc 50
python -m sim_stochastic_pv.cli analyze --no-save     # skip CSV/plots
```

- `analyze`: single-scenario Monte Carlo.
- `optimize`: multi-scenario (campaign) sweep.
- Both accept `--file` (alias `--scenario-file`), `--seed`, `--n-mc`,
  `--no-save`.
- Outputs: DB entries + optional CSV/plots under
  `results/<timestamp>_<scenario>/`.

#### Hardware catalog

```bash
python -m sim_stochastic_pv.cli hardware upsert-inverter --name Primo5 --p-ac-max-kw 5 --price-eur 1500
python -m sim_stochastic_pv.cli hardware upsert-panel --name Longi540 --power-w 540 --price-eur 200
python -m sim_stochastic_pv.cli hardware upsert-battery --name BYD4.0 --capacity-kwh 4 --cycles-life 8000
python -m sim_stochastic_pv.cli hardware list --type inverter
```

#### Saved scenarios

```bash
python -m sim_stochastic_pv.cli scenario save --name base_case --file examples/home_away_default.json
python -m sim_stochastic_pv.cli scenario list
python -m sim_stochastic_pv.cli scenario run --name base_case --seed 42 --n-mc 200
python -m sim_stochastic_pv.cli scenario run --file examples/inline_scenario.json   # ad-hoc, not saved
```

#### Saved campaigns

The `campaign` subcommand (alias `design`) replaces `optimization` in the new
glossary. All three names work; pick whichever reads best.

```bash
python -m sim_stochastic_pv.cli campaign save --name premium_sweep --file examples/optimization_template.json
python -m sim_stochastic_pv.cli campaign list
python -m sim_stochastic_pv.cli campaign run --name premium_sweep --seed 123 --n-mc 50

# Equivalent legacy spellings (kept for backward compat):
python -m sim_stochastic_pv.cli design  save --name … --file …
python -m sim_stochastic_pv.cli optimization save --name … --file …
```

- Scenario / campaign JSON format mirrors the API payloads (hardware IDs,
  load profile, solar, energy, price, economic, optimization, …).
- DB-backed runs hydrate hardware and profiles by ID from the catalog at
  run time, so they always use the latest specs.
- Post-hydration validation runs automatically and produces structured
  error messages on stderr.

### FastAPI backend

Start locally:

```bash
source venv/bin/activate
uvicorn api_main:app --reload
```

Swagger UI: `http://localhost:8000/docs` · ReDoc: `http://localhost:8000/redoc`.

**Key endpoints** (all prefixed `/api`):

*Simulation execution*

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analysis` | Run a single scenario synchronously; return summary. |
| `POST` | `/optimization` | Run a campaign synchronously; return ranked options. |
| `POST` | `/scenarios/{scenario_id}/run` | Run a saved scenario (hydrated from DB). |
| `POST` | `/optimizations/{optimization_id}/run` | Run a saved campaign. |
| `POST` | `/campaigns/{campaign_id}/run` | Phase-7 alias of the above (preferred). |

*Async jobs (Phase 12)*

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs/analysis` | Submit `/analysis` as a background job. |
| `POST` | `/jobs/optimization` | Submit `/optimization` as a background job. |
| `GET`  | `/jobs/{job_id}` | Poll job state and progress. |
| `GET`  | `/jobs` | List recent jobs (debug helper). |

*Runs and exports*

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/runs` | List persisted runs (paginated, filterable). |
| `GET`  | `/runs/locations` | Distinct locations across runs (for filters). |
| `PATCH`| `/runs/{id}/archive` | Soft-archive a run (Phase 12). |
| `PATCH`| `/runs/{id}/unarchive` | Restore an archived run. |
| `DELETE` | `/runs/{id}` | Hard-delete a run. |
| `GET`  | `/runs/{id}/export/cashflow.xlsx` | Excel cash-flow + KPI workbook. |
| `GET`  | `/runs/{id}/export/report.pdf` | Multi-page PDF report. |

*Profiles*

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/profiles/solar` | List solar profiles. |
| `GET`/`POST`/`DELETE` | `/profiles/load[…]` | Manage load profiles. |
| `GET`/`POST`/`DELETE` | `/profiles/price[…]` | Manage price profiles. |
| `GET`  | `/profiles/price/{id}/preview` | Fan-chart preview of a saved price profile (Phase 10). |
| `POST` | `/profiles/price/preview` | Live preview from inline params. |
| `GET`  | `/load-profiles/template/{kind}.xlsx` | Download a load-profile Excel template. |
| `POST` | `/load-profiles/parse-xlsx/{kind}` | Parse a filled template back into a profile payload. |

*Thermal lab (Phase 19)*

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/thermal-lab/compare` | Monte-Carlo HVAC comparison of several house variants (insulation presets / custom `UA`) against a saved climate profile: annual energy, cost, comfort breaches, worst days, typical-year daily series. |
| `POST` | `/thermal-lab/timeseries` | Hourly preview (outdoor/indoor temperature, electric draw, setpoints) for one house configuration. |
| `POST` | `/thermal-lab/compare/export.xlsx` | Same body as `/compare`; streams a multi-sheet Excel workbook. |
| `POST` | `/thermal-lab/compare/export.pdf` | Same body as `/compare`; streams a charted PDF report. |

*Catalogs and configurations*

| Method | Path | Description |
|--------|------|-------------|
| `GET`/`POST`/`DELETE` | `/inverters[…]`, `/panels[…]`, `/batteries[…]` | Hardware catalog CRUD. |
| `GET`/`POST`/`DELETE` | `/configurations[…]` | Saved scenarios and campaigns. |
| `GET`  | `/scenarios` | Snapshot list of scenario records. |

The export endpoints read only the persisted summary JSON, so they work even
after the on-disk `results/` artefacts are deleted. The PDF degrades
gracefully on legacy runs predating Phase 11: missing sections are skipped
instead of erroring.

### Frontend (wizard + dashboard)

The Svelte UI is the primary surface for end users. It is built around a
6-step wizard mirroring the natural planning flow:

1. **Luogo** — select a solar profile (per-location weather + tilt/azimuth).
2. **Impianto** — kWp, panels, inverter, battery (DB-backed or inline).
3. **Carico** — ARERA, monthly, weekly pattern, or home/away (DB-backed
   profiles supported via `load_profile_id`).
4. **Mercato elettrico** — escalating / GBM / Ornstein–Uhlenbeck price
   model with live fan-chart preview.
5. **Investimento** — total CAPEX, horizon (years), `n_mc`, optional tax
   bonus and stochastic inflation.
6. **Riepilogo & Esegui** — summary table, save-as-named, and submit
   (synchronous or as a background job).

The dashboard surfaces a top-level **"Decisione"** card (probability of
break-even within horizon, expected break-even month with p05–p95 band,
mean IRR, median NPV) plus tabs for profit, energy, SoC/SoH, price fan
chart, inflation, and the per-month cash-flow table.

A separate **"Lab termico"** page (Phase 19) lets the user compare insulation
levels and size the heat pump *before* the economic scenario: pick a climate
profile and several house variants (insulation presets or a custom `UA`), then
read a KPI comparison table (with a heating/cooling energy split) and overlaid
charts (daily HVAC energy per variant vs outdoor temperature with worst-day
markers, cost per variant, and an hourly setpoint-vs-indoor-temperature preview
with a season selector). The annual cost can use a flat price or a stochastic
electricity-price model (escalation / GBM / mean-reverting), so the cost band
reflects price uncertainty. Exports the comparison as CSV, as a multi-sheet
Excel workbook, or as a charted PDF report (and each chart as PNG).

Run manually outside Docker:

```bash
cd frontend
npm install
npm run dev      # talks to backend on http://localhost:8000
```

---

## Configuration highlights

- `.env` handled by `sim_stochastic_pv.config`: `POSTGRES_DSN` and
  `SIM_PV_DB_PATH` control the SQLAlchemy DSN.
- Persistence service writes to the configured DB and is shared by CLI / API
  / job queue.
- Report builder writes CSV / plots under `results/`.
- Randomness is controlled via `numpy.random.Generator`; seeds are set via
  CLI / API payloads for reproducibility.
- Post-hydration validation (`validation.py`) enforces required sections,
  positive numeric ranges, and non-empty option lists for campaigns.

---

## Recent features

### Phase 10 — Live price-model preview

Saved price profiles and inline parameters can be previewed as fan charts
(mean + p05/p95 band + sample paths) via `/profiles/price/{id}/preview`
and `/profiles/price/preview`. The frontend `PriceProfileManager` renders
a live chart with a 500 ms debounce.

### Phase 11 — Tax bonus, stochastic inflation, Excel/PDF export

- **Tax bonus** — a fraction of the upfront CAPEX returned yearly at
  year-end (Italian-style *Detrazione fiscale*). Configurable via
  `economic.tax_bonus = {enabled, fraction_of_investment (0–1),
  duration_years}`.
- **Stochastic inflation** — the legacy scalar `inflation_rate` is
  superseded by `economic.inflation = {mode: 'stochastic'|'deterministic',
  mean, std, min_clip, max_clip}`. In `stochastic` mode the simulator
  samples one annual rate per (path, year) from a Truncated Normal,
  widening the real-return uncertainty band.
- **Excel export** (`/runs/{id}/export/cashflow.xlsx`) — cash-flow table +
  KPI sheet.
- **PDF export** (`/runs/{id}/export/report.pdf`) — KPI cards plus all
  charts (profit / energy / price / inflation) plus the cash-flow table.

Both opt-in: legacy scenarios continue to produce byte-identical results.

### Phase 12 — Background jobs and dashboard polish

- **Job queue** (`/jobs/*`) — long-running Monte Carlo and campaign runs
  execute in a worker pool so the UI can show a floating progress widget
  and redirect when done.
- **Soft-archive** of runs (`PATCH /runs/{id}/archive` ↔ `unarchive`).
  Hidden by default in the Dashboard sidebar; toggle "Mostra archiviati"
  to bring them back.
- **Dashboard filters** — location, date range, archived, paginated.
- **Chart.js zoom plugin** registered globally; toggle nominal ↔ real
  on the profit projection; month-range filter on time series; inline
  cash-flow table.

---

## Testing

```bash
source venv/bin/activate
pytest tests/ -q
```

The suite contains **246 deterministic tests** (~60 s total) covering API
contract, configuration defaults, persistence layer, simulation models
(battery, solar with Markov weather, Monte Carlo, GBM / mean-reverting
prices), result builder, exporters, background job queue, soft-archive,
and CLI grammar. Tests use SQLite in-memory and temporary folders, so
they never mutate `sim_pv.db` or `results/`.

For new tests:

- Add fixtures to `tests/conftest.py`.
- Seed all RNGs with `np.random.default_rng(seed)` for reproducibility.
- Keep individual tests under 5 s (use small `n_mc`).

---

## Validation errors

When configuration validation fails the CLI prints structured errors:

```bash
❌ Errori di configurazione:
  • Missing required section: 'load_profile'
  • energy.pv_kwp must be positive
  • optimization.inverter_options cannot be empty
```

Typical causes:

- Missing required sections (`load_profile`, `solar`, `energy`, `economic`,
  `price`).
- Type mismatch (e.g. `n_mc` / `n_years` passed as strings).
- Empty option lists when running a campaign.
- Out-of-range numeric values (`pv_kwp ≤ 0`, etc.).

Long-running jobs show a tqdm progress bar in TTY contexts:

```
Monte Carlo simulation: 100%|████████████████| 200/200 [00:45<00:00, 4.42path/s]
Evaluating scenarios:    100%|████████████████| 36/36 [02:30<00:00, 4.17s/scenario]
```

In non-interactive environments (CI, scripts) the bar is suppressed.

---

## Troubleshooting

- **Swagger UI not reachable**: verify the backend is running (`docker
  compose ps` or `uvicorn api_main:app --reload`).
- **ImportError for SciPy constants**: `pip install -r requirements.txt`.
- **Battery tests failing on SoC mismatch**: the default battery starts at
  50% SoC; pass `soc_init=0.0` when testing empty packs.
- **Matplotlib cache errors in Docker**: set `MPLCONFIGDIR=/tmp/matplotlib`
  and `XDG_CACHE_HOME=/tmp` (already configured in `docker-compose.yml`).
- **PDF export fails with WeasyPrint error**: install the native libs
  listed under *Requirements*, or use the Docker image.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the living plan. As of 2026-05-28, Phases
1–12 are completed; Phase 13 (documentation sync, this work) is in progress;
Phases 14–17 are planned and cover geolocation + PVGIS + Open-Meteo, a
stochastic thermal model with extreme events, an opt-in detailed electrical
model for inverters and panels (MPPT window, multi-MPPT, derating outside
the operating window), and a stochastic load profile with heat-pump + house
RC thermal model coupling.

See also [CLAUDE.md](CLAUDE.md) for design principles, glossary, and
contribution rules.
