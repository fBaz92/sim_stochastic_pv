# ROADMAP

Piano di evoluzione del simulatore stocastico PV concordato il 2026-05-26.

L'obiettivo finale è una web app in cui l'utente possa **pianificare in un
flusso unico**: luogo di installazione → impianto → carico → mercato
elettrico → investimento → analisi del guadagno, con metriche di rischio
robuste (probabilità di guadagno, break-even atteso con incertezza,
traiettorie simulate del prezzo).

Le fasi sono ordinate per priorità di valore. Le 1 e 2 sono indipendenti.
La 6 chiude e richiede tutte le precedenti.

---

## Fase 1 — Catena di Markov meteo per luogo

**Problema**: oggi la classificazione sunny/cloudy è una Bernoulli iid
giorno-per-giorno (`solar.py:simulate_daily_energy`). Non c'è persistenza,
quindi le bande di produzione sottostimano la varianza che si osserva nei
periodi prolungati di maltempo.

**Punto di vista architetturale**: i parametri meteo sono proprietà del
**sito**, non dello scenario. Vivono in `SolarProfileModel`.

**Deliverable**:

- `SolarProfileModel` esteso con `weather_persistence` (array 12 float,
  uno per mese, `0` = nessuna persistenza/iid, `1` = perfetta persistenza).
- `SolarMonthParams` esteso con `weather_persistence: float = 0.0`.
- `SolarModel.simulate_daily_energy` riscritto come catena di Markov a
  due stati che **preserva per costruzione** la marginale `p_sunny[m]`
  (formula: `p_ss = p_sunny + (1-p_sunny)·persistence`,
  `p_cc = (1-p_sunny) + p_sunny·persistence`).
- Migrazione DB nullable + upsert nei seed.
- UI: la sezione "Luogo" mostra (read-only) p_sunny e persistence mensile.

**Out of scope ora**: catene di Markov a più stati (sereno/variabile/coperto),
correlazione spaziale, dati storici reali.

---

## Fase 2 — Modello prezzo come random walk (GBM)

**Problema**: `EscalatingPriceModel` è una crescita deterministica
compostata con jitter iid clippato. Non è una random walk: gli shock
non persistono e la traiettoria torna sempre verso il trend. La banda
di guadagno cumulato risulta artificialmente stretta sull'orizzonte di
20 anni e questo nasconde il vero rischio dell'investimento.

**Deliverable**:

- Nuova classe `GBMPriceModel(PriceModel)`:
  ```
  log P_{t+1} = log P_t + (μ − σ²/2) Δt + σ √Δt · ε_t
  ```
  con `drift_annual`, `volatility_annual`, `time_step` (`monthly`/`annual`),
  fattori stagionali moltiplicativi opzionali.
- Opzionale: `MeanRevertingPriceModel` (Ornstein-Uhlenbeck) per chi non
  crede alla random walk pura, parametrizzato con velocità di mean
  reversion e livello di equilibrio.
- `EscalatingPriceModel` resta come modalità `deterministic` per backward
  compat.
- Validazione: `volatility_annual >= 0`, `time_step ∈ {monthly, annual}`.
- Default sensati: drift 2.5%/anno, vol 8%/anno (storico residenziale EU
  pre-2021).
- UI: dropdown "Modello prezzo" → form parametri condizionato.

**Out of scope ora**: regimi multipli (GARCH/jump diffusion), correlazione
con il PUN orario, prezzo di vendita differente da quello di acquisto.

---

## Fase 3 — Path di prezzo nei risultati Monte Carlo

**Problema**: il `MonteCarloSimulator` non tiene traccia dei prezzi
generati dal `price_model` per ogni path, solo dei savings in €.
Non si può visualizzare il "fan chart" delle traiettorie di prezzo
simulate.

**Deliverable**:

- `MonteCarloResults` aggiunge `price_paths_eur_per_kwh: np.ndarray`
  di shape `(n_mc, n_months)`.
- `MonteCarloSimulator.run` registra ogni `get_price(y, m)` per path.
- `df_price` (analogo a `df_profit`): media + p5/p95 per mese.
- API: nuovo blocco `data.price` nel response con mean/p05/p95 + un
  campione di 10–20 path (per fan chart, troppi pesano).
- Dashboard: nuovo tab "Prezzo energia" con fan chart + bande.

---

## Fase 4 — Break-even visibile e KPI "investimento conviene?"

**Problema**: il `ResultBuilder` calcola già `break_even_month` per ogni
evaluation (vedi `output/result_builder.py`), ma è solo per
l'ottimizzazione, non per la single-run, e non è esposto nei risultati
API/Dashboard.

**Deliverable**:

- `MonteCarloResults` aggiunge: `break_even_month_per_path`,
  `prob_break_even_within_horizon`, e statistiche aggregate
  (mediana, p5/p95) del break-even tempo.
- Summary API include questi KPI nel response di `/api/analysis`.
- Dashboard: sezione **"Decisione"** in alto con card grandi
  (probabilità di guadagno a fine orizzonte, break-even atteso,
  IRR atteso, NPV mediano).
- Grafico profitto cumulato: linea verticale tratteggiata al break-even
  mediano + area "zona break-even p05–p95" sull'asse x.

---

## Fase 5 — Profilo di carico settimanale

**Problema**: i `LoadProfile` esistenti hanno granularità oraria con
media mensile. L'argomento `weekday` esiste nell'interfaccia ma viene
ignorato dalle implementazioni → no distinzione feriale/weekend.

**Deliverable**:

- Nuova classe `WeeklyPatternLoadProfile(LoadProfile)` che accetta
  una matrice `weekly_pattern_w` di shape `(7, 24)` e la modula sulla
  baseline mensile.
- Preset: `residential_typical`, `smart_worker`, `commuter`.
- UI: editor 7×24 nella sezione "Carico", con dropdown preset e
  toggle "applica pattern settimanale".

**Out of scope ora**: granularità 15 min, stocasticità intra-day,
correlazione carico-temperatura.

---

## Fase 6 — Riorganizzazione UI come wizard

**Problema**: oggi `ScenarioBuilder.svelte` espone tutti i parametri in
un unico form lungo, e gli utenti non hanno una guida sequenziale al
"come si pianifica un impianto". Inoltre, manca completamente la sezione
"Luogo".

**Deliverable**:

- Wizard a step con tab orizzontali (o stepper verticale):
  1. **Luogo di installazione** — dropdown `solar_profile`, preview dati
     meteo (avg daily, p_sunny, persistence) read-only.
  2. **Impianto** — kWp, tilt/azimuth, n_panels, batteria
     (capacity_kwh, cycles_life), n_batteries, inverter (p_ac_max_kw),
     degradazione pannelli/batteria con default sensati e tooltip.
  3. **Profilo di carico** — selettore tipologia
     (ARERA / monthly / home-away / weekly), editor condizionato.
  4. **Mercato elettrico** — modello (deterministic/random_walk/MR),
     base price, drift, volatility, stagionalità.
  5. **Investimento** — investimento totale, n_mc, n_years.
  6. **Esegui analisi** → redirect alla Dashboard sul run creato.
- Dashboard: in alto card "Decisione" (Fase 4), poi tab Guadagno
  (con break-even), Prezzo (fan chart Fase 3), Energia, SoC, SoH.
- Salvare lo scenario come configurazione named anche da wizard.

---

## Fase 7 — Disambiguare "scenario" vs "campagna" nella UI

**Problema**: oggi nella web app c'è confusione tra due concetti che
servono a cose diverse:

- **Scenario** = analisi economica di **una specifica configurazione**
  (kWp, batteria, inverter, profilo, prezzo già scelti) → "questo
  impianto rende?".
- **Campagna** = **esplorazione di design**: provo combinazioni diverse
  per capire quale converge meglio → "qual è il sistema ottimale?".

L'utente entra in `ScenarioBuilder.svelte` aspettandosi di analizzare un
sistema già definito ma trova un mix di campi che oscillano fra
single-point e sweep. `CampaignBuilder.svelte` esiste come pagina
separata ma la nomenclatura "ottimizzazione" lato backend e il "save
configuration" lato API confondono ulteriormente.

**Deliverable**:

- Glossario nel README/CLAUDE.md che fissa la terminologia:
  - "Scenario" = configurazione singola, deterministica nella scelta
    dell'hardware (PV, batteria, inverter selezionati). L'unica
    stocasticità è il Monte Carlo (meteo, carico, prezzo).
  - "Campagna" = sweep su più scenari per ricerca dell'ottimo (sostituisce
    completamente la dicitura "optimization" lato UI).
- Pagine UI: `ScenarioBuilder` accetta UNA singola configurazione hardware
  (un inverter, una batteria, un numero di pannelli) — niente liste.
  `CampaignBuilder` accetta liste di alternative.
- Endpoint API rinominati lato schema/route docs (mantenendo backward
  compat su path attuali con `deprecated: true`):
  `/api/analysis` → "Run scenario", `/api/optimization` → "Run campaign".
- Sidebar/Navbar: rinominare voci di menu coerentemente
  ("Nuovo scenario" / "Nuova campagna").
- Dashboard distingue visivamente i due tipi di run con badge diversi
  (oggi: `analysis` / `optimization`).

**Out of scope ora**: rinomina dei tipi nel DB (`SavedConfigurationModel.config_type`
resta `scenario` / `optimization` per non rompere i dati esistenti);
la traduzione nei messaggi user-facing è sufficiente.

---

## Fase 8 — Load profile come oggetto completo nel DB

**Problema**: due bug strettamente correlati.

1. **Bug visibilità**: il `LoadProfileManager.svelte` permette di
   salvare profili di carico nel DB, ma il `ScenarioBuilder.svelte`
   non li consuma — l'utente li ridefinisce inline a ogni scenario.
   Backend già pronto (`persistence/hydration.py:116` gestisce
   `load_profile_id`), manca solo il selettore lato UI.
2. **Struttura concettuale errata**: oggi un "load profile" salvato è
   solo un pattern (`monthly_w` o `monthly_24h_w`). Ma home/away sono
   due *attributi della stessa persona/utenza*: come consumo quando
   sono in casa e come consumo quando sono via. Sono proprietà del
   profilo di carico, non dello scenario. Lo scenario decide solo
   *quanti* giorni l'utente è home/away (mensilmente).

**Deliverable**:

- Schema DB esteso: `LoadProfileModel.data` accetta una nuova forma
  con due sotto-profili:
  ```json
  {
    "kind": "home_away",
    "home": { "monthly_24h_w": [[…24…], …12…] },
    "away": { "monthly_24h_w": [[…24…], …12…] }
  }
  ```
  Le forme legacy (`monthly_w`, `monthly_24h_w` a livello root) restano
  supportate per backward compat e interpretate come "stesso profilo
  home e away".
- `LoadProfileManager.svelte` editor a due tab "Quando sono a casa" /
  "Quando sono via" con il pattern 12×24 ciascuno.
- `ScenarioBuilder.svelte` aggiunge dropdown "Profilo di carico" che
  legge da DB, e i campi `min_days_home`/`max_days_home` mensili
  diventano parte dello scenario (non del profilo). UI mostra
  preview read-only del profilo selezionato.
- `scenario_builder.build_default_load_profile` riconosce la nuova
  forma `home_away` e costruisce un `HomeAwayLoadProfile` dai due
  sotto-profili.
- Migrazione: nessun campo DB nuovo (il payload `data` è già JSON), ma
  un test che verifica round-trip della forma `home_away`.
- 4-5 test (creazione DB, lettura via API, hydration corretta, run
  end-to-end con `load_profile_id`).

**Out of scope ora**: variabilità intra-giornaliera del carico
(stocasticità sul singolo profilo home), profili settimanali (resta
in Fase 5).

---

## Fase 9 — Modalità "semplificata" per dimensionamento stringhe + inverter

**Problema**: la `CampaignBuilder` chiede all'utente di specificare il
**numero di pannelli per stringa** come parametro di sweep. Ma:

1. Lato fisica, oggi **non c'è alcun calcolo MPPT/tensioni di stringa**
   nel codice (`InverterAC` espone solo `p_ac_max_kw` e cap DC; nessuna
   `v_oc`, `v_mpp`, range MPPT). Quindi quel parametro influenza solo
   il CAPEX (costo extra del pannello in più) e mai la produzione.
2. L'utente normalmente non vuole pensare alle stringhe quando fa una
   simulazione economica esplorativa — vuole indicare quanti pannelli
   in totale, o la potenza DC desiderata.

**Deliverable**:

- **Documentare l'assenza di logica MPPT** nei docstring di `InverterAC`,
  `PanelOption`, `ScenarioOptimizer`: "il modello attuale non calcola
  tensioni di stringa; vedi Fase 9-bis (futura) per integrazione MPPT
  detail".
- Nuova modalità **"simplified sizing"** in `ScenarioBuilder` (e
  parametri compatibili in `scenario_builder.py`):
  - Input utente: `n_panels` totali oppure `target_dc_overcapacity_pct`
    (default 20%): "voglio almeno il 20% di sovrappotenza DC rispetto
    all'inverter".
  - Calcolo automatico: dato `panel.power_w`, `inverter.p_ac_max_kw`
    e l'overcapacity desiderata, sceglie il numero minimo di pannelli
    che rispetta il vincolo `n_panels * power_w >= p_ac_max * (1 + overcap)`.
  - `n_panels_per_string` non viene mostrato all'utente in modalità
    simplified (resta gestito internamente come "tutti su una sola
    stringa", coerente con l'assenza di logica MPPT).
- In `CampaignBuilder` lo sweep sui pannelli usa la stessa modalità
  semplificata per default; l'utente avanzato può sbloccare un toggle
  "advanced sizing" che riespone `n_panels_per_string` (per quando in
  futuro arriverà il modello MPPT).
- Documento `docs/electrical_simplifications.md` che elenca cosa il
  modello attualmente NON considera (tensione stringa, MPPT,
  string-mismatch, derating thermal-dipendente) e perché.

**Out of scope ora**:

- Implementazione vera del modello MPPT con datasheet del pannello
  (single-diode model in `pv_model.py` esiste ma non è collegato al
  simulatore). Verrà schedulata come **Fase 9-bis** quando l'utente
  vorrà accuratezza fisica.
- Sweep su numero di MPPT tracker dell'inverter.

---

## Fase 11 — Bonus fiscale, inflazione stocastica, export Excel/PDF, rename Campagna→Design

**Problema**: tre lacune significative nello strumento di valutazione economica.

1. **Bonus fiscale assente.** L'utente italiano ha tipicamente diritto a una
   detrazione del 50% (o più) del CAPEX, erogata su 10 anni. Oggi non è
   modellabile: IRR e break-even sono sistematicamente sottostimati di 30–40
   punti percentuali rispetto alla realtà fiscale.
2. **Inflazione deterministica.** `EconomicConfig.inflation_rate=0.025` è uno
   scalare fisso. In EU negli ultimi 20 anni l'inflazione ha oscillato fra 0%
   e 8%; la banda del profit reale (`p05–p95`) è artificialmente stretta.
3. **Nessun export.** L'utente non può portare via i risultati: né tabelle
   per analisi su Excel, né report per condividere con consulenti.

Inoltre, "Campagna" continua a confondere in UI. L'utente capisce meglio
"Design" come *esplorazione economica al variare di CAPEX (configurazioni
hardware) e OPEX (parametri operativi)*.

**Deliverable**:

- `TaxBonusConfig` dataclass + integrazione opzionale in `EconomicConfig`.
  Importo annuo = `investment_eur × fraction / duration_years`, pagato a
  fine anno (mesi 11, 23, 35, …). Truncato graziosamente se
  `duration_years > n_years`.
- `InflationConfig` dataclass con `mode='deterministic' | 'stochastic'`.
  In modalità stocastica estrae `n_mc × n_years` tassi annuali da
  Normale(mean, std) clippata in [min_clip, max_clip]. Il legacy scalar
  `inflation_rate` resta come fallback retrocompat: identico byte-per-byte
  in deterministico.
- `inflation_factors_paths` shape `(n_mc, n_months)` pre-campionati prima
  del loop MC; `bonus_per_month` vettore sparso sommato a `monthly_savings_eur`
  prima di `cumsum`/IRR (entra automaticamente nei `profit_cum`, `profit_cum_real`
  e nei cashflows IRR).
- `MonteCarloResults` esteso con `inflation_annual_rates_paths`,
  `df_inflation`, `bonus_per_month_eur`, `tax_bonus_total_eur`.
- `application._build_inflation_plot_payload` (fan chart inflazione) e
  `_build_cashflow_table_payload` (medi mensili) emessi in
  `summary.plots_data`. `tax_bonus_total_eur` aggiunto al top-level.
- `scenario_builder.build_default_economic_config` legge i due sotto-blocchi
  da JSON; `validation._validate_tax_bonus` e `_validate_inflation`
  enforce dei limiti (fraction in [0,1], std≥0, min_clip≤max_clip).
- Schemi Pydantic `TaxBonusSchema` e `InflationSchema` + campo
  `tax_bonus_total_eur` in `AnalysisResponse`.
- **Export Excel** (openpyxl): `GET /api/runs/{id}/export/cashflow.xlsx`
  → workbook con foglio "Cash flow medio" (vettori mensili) e "KPI" (decision metrics).
- **Export PDF** (WeasyPrint + Jinja2 + matplotlib): `GET /api/runs/{id}/export/report.pdf`
  → report multi-pagina con KPI Decisione, fan chart profitto/energia/prezzo/inflazione
  e tabella cash flow. Degrado grazioso su run pre-Fase-11.
- Frontend `ScenarioBuilder.svelte` e `CampaignBuilder.svelte`: sezioni
  opzionali "Bonus fiscale" e "Inflazione" nello step Investimento
  (conversione UI % ↔ payload 0–1).
- Frontend `Dashboard.svelte`: nuovo tab "Inflazione" (fan chart con
  fattore cumulativo), card KPI "Bonus fiscale totale", pulsanti
  "Scarica Excel" e "Scarica PDF" sopra la sezione Decisione.
- Frontend `ResultsChart.svelte`: icona overlay Download in alto a destra
  di ogni grafico, basata su `chart.toBase64Image()` (Chart.js nativo).
- **Rinomina UI Campagna → Design**: route `/design` (più alias `/campaign`
  per retrocompat dei bookmark), label Navbar, titoli e badge, blocco
  didattico CAPEX/OPEX in cima alla pagina Design. NON cambiano:
  `config_type='campaign'` nel DB, API path `/api/campaigns/...`,
  variabili JS (`selectedSavedCampaignId`, `runSavedCampaign`, ecc.),
  nome file `CampaignBuilder.svelte` (CLAUDE.md §Glossario).
- Aggiunti `openpyxl`, `weasyprint`, `Jinja2` a `requirements.txt`; deps
  native pango/cairo/gdk-pixbuf-2.0 nel `Dockerfile.backend`.

**Out of scope ora**:
- Bonus tax-bracket-dependent (Superbonus 110%, scaglioni IRPEF, cap di
  spesa). Resta un flat percent × n_years.
- Inflazione path-dependent AR(1) o regime switching (resta Normale iid
  per anno).
- Export CSV separato (l'Excel è già un superset utile).
- Localizzazione del PDF in lingue diverse dall'italiano.
- Confronto side-by-side di più run nello stesso PDF.

---

## Fase 10 — Preview traiettorie prezzo nella sezione Database

**Problema**: l'utente sceglie i parametri di un `price_profile`
(es. GBM con drift 2.5%, vol 15%) senza alcun feedback visivo. Non sa
se 15% è "tanto" o "poco" finché non lo prova in uno scenario completo.

Idem per chi guarda un profilo già salvato: cliccando su un
`PriceProfileManager` item non si vede nulla. La fan chart di Fase 3
esiste solo dentro Dashboard, dentro un run completo.

**Deliverable**:

- Nuovo endpoint `GET /api/profiles/price/{id}/preview` (e
  `POST /api/profiles/price/preview` per parametri inline non
  ancora salvati) con query string `?n_paths=20&n_years=20&seed=42`
  che ritorna lo stesso schema del blocco `plots_data.price` (mean,
  p05, p95, sample_paths).
- Implementazione: helper riusabile `simulate_price_preview(price_model, n_years, n_paths, seed)`
  che istanzia il modello, fa `n_paths` reset indipendenti e raccoglie
  i path. Mette in evidenza il legame Fase 2 ↔ Fase 3.
- `PriceProfileManager.svelte`: quando l'utente clicca su un profilo
  salvato apre un pannello laterale con il fan chart (riuso del
  componente Chart.js già usato in Dashboard).
- Nuovo componente "live preview" nel form di creazione/edit:
  parametri → debounce 500 ms → fan chart aggiornata. Bottone
  "rigenera con nuovo seed" per esplorare visualmente la varianza.
- 2 test: endpoint risponde con schema corretto; preview con vol=0
  collassa a una linea.

**Out of scope ora**: confronto side-by-side di più modelli prezzo,
calibrazione automatica drift/vol da dati storici importati.

---

## Fase 13 — Sync README, CLI e glossario (allineamento documentale)

**Problema**: il README riflette uno stato precedente alla maggior parte
delle fasi consegnate. Esempi concreti:

- Cita una test suite "di 15 test" mentre è a 225/225.
- Documenta comandi CLI `optimize save`/`optimize run` (riga 159-160)
  che non esistono — il CLI espone `optimization save`/`optimization run`
  (`cli.py:158-173`).
- "Next steps" elenca lavori già fatti (collegare `PVModelSingleDiode`
  è un open item tracciato meglio in `docs/electrical_simplifications.md`
  e nella futura Fase 16).
- Manca tutto: glossario Scenario/Campagna/Run, wizard a 6 step (Fase 6),
  fan-chart prezzi (Fase 3), KPI Decisione (Fase 4), export PDF/Excel
  (Fase 11), job queue + soft-archive (Fase 12).
- La CLI non riflette il rename UI Campagna→Design (Fase 11): il
  sottocomando `optimization` confonde rispetto al glossario.

**Deliverable**:

- README riscritto: sezioni *Repository layout*, *Usage modes*,
  *FastAPI endpoints*, *Testing* allineate al codice attuale; sezione
  *Next steps* sostituita da rimando esplicito al ROADMAP.
- Tabella endpoints aggiornata con `/api/profiles/{solar,price,load}`,
  `/api/jobs/*`, archive endpoints della Fase 12, preview prezzi della
  Fase 10.
- Glossario sintetico in testa al README con link a `CLAUDE.md §Glossario`.
- CLI: aggiunta del sottocomando `campaign` come alias preferito di
  `optimization` (entrambi continuano a funzionare per backward compat),
  e di `design` come ulteriore alias UI-friendly. Default invariati,
  zero break.
- Aggiunta sezione "Avvio dev rapido" con i 3 comandi minimi
  (venv + uvicorn + npm dev) per chi entra da zero senza Docker.
- Aggiornamento `CHANGELOG.md` con la voce della modifica.

**Out of scope ora**: documentazione OpenAPI estesa (resta su Swagger
autogenerato), tutorial passo-passo "il mio primo scenario" (resta
materiale di onboarding separato).

---

## Fase 14 — Geolocation + PVGIS + Open-Meteo per il sito

**Problema**: oggi creare un `SolarProfileModel` per una nuova località
significa cercare a mano i 12 valori di `avg_daily_kwh_per_kwp` e
`p_sunny` su PVGIS, copiarli in un seed JSON, riavviare. Per un utente
non specialista è insormontabile; per uno specialista è comunque lento
e error-prone. Inoltre lo step "Luogo" del wizard non mostra dove si è
geograficamente — non c'è alcuna mappa.

**Deliverable**:

- Nuovo modulo `sim_stochastic_pv/external/` con tre client puri
  (no side effect, cache opzionale su tabella `external_cache`):
  - `nominatim_client.py` — geocoding nome → (lat, lon, display_name).
    Rate limit 1 req/s come da policy OSM, User-Agent identificativo.
  - `pvgis_client.py` — fetch `MRcalc` (PVGIS v5.2) per
    `(lat, lon, tilt, azimuth)` → 12 valori mensili di radiazione
    `H(i)_m`; conversione in `kWh/kWp/day` con PR (Performance Ratio)
    di default 0.78 esposto come parametro.
  - `openmeteo_client.py` — fetch normali climatiche (ERA5 30y) e
    percentili mensili di T_max/T_min. Preparatorio per Fase 15.
- Nuovo endpoint `POST /api/profiles/solar/from_location`: input
  `{lat, lon, tilt, azimuth, name, pr_default}` → salva un
  `SolarProfileModel` con valori da PVGIS, restituisce il record.
- Nuovo endpoint `POST /api/external/geocode` (wrapper Nominatim, con
  cache su DB) per non far chiamare l'API direttamente dal browser
  e per evitare di sforare il rate limit.
- Frontend wizard step "Luogo":
  - Campo ricerca località con autocomplete (Nominatim).
  - Mappa **Leaflet + tile OpenStreetMap** sotto, marker draggable per
    fine-tuning di lat/lon. Niente API key, niente costi.
  - Bottone "Importa profilo da PVGIS" che pre-popola la tabella
    read-only mensile.
  - Pannello "Clima locale" con tmax/tmin mensili (read-only, fonte
    Open-Meteo) — preparatorio per Fase 15.
- 4–6 test che mockano le risposte HTTP (no rete in CI): geocode
  parsing, PVGIS conversion math, Open-Meteo aggregation, end-to-end
  `from_location → solar profile` salvato in DB.

**Out of scope ora**: cache TTL avanzata sui client esterni (basta il
salvataggio in DB del profilo finale), supporto a multiple sorgenti
(es. NREL/NASA POWER), copertura PVGIS oltre EU+Mediterraneo (è già
il dominio di calibrazione naturale del progetto).

---

## Fase 15 — Modello termico stocastico con eventi estremi

**Problema**: il simulatore non ha alcuna nozione di temperatura
ambientale. Questo blocca due cose:

1. Modello elettrico realistico (Fase 16) — perché il derating
   dell'inverter e la finestra MPPT dipendono da T_cell, che a sua
   volta dipende da T_ambient (NOCT).
2. Modello di carico HVAC (Fase 17) — perché il consumo per
   riscaldamento/raffrescamento scala con HDD/CDD.

Inoltre, le decisioni di design ottimale ignorano completamente gli
eventi estremi (ondate di calore, gelate) che possono cambiare la
valutazione di rischio di un impianto su 20 anni — es. "gennaio gelido
a -10°C → V_oc dei pannelli sale ~12% → V_stringa eccede V_dc_max →
inverter spento per ore".

**Deliverable**:

- Nuova classe `ThermalModel` in `simulation/thermal.py`:
  - **Stagionalità deterministica**: media giornaliera tramite armoniche
    annuali calibrate dai dati Open-Meteo (Fase 14):
    `T_mean(d) = a₀ + a₁·cos(2π·d/365) + a₂·sin(2π·d/365)`.
    Curva diurna a partire da `tmax`/`tmin` con modello sinusoidale.
  - **Residuo stocastico**: AR(1) con `φ` (autocorrelazione lag-1) e
    `σ` (innovazioni). Default tarati su clima italiano
    (`φ ≈ 0.80`, `σ ≈ 2.5 °C`).
  - **Code estreme**: Generalized Pareto Distribution sopra soglia
    mensile (POT). Parametri `(threshold, shape ξ, scale σ_GPD)` mensili
    separati per code superiori (ondate di calore) e inferiori (gelate).
  - **Trend climatico** (opt-in): drift lineare annuo `δT/anno`
    (default 0.0 = no trend; valore tipico EU 0.03 °C/anno).
- Nuova dataclass `ThermalMonthParams` con i parametri sopra elencati.
- Nuovo modello DB `ClimateProfileModel(name, location_name, lat, lon,
  source, monthly_params: JSON)` separato da `SolarProfileModel` per
  pulizia concettuale (un profilo solare può vivere senza un climate
  profile, e viceversa).
- `Scenario` referenzia opzionalmente `climate_profile_id`; assenza =
  nessun modello termico → simulazione invariata (default backward
  compat).
- Helper `simulate_temperature_paths(model, n_days, rng) → np.ndarray
  (24·n_days,)` che produce anche un mini-report sugli eventi estremi
  registrati (n. ondate di calore, durata media, T_peak, ecc.).
- Integrazione automatica con Fase 14: nuovo endpoint
  `POST /api/profiles/climate/from_location` calibra il modello dai
  30 anni di Open-Meteo (fit armoniche + φ + σ + parametri GPD).
- **Preview temperatura nel wizard "Luogo"** (mutua il pattern della
  Fase 10 sul prezzo): nuovo endpoint
  `GET /api/profiles/climate/{id}/preview?n_paths=50&n_years=1&seed=42`
  (e `POST .../preview` per parametri inline non ancora salvati).
  Risposta: `{days, mean_t_c, p05_t_c, p95_t_c, sample_paths}` con
  ~50 traiettorie giornaliere su un anno. Lato frontend, nello step
  Luogo del wizard appare un fan chart che mostra la media annuale,
  banda p05–p95 e 50 path simulati semi-trasparenti — così l'utente
  *vede* il clima del posto e si rende conto se gli eventi estremi
  sono già "dentro" il modello.
- 8–10 test: stazionarietà della media giornaliera entro tolleranza,
  autocorrelazione lag-1 della simulazione coerente col parametro,
  mean-reversion della media annuale, tail empirica della GPD coerente
  con `(ξ, σ_GPD)`, riproducibilità da seed, endpoint preview ritorna
  schema corretto.

**Out of scope ora**: temperatura del suolo, umidità, ventosità
(sufficienti per inverter/HVAC sono solo T_ambient + irradianza).
Modelli regional climate / statistical downscaling RCP. Calibrazione
GPD su dati custom dell'utente — resta su default + override manuale.

---

## Fase 16 — Modello elettrico inverter + pannelli (opt-in)

**Problema**: oggi `InverterAC` (`simulation/inverter.py`) ha solo
`p_ac_max_kw` e cap DC. Il numero di MPPT, la finestra di tensione, le
tensioni V_oc/V_mpp dei pannelli e i coefficienti termici non esistono.
Risultato: lo splitting in stringhe non ha effetto fisico, e scenari
critici noti — gelida giornata di gennaio (V_oc fuori scala) o estate
torrida (V_mpp sotto V_mppt_min) — sono invisibili al modello. Non si
possono fare design choices realistiche.

`PVModelSingleDiode` esiste già in `simulation/pv_model.py` ma è
scollegato dal simulatore. La Fase 9 ha tracciato la lacuna in
`docs/electrical_simplifications.md` come "Fase 9-bis"; questa Fase 16
chiude quel debito.

**Deliverable**:

- Estensione `InverterAC` con campi nullable (default `None` =
  comportamento attuale invariato):
  - `v_dc_min_v`, `v_dc_max_v` — finestra operativa assoluta
    (fuori = shutdown).
  - `v_mppt_min_v`, `v_mppt_max_v` — finestra MPPT a piena potenza.
  - `n_mppt_trackers` (default 1), `i_dc_max_per_mppt_a`.
- Estensione `PanelModel.specs` (campi nullable):
  - `v_oc_stc_v`, `v_mpp_stc_v`, `i_sc_stc_a`, `i_mpp_stc_a`,
    `n_cells_series`.
  - Coefficienti termici: `beta_voc_pct_per_c` (tipico -0.30),
    `gamma_pmax_pct_per_c` (tipico -0.38), `noct_c` (tipico 45).
- Nuovo flag scenario `electrical.mode ∈ {"off","mppt_window"}`
  (default `"off"`):
  - `"off"`: comportamento attuale, niente parametri richiesti.
  - `"mppt_window"`: usa T_cell da `T_ambient` (Fase 15) + NOCT,
    calcola V_string per ora, applica:
    - finestra MPPT OK → no loss;
    - V_string fuori finestra MPPT ma dentro finestra DC →
      derating `P = P_mp · (V_target/V_string)^k`, `k≈0.5`
      (parametrizzabile per tarare su dati reali);
    - V_string fuori finestra DC → **shutdown** (0 W, conteggio ore
      sotto `hours_dc_overvoltage` come **flag di rischio hardware**).
- **Requisito dati per `mppt_window`**: il modello richiede che il
  pannello selezionato abbia `v_oc_stc_v`, `v_mpp_stc_v`,
  `n_cells_series`, `beta_voc_pct_per_c`, `gamma_pmax_pct_per_c`,
  `noct_c` valorizzati, e che l'inverter abbia
  `v_mppt_min_v`, `v_mppt_max_v`, `v_dc_min_v`, `v_dc_max_v`. Se uno
  qualunque manca, `validation.py` invalida lo scenario con un
  messaggio che elenca esplicitamente i campi mancanti (no fallback
  silenzioso). In altre parole: l'analisi superficiale si fa sempre,
  ma l'analisi dettagliata richiede un catalogo completo.
- Multi-MPPT: lo scenario può specificare
  `pv_strings: [{tilt, azimuth, n_panels, mppt_id}, ...]`. Default =
  una stringa unica = comportamento attuale.
- Nuovo blocco `summary.electrical` (solo se mode != off):
  `hours_dc_overvoltage_per_year`, `hours_outside_mppt_per_year`,
  `peak_v_string_v`, `min_v_string_v`. KPI esposti nel Dashboard.
- UI: toggle "Modello elettrico dettagliato" nello step Impianto del
  wizard (gated, default off). I form `InverterManager` /
  `PanelManager` espongono i campi elettrici solo se l'utente attiva
  il flag (per non confondere chi vuole il default semplice).
- Catalogo seed esteso con 3–4 pannelli e 3–4 inverter con dati
  elettrici realistici (es. Longi LR5-72HPH-540M, JA Solar JAM72S30,
  Fronius Primo 5.0, SMA Sunny Boy 5.0, Huawei SUN2000-5KTL).
- 10–12 test: mode=off byte-identico, calcolo T_cell, V_string(T)
  con coefficienti reali, shutdown corretto su V_oc > V_dc_max,
  derating monotono fuori finestra MPPT, multi-MPPT con orientamenti
  diversi somma correttamente, `hours_dc_overvoltage > 0` in caso
  patologico (gennaio gelido + sole pieno).

**Out of scope ora**: mismatch tra moduli della stessa stringa
(assumiamo stringhe uniformi), shading dinamico, inverter ibridi con
secondo inverter dedicato alla batteria AC-coupled, calibrazione
automatica del parametro `k` di derating da dati misurati reali.
Modello a diodo singolo per IV-curve esatta (`PVModelSingleDiode`
resta scollegato): se servirà davvero accuratezza fisica oltre il
modello MPPT-window, sarà una **Fase 16-bis** dedicata.

---

## Fase 17 — Carico stocastico con accoppiamento termico

**Problema**: i `LoadProfile` attuali (`arera`, `monthly`,
`monthly_24h`, `weekly`, `home_away`) sono pattern deterministici
scalati. Mancano due dimensioni di realismo che pesano sulla
valutazione economica:

1. **Variabilità intra-day**: il 1° gennaio e il 31 gennaio hanno la
   stessa identica curva oraria. In realtà giornate simili oscillano
   di ±20–30% sulla potenza istantanea per via di sequenze umane
   casuali (orario doccia, accensione caldaia, lavatrice).
2. **Accoppiamento col meteo**: il consumo HVAC (pompa di calore in
   inverno, split AC in estate) è proporzionale agli HDD/CDD del
   giorno. Senza modello termico (Fase 15), questo non esiste oggi.

**Deliverable**:

- Nuovo `StochasticLoadProfile(LoadProfile)` decoratore di un profilo
  base (wrapper Pattern):
  - Rumore moltiplicativo orario `~ LogN(0, σ_log)`, σ_log default
    0.20 (≈ ±20% 1-sigma).
  - AR(1) intra-day sul log-rumore con `φ_intra_day = 0.5` per evitare
    white noise non realistico.
  - Parametri esposti nel JSON scenario:
    `load_profile.stochastic = {enabled, sigma_log, phi_intra_day}`.
    Default `enabled = false` → comportamento attuale.
- Nuovo blocco scenario `thermal_load` (opt-in, richiede
  `climate_profile_id` valorizzato — Fase 15). Modello fisico
  consapevolmente semplificato ma onesto:

  - **Modello termico della casa al 1° ordine RC** (singolo parametro
    fisico principale): l'edificio è una capacità termica `C` (kWh/°C)
    con coefficiente di dispersione `UA` (kW/°C) verso l'esterno.
    Bilancio termico:

        C · dT_in/dt = UA · (T_out − T_in) + P_thermal(t)

    dove `P_thermal` è positiva in heating e negativa in cooling.
    Il parametro fisico **principale** è `UA`, esposto via 3 preset
    user-friendly:
    - `"poor"`: ~2.5 W/°C/m² (case anni '60–'70 non riqualificate);
    - `"standard"`: ~1.5 W/°C/m² (case anni '90, isolamento medio);
    - `"good"`: ~0.8 W/°C/m² (NZEB / classe A).

    L'utente specifica `floor_area_m2` (default 100) e il preset; `UA`
    viene calcolato in automatico. `C` deriva da `floor_area_m2` con
    una costante di stima 0.05 kWh/°C/m² (capacità termica di una
    massa interna media), esposta come parametro avanzato.

  - **Pompa di calore** caratterizzata da:
    - `cop_heating` (default 3.5) e `cop_cooling` (default 3.0).
      Costanti per la prima passata; campo opzionale
      `cop_heating_curve` (lookup `[(T_out, COP), ...]`) per la
      dipendenza da T_amb (i COP veri scendono col freddo).
    - `p_elec_max_kw`: assorbimento elettrico massimo (es. 3 kW). La
      potenza elettrica richiesta è cappata a questo valore.

  - **Setpoint comfort**:
    - `t_setpoint_heating_c` (default 20 °C) — soglia sopra la quale
      si attiva l'heating.
    - `t_setpoint_cooling_c` (default 26 °C) — soglia sotto la quale
      si attiva il cooling. La dead-band tra le due è "no HVAC".
    - `t_setpoint_away_c` (opzionale): se l'utente non è a casa,
      l'HVAC è disattivato. Se valorizzato, mantiene un setback
      ridotto invece di spegnersi del tutto (utile per non far gelare
      la casa).

  - **Calcolo orario del carico elettrico HVAC** (steady-state
    "calcolo a ritroso" — non un PID dinamico, ma l'inverso istantaneo
    del modello RC):
    1. Se l'utente non è a casa quell'ora (derivato da
       `min_days_home`/`max_days_home` + `t_setpoint_away_c`
       eventualmente non valorizzato) → `P_elec_hvac(h) = 0`.
    2. Modalità: heating se `T_out(h) < t_setpoint_heating_c`,
       cooling se `T_out(h) > t_setpoint_cooling_c`, altrimenti off.
    3. Potenza termica richiesta in steady-state per mantenere il
       setpoint dato T_out:
       `P_thermal_req = UA · (T_setpoint − T_out(h))`
       (segno positivo in heating, negativo in cooling).
    4. Potenza elettrica: `P_elec = |P_thermal_req| / COP`
       (con COP scelto in base alla modalità e, se la curva esiste,
       interpolato da `cop_heating_curve` a T_out).
    5. Cap a `p_elec_max_kw`. Quando viene cappata si registra l'ora
       come **comfort breach** (T_in non raggiungerebbe il setpoint
       a regime).

  - **Integrazione dinamica RC opzionale** (`thermal_load.dynamic = true`,
    default `false`): invece dello steady-state, si risolve l'ODE
    con Eulero implicito su passo orario, permettendo lag nelle
    transizioni day↔night. Marginale per l'effetto economico ma utile
    per chi vuole valutare strategie di pre-riscaldamento.

- Si somma al carico base del `LoadProfile` esistente,
  indipendentemente dal tipo (arera/monthly/weekly/home_away). Il
  modello HVAC è additivo, non sostitutivo.

- UI: nel wizard step "Carico" due toggle aggiuntivi:
  - "Variabilità giornaliera del consumo" (gated, σ_log + φ).
  - "Pompa di calore / HVAC con modello casa" (gated, richiede
    profilo climatico Fase 15). Accordion con:
    - Radio 3 preset isolamento (`poor` / `standard` / `good`) +
      tooltip esplicativo con W/°C/m² e bolletta tipica.
    - `floor_area_m2`.
    - Parametri pompa: cop_heating, cop_cooling, p_elec_max_kw.
    - Setpoint heating, cooling, away (away opzionale).

- KPI nuovi nel `summary.thermal` (solo se `thermal_load` attivo):
  `hvac_kwh_annual_mean`, `hvac_share_of_total_load_pct`,
  `comfort_breach_hours_per_year`, `p_elec_hvac_peak_kw`
  (utile per dimensionamento inverter/contratto in kW).

- 10 test: media long-run preservata dalla stocasticità (entro 1%),
  AR(1) decay lag-k, HVAC zero quando COP→∞ o UA→0, HVAC positivo
  nei mesi giusti, scaling lineare con `floor_area_m2`, comfort
  breach attivato in scenario con `p_elec_max_kw` insufficiente,
  away setpoint rispettato quando l'utente non è a casa,
  autoconsumo cresce in estate quando il cooling è attivo (carico
  correlato al solare).

**Out of scope ora**: modello event-based con appliance discreti
(lavatrice/forno/auto EV) — schedulabile come **Fase 17-bis** se
emerge l'esigenza. Pattern di assenza (vacanze, weekend lunghi)
generati stocasticamente (per ora restano sliders giorni-home/away
mensili). Demand response e tariffe orarie/PUN. Modello multi-zona
della casa (l'edificio è single-zone). Acqua calda sanitaria
(separabile dalla pompa di calore in futuro se serve).

---

## Fase 17-bis — Carico event-based con appliance discreti

**Problema**: la Fase 17 ha aggiunto al carico due livelli di realismo —
varianza intra-day LogN/AR(1) e contributo HVAC continuo — ma il consumo
domestico reale non è una curva *liscia* moltiplicata per un fattore
stocastico. È una sequenza di **eventi discreti**: la lavatrice si
accende per 90 minuti a 1.5 kW, l'auto EV si attacca alla colonnina per
8 ore a 2.3 kW, il forno cuoce per 45 minuti a 2.5 kW. La firma oraria
(spike + ritorno alla baseline) è bimodale per costruzione, e nessun
moltiplicatore log-normale può riprodurla.

Questo impatta tre conti economici reali del simulatore:

1. **Autoconsumo PV** — quando un evento da 2 kW capita alle 13:00 (sole
   pieno) l'autoconsumo è ~100 %; lo stesso evento alle 21:00 (sole
   zero) si imputa interamente alla rete. Il LogN della Fase 17, per
   costruzione mean-1, *diluisce* questi picchi su tutte le ore e
   sottostima entrambi i casi estremi.
2. **Dimensionamento batteria** — la batteria viene caricata e scaricata
   in base ai picchi orari, non ai valori medi. Il `peak_load_kw` di
   un MC path con appliance discreti è 2–5 volte quello calcolato dal
   profilo liscio + moltiplicatore stocastico, e questo si trasla in
   un dimensionamento *molto* diverso (cap charge/discharge rate,
   numero di cicli, fine vita).
3. **Strategie di scheduling** — la versione "smart PV" (l'utente avvia
   lavatrice/lavastoviglie/auto EV durante le ore di sole) cambia
   completamente il payback dell'impianto rispetto a "naive timer"
   (eventi distribuiti uniformemente nelle ore consentite). Senza un
   modello a eventi non è possibile *confrontare* queste due strategie
   nello stesso scenario.

**Deliverable**:

- Nuovo modulo `sim_stochastic_pv/simulation/load_profiles/appliances.py`:
  - `ApplianceEvent(dataclass frozen)` con campi:
    - `name: str` — identificatore leggibile (per i KPI).
    - `p_kw: float` — potenza istantanea durante l'evento (kW).
    - `duration_hours: float` — durata, *non* arrotondata a un'ora
      intera (es. 0.75 h per il forno).
    - `monthly_frequency: list[float] (12)` — eventi attesi al mese
      (12 valori, calibrabili stagionalmente: es. lavatrice 12/mese
      costante, condizionatore-cucina 0 d'inverno).
    - `allowed_hours: list[int]` — sottoinsieme di `range(24)` delle
      ore in cui l'evento può *iniziare* (lavatrice prefere 9-18,
      forno 11-13 ∪ 18-20).
    - `hour_of_day_weights: list[float] (24)` — pesi morbidi sull'ora
      di partenza all'interno di `allowed_hours`. Quando ``None`` si
      usa una distribuzione uniforme.
    - `schedule_mode: Literal["naive_timer", "smart_pv"]` (default
      `"naive_timer"`). In modalità `"smart_pv"` i pesi vengono
      riponderati dalla curva oraria di produzione PV deterministica
      (`SolarModel.hourly_shape`) — gli eventi vengono biasati verso
      le ore di sole.
  - `ApplianceCatalog` con preset realistici per il mercato residenziale
    italiano (potenze e durate medie da indagini ISTAT/RSE):
    - `washing_machine` (1.5 kW × 1.5 h, ~12/mese, allowed 9-18)
    - `dishwasher` (1.2 kW × 1.0 h, ~15/mese, allowed 13-22)
    - `oven` (2.5 kW × 0.75 h, ~8/mese, allowed 11-13 ∪ 18-20)
    - `dryer` (2.2 kW × 1.0 h, ~6/mese, prefere 10-14)
    - `ev_charger_slow` (2.3 kW × 8 h, ~20/mese, allowed 22-06 in
      naive_timer / 9-16 in smart_pv)
    - `ev_charger_fast` (7.4 kW × 2.5 h, ~15/mese, allowed 22-06)
    - `induction_cooktop` (1.8 kW × 0.5 h, ~30/mese, allowed 11-13
      ∪ 19-21)
    - `dhw_heat_pump_cycle` (1.8 kW × 0.5 h, ~30/mese, allowed 7-9 ∪
      17-19) — vita autonoma rispetto al modello HVAC della Fase 17,
      che resta dedicato al riscaldamento/raffrescamento d'ambiente.
- Nuovo `EventBasedApplianceProfile(LoadProfile)` *additivo* sul
  profilo base (come l'HVAC della Fase 17, non sostitutivo). API:
  - `__init__(appliances: list[ApplianceEvent], smart_pv_hourly_shape:
    np.ndarray | None)` — `smart_pv_hourly_shape` è la curva PV
    deterministica letta da `SolarModel.hourly_shape`, usata solo
    quando almeno un `ApplianceEvent` ha `schedule_mode="smart_pv"`.
  - `reset_for_run(rng, n_years)` — per ogni appliance × mese × anno
    estrae N ~ Poisson(`monthly_frequency[m]`), poi per ogni evento
    estrae uniformemente un giorno e un'ora di partenza
    secondo `hour_of_day_weights`. Memorizza la lista di tuple
    `(day_global, start_hour, duration_hours, p_kw, appliance_name)`
    ordinate per indice orario per ricerca O(1).
  - `get_hourly_load_kw(year_index, month_in_year, day_in_month,
    hour_in_day, weekday)` — somma `p_kw` di tutti gli eventi che
    *coprono* l'ora corrente (un evento iniziato all'ora *h* con durata
    `d` copre le ore `h, h+1, …, h+⌈d⌉-1`, con peso *frazionario*
    sull'ultima ora pari a `d - floor(d)` per preservare l'energia
    totale dell'evento).
  - Picco di concorrenza tracciato lato simulatore via il
    consolidato `peak_p_load_kw` (vedi sotto KPI).
- Decoratore composto in `EnergySystemSimulator`: la pipeline diventa
  `base_load → StochasticLoadProfile (Fase 17) → EventBasedApplianceProfile
  → HVAC additivo (Fase 17)`. L'ordine *moltiplicativo prima, additivo
  dopo* è importante: applicare il moltiplicatore LogN agli eventi
  discreti li sporcherebbe (un evento da 1.5 kW × 0.85 = 1.275 kW non
  è realistico — un appliance accende o spegne, non si modula). Gli
  eventi e l'HVAC restano *deterministicamente* additivi sopra il
  carico-base-rumorizzato.
- `EnergySystemConfig` esteso con `appliance_profile_config:
  ApplianceProfileConfig | None` (lista di `ApplianceEvent` +
  `smart_pv_enabled` flag globale che propaga `schedule_mode` agli
  appliance).
- Nuovo blocco `summary.appliances` (solo se attivo):
  - `total_appliance_kwh_annual` — totale kWh/anno della somma di
    tutti gli appliance (utile per sanity-check: confrontarlo con il
    profilo base aiuta a capire se la calibrazione è realistica).
  - `appliance_kwh_annual_by_name` — dizionario `{name: kWh/anno}`
    per i grafici a barre nella Dashboard.
  - `peak_simultaneous_kw_mean` — picco simultaneo medio tra i path
    (E[max_h sum_a P_a(h)]). Quando alto rispetto a `inverter_p_ac_max_kw`
    indica clipping inverter probabile.
  - `share_of_total_load_pct_mean` — frazione del carico totale
    imputabile agli appliance discreti.
  - `smart_pv_self_consumption_pct` — quando `smart_pv` è abilitato,
    quota di kWh degli eventi che cade sotto la curva PV oraria di
    quel path. KPI di efficacia dello scheduling intelligente.
- `scenario_builder.build_default_appliance_profile_config(scenario)`
  — parsing del blocco `load_profile.appliances`. Forma JSON:
  ```json
  "load_profile": {
    ...
    "appliances": {
      "enabled": true,
      "smart_pv": true,
      "items": [
        {"type": "washing_machine"},
        {"type": "ev_charger_slow", "schedule_mode": "smart_pv"},
        {"type": "dishwasher", "monthly_frequency_override": [10, 10, ...]}
      ]
    }
  }
  ```
- `validation._validate_appliances(raw)` — enforce:
  - `items` lista non vuota di dict;
  - ogni `type` ∈ catalogo (oppure `custom` con campi pieni esplicitamente);
  - `monthly_frequency` lunghezza 12, valori ≥ 0;
  - `p_kw > 0`, `duration_hours > 0`;
  - `allowed_hours` subset di `range(24)`, non vuoto;
  - `schedule_mode ∈ {"naive_timer", "smart_pv"}`.
- Frontend wizard step Carico: terzo toggle gated "Appliance discreti"
  con multi-select dei preset + per-appliance toggle "Avvia durante le
  ore di sole". Mostra una tabella riassuntiva "kWh/anno previsti per
  appliance" calcolata dal frontend come
  `Σ_m frequency[m] × p_kw × duration_hours` per dare feedback
  immediato all'utente prima di lanciare la simulazione.
- Dashboard: nuovo widget "Carico appliance" con bar chart
  `appliance_kwh_annual_by_name` + card `peak_simultaneous_kw_mean` +
  (se smart_pv abilitato) card `smart_pv_self_consumption_pct` con
  delta vs naive_timer di riferimento.
- Test (~12 backend in `tests/test_phase17bis_appliances.py`):
  - long-run kWh/anno medio per appliance entro 5 % di
    `n_events_expected × p_kw × duration_hours`;
  - distribuzione delle ore di partenza confinata in `allowed_hours`
    (no leak);
  - `smart_pv` shifta il centroide degli avvii verso le ore PV peak;
  - `peak_simultaneous_kw` cresce monotonicamente col numero di
    appliance concorrenti;
  - byte-identità legacy quando il blocco è assente / `enabled=false`;
  - integrazione end-to-end con HVAC (Fase 17) e stocastico (Fase 17)
    — i tre contributi si sommano correttamente nell'`p_load_kw`
    visto dal dispatcher inverter;
  - validazione rifiuta `monthly_frequency` con shape sbagliata, ore
    illegali, durate negative;
  - reproducibility byte-identica con stesso seed.

**Out of scope ora**:

- **Demand response price-responsive** (rinviare l'avvio degli eventi
  in base al PUN orario o a un segnale dell'inverter ibrido). Resta una
  potenziale **Fase 17-ter**: richiederebbe un modello di prezzo
  orario, che oggi non esiste (la Fase 2 lavora con prezzi mensili).
- **Comportamento occupante differenziato per weekday/weekend**: la
  Fase 5 ha introdotto `WeeklyPatternLoadProfile` lato baseline, ma
  qui assumiamo `monthly_frequency` indipendente dal giorno della
  settimana. Un'estensione naturale è suddividere
  `monthly_frequency_weekday` vs `_weekend`. Schedulabile in coda alla
  Fase 17-bis se serve.
- **Modello termico interno dell'appliance**: il forno non parte a
  2.5 kW istantanei, c'è un ramp di preriscaldamento; il
  condizionatore-cucina ha un duty cycle interno. Tutti questi
  dettagli vengono assorbiti nell'astrazione "evento rettangolare
  p_kw × duration_hours" — per gli scopi di valutazione economica
  l'errore è trascurabile rispetto alle altre approssimazioni del
  simulatore.
- **Coordinazione tra appliance** (es. la lavatrice non può partire
  se l'EV è in carica perché supera il contratto in kW). Il
  `peak_simultaneous_kw_mean` viene calcolato post-hoc come KPI
  diagnostico, ma il *clipping* viene fatto dal contratto-utente
  ipotizzando che superare il limite porta semplicemente al distacco
  dell'utenza — schedulabile in coda alla Fase 17-bis se serve.
- **EV vehicle-to-home (V2H)**: l'auto come batteria scaricata verso
  la casa nelle ore serali. Richiederebbe estendere `BatteryBank` con
  una sorgente esterna disponibile solo in alcune ore — fuori scope
  per ora.
- **DHW (acqua calda sanitaria) modellata come *flusso* anziché
  *eventi*** — la modalità "eventi" qui inclusa è sufficiente per il
  payback economico; una vera modellazione DHW richiederebbe un
  serbatoio termico (capacitanza separata dal modello HVAC della
  Fase 17). Fuori scope.

---

## Fase 18 — Modalità dinamica RC della casa (temperatura interna)

**Problema**: il modello HVAC della Fase 17 (`HvacController` in
`simulation/thermal_load.py`) gira solo in *steady-state*: assume la casa
sempre al setpoint e calcola la potenza istantanea che servirebbe per
tenerla lì. Il flag `ThermalLoadConfig.dynamic` esiste ed è instradato dallo
`scenario_builder`, e la capacità termica `C`
(`capacitance_kwh_per_c_per_m2`) è parsata — ma **entrambi sono ignorati**:
la modalità dinamica era uno stub documentato. Conseguenza: non è possibile
*vedere* la temperatura interna durante un'ondata di freddo/caldo (in
steady-state è per definizione il setpoint), né valutare il sottodimensionamento
della pompa di calore se non come semplice conteggio di ore-breach.

**Deliverable**:

- **Integratore RC a passo orario, Eulero implicito** (incondizionatamente
  stabile anche con τ = C/UA corta, ~17 h per `poor`). Bilancio
  `C·dT_in/dt = Q_HVAC + Q_interni − UA·(T_in − T_out)`, con
  `Q_HVAC` capata a `p_elec_max·COP`.
- **Controllore "deadbeat"**: porta al setpoint in un'ora quando può,
  altrimenti satura. Invariante di non-regressione: con guadagni interni
  nulli e potenza non capata, l'energia dinamica **coincide** con la
  steady-state (e parte dal setpoint ⇒ identica ora per ora).
- `HouseThermalConfig.internal_gains_kw` (nuovo campo avanzato, default
  `0.0`) e property `capacitance_kwh_per_c`. `HvacController` rifattorizzato
  in dispatcher `_compute_steady_state` / `_compute_dynamic` (firma di
  ritorno invariata) con helper condiviso per gli array orari di setpoint
  (già pronto per le fasce orarie della Fase 19).
- **Temperatura interna esposta**: KPI scalari `t_in_min_c` / `t_in_max_c`
  in `ThermalLoadKPIs` (in steady-state valgono i setpoint; in dinamico il
  min/max reale) + serie oraria completa cacheata su
  `HvacController.last_indoor_temp_c` e
  `EnergySystemSimulator.last_indoor_temp_c` (per il futuro preview).
  `aggregate_thermal_kpis` aggrega `t_in_min_c` come **min** e `t_in_max_c`
  come **max** sui path (caso peggiore). Le nuove chiavi fluiscono nel dict
  pass-through `summary["thermal"]` senza modifiche di schema API.
- Test in `tests/test_phase17_stochastic_load_and_hvac.py`
  (`TestPhase17DynamicRc`): invariante con steady-state, drift sotto setpoint
  per casa `poor` con pompa piccola, stabilità numerica, dead-band, semantica
  min/max dell'aggregazione.
- Aggiornamento del notebook `notebooks/modelli_temperatura.ipynb` per usare
  la modalità dinamica reale al posto della ricostruzione manuale.

**Out of scope ora** (→ Fase 19): setpoint a fasce orarie, curva COP(T_out),
endpoint di preview della timeseries di T interna, UI di confronto isolamenti.

## Fase 19 — Laboratorio termico nella webapp (confronto isolamenti + report)

**Problema**: l'utente vuole una sezione dedicata della web app per
ragionare sull'involucro e sulla pompa di calore *prima* dell'analisi
economica completa: dimensionare la pompa, scegliere i setpoint, confrontare
più livelli di isolamento e vederne l'effetto su consumi e temperatura
interna, con un report scaricabile.

**Deliverable** (bozza, da raffinare a inizio fase):

- Nuova sezione "Laboratorio termico" che, fissati un profilo climatico
  (Fase 15), un profilo di presenza/assenza e un profilo di temperatura
  richiesta, lancia N run Monte Carlo e confronta più configurazioni di casa
  (es. `poor` / `standard` / `good` + custom `UA`).
- **Setpoint a fasce orarie** (oltre al singolo): schedule oraria
  heating/cooling — l'integratore della Fase 18 è già schedule-ready.
- Dimensionamento pompa: `cop_heating`, `cop_cooling`, `p_elec_max_kw`.
- Grafici sovrapposti: (a) temperatura esterna; (b) consumi elettrici
  giornalieri per ciascuna configurazione di casa; (c) setpoint vs
  temperatura interna ottenuta. Evidenziazione dei **giorni più gravosi** in
  riscaldamento e in raffrescamento.
- Analisi economica per livello di isolamento usando il modello prezzo
  energia, ed **export report termico** (Excel + grafici energia come nel
  notebook).
- Endpoint API di preview della timeseries (T esterna, T interna, P_elec)
  alimentato dalla serie cacheata in Fase 18.

**Out of scope ora**: modello multi-zona, acqua calda sanitaria separata,
ottimizzazione automatica del dimensionamento pompa, demand response.

---

## Fase 19-bis — Laboratorio termico: accuratezza economica e diagnostica

**Problema**: la Fase 19 ha consegnato il laboratorio termico ma con tre
semplificazioni note (tracciate come "out of scope" alla chiusura):

1. **Prezzo energia scalare.** Il costo annuo HVAC è `kWh × prezzo_fisso`. Non
   cattura l'incertezza del prezzo su 20 anni (escalation + volatilità) che il
   resto del simulatore modella già via `PriceModel` (Fase 2). Confrontare
   isolamenti "a prezzo fisso" sottostima la banda di costo p05–p95 della casa
   poco isolata, che è proprio quella più esposta al rischio prezzo.
2. **Nessuno split riscaldamento/raffrescamento.** La tabella mostra solo il
   totale HVAC; non si vede quanta energia/costo è inverno vs estate — info
   chiave per dimensionare la pompa e scegliere i setpoint.
3. **Anteprima oraria solo invernale.** `simulate_thermal_timeseries` parte
   sempre dal 1° gennaio: non si può ispezionare un'ondata di calore estiva.

**Deliverable**:

- **Accoppiamento `PriceModel`**: `compare_house_variants` accetta un
  `price_model: PriceModel | None`. Quando presente, il costo per path =
  `Σ_giorno kWh_HVAC(g) × prezzo(anno_g, mese_g) / n_years` (media annua sul
  l'orizzonte), con `reset_for_run` per path su uno stream RNG **indipendente**
  da quello della temperatura. Quando `None` → fallback allo scalare
  `electricity_price_eur_per_kwh`, **byte-identico** al comportamento Fase 19.
  Schema/route: blocco `price` opzionale (model_type escalating/gbm/
  mean_reverting + parametri), costruito via `build_default_price_model`.
- **Split energia riscaldamento/raffrescamento**: per variante
  `heating_kwh_annual_mean` / `cooling_kwh_annual_mean`, classificando le ore
  per `T_out` vs i setpoint effettivi (`HvacController.setpoint_arrays`).
  Esatto in steady-state, approssimato in dinamico (documentato). KPI in
  tabella + barra impilata.
- **Finestra anteprima oraria**: `simulate_thermal_timeseries(..., start_day=0)`
  per ispezionare inverno/primavera/estate/autunno. UI: selettore stagione.
- UI: selettore "Prezzo energia" (Fisso / Escalation / GBM / Mean-reverting)
  con campi condizionati; colonne risc./raffr. in tabella; selettore stagione
  per l'anteprima. Export Excel/PDF estesi con lo split.
- Test (~8-10): GBM allarga la banda di costo vs scalare; escalating
  deterministico fa crescere il costo medio annuo; reset prezzo indipendente
  per path; `heating + cooling == totale` in steady-state; cooling = 0 in clima
  freddo; preview estiva mostra cooling; byte-identità quando `price` assente;
  endpoint con blocco `price`.

**Out of scope ora**: editor schedule 24h completo (resta il solo setback
notturno), tariffe orarie/PUN (il `PriceModel` è mensile), demand response.

---

## Fase 20 — Mercato elettrico sottostante (prezzo endogeno + ritiro dedicato)

**Problema**: oggi l'energia prodotta in eccesso (PV che non è né autoconsumata
né immagazzinata) viene **scartata silenziosamente** in `inverter.dispatch()`
quando la batteria è piena: non è valorizzata in alcun modo. Inoltre il prezzo
dell'energia è esogeno (`EscalatingPriceModel`/`GBM`/`MeanReverting`): non è
"agganciato" né alle condizioni operative (settimana di maltempo → poco PV di
sistema → prezzo diurno alto; estate → midday a poco prezzo) né a quelle di
mercato (shock gas → elettricità cara). Manca un modello del mercato elettrico
che, da un mix di generazione, determini un prezzo orario sensato, e manca la
valorizzazione dell'immissione tramite **ritiro dedicato** (energia immessa
pagata a `max(prezzo zonale orario, prezzo minimo garantito)`).

**Decisioni utente (2026-05-31)**: motore **completo** (incl. interconnessioni
con l'estero, price-areas, storage di rete); prezzo applicato all'impianto a
risoluzione **oraria** (superficie mese×ora); il mercato può guidare **sia
l'immissione sia l'acquisto**, via toggle indipendenti; PMG **indicizzato
all'inflazione** (`PMG_base·(1+infl)^anno`, riusa l'infrastruttura inflazione
della Fase 11). Motore portato dallo standalone `energy_mix_simulator`
(adattato dentro il repo, non come dipendenza esterna).

**Dipendenze**: Fase 11 (infrastruttura inflazione, riusata per il PMG),
Fase 15 (precedente `ClimateProfileModel` per il profilo riutilizzabile salvato
nel DB), Fase 19 (pattern di sezione/lab end-to-end).

**Deliverable** (7 slice indipendenti, ognuna a suite verde):

- **20a — Port del motore di mercato** → nuovo sottopacchetto
  `sim_stochastic_pv/market/` (mirror del pattern `external/`): `config`,
  `grid` (TimeGrid + load di sistema), `generators` (prezzi fuel/CO₂ OU,
  availability solare/eolico, `Generator`, `build_generators`), `dispatch`
  (merito economico + fix inerzia + export interconnessioni + storage),
  `storage`, `interconnections`, `price_areas`, `reliability`, `simulation`
  (`run_monte_carlo` + sweep). Niente layer di plotting (i grafici vivono in
  frontend/exporters). Test per modulo (`tests/test_market_*.py`): merito
  economico sceglie il più economico, prezzo sale con μ_gas, solare nullo di
  notte, riproducibilità da seed, autocorrelazione lag-1 dei path fuel.

- **20b — Trend di mix + superficie di prezzo** → `market/horizon.py`:
  `MixTrend` (capacità base + crescita %/anno + step a un anno dato, es.
  "nucleare 0 fino al 2035 poi +X GW"), `build_mix_for_year(year)`, e
  `PriceSurface` che gira il MC di mercato su **anni rappresentativi** e
  interpola. Aggiunge a `market/simulation.py` l'aggregato mancante: prezzo
  medio per `(mese, ora)` shape `(n,12,24)` + bande p05/p95. Superficie
  **cachabile** (calcolo costoso una volta sola).

- **20c — Cattura del surplus** (risolve la lacuna originale): oggi l'eccesso
  PV è scartato silenziosamente. `inverter.dispatch()` ora ritorna due termini
  invece di buttarlo: `e_pv_to_grid` (immettibile) e `e_pv_curtailed` (perso).
  Lo split è necessario perché l'eccesso esce dallo **stesso inverter** e quindi
  è limitato dal tetto AC residuo (`p_ac_max` meno la potenza AC già usata da
  PV-diretto e scarica batteria); ciò che eccede il tetto è fisicamente
  tagliato. Invariante di bilancio orario:
  `prod = diretto + a_batteria + to_grid + curtailed` (modello DC↔AC 1:1, senza
  perdite). `EnergySystemSimulator` accumula `monthly_export_kwh`,
  `monthly_curtailed_kwh`, `monthly_pv_to_batt_kwh` e
  `export_kwh_by_year_month_hour` (shape `(n_years,12,24)`, l'input orario per
  la valorizzazione in 20d). Tutto esposto via attributi `self.last_*` (stesso
  pattern di `last_electrical_kpis`/`last_thermal_kpis`): la firma di
  `run_one_path` **non cambia**, quindi il MC fotovoltaico resta byte-identico
  finché 20d non legge i nuovi attributi. Solo contabilità energetica, ancora
  senza prezzi. Test: bilancio del dispatch inverter, export limitato dal tetto
  AC, surplus→batteria prima di →rete, export nullo di notte, bilancio
  d'orizzonte nel simulatore.

- **20d — Provider prezzo + integrazione economica**: `MarketPriceProvider`
  (wholesale/retail/export), estensione opzionale di `PriceModel` con
  `get_price_hourly` (default = prezzo mensile su tutte le ore → **byte-identico**
  all'attuale a mercato spento); `export_orario = max(wholesale, PMG)`,
  `PMG(anno)=PMG_base·(1+infl)^anno`; retail orario opzionale
  `= wholesale·(1+markup)+componenti_fisse`. Il ricavo da immissione entra nel
  cashflow come il bonus fiscale (Fase 11) → NPV/IRR/break-even lo includono.
  Nuovi campi in `MonteCarloResults` (`monthly_export_kwh_paths`,
  `monthly_export_eur_paths`, `df_export`, totali).

- **20e — Persistenza**: `MarketProfileModel(name, description, data JSON)`
  (segue il pattern `ClimateProfileModel`), `market_profile_id` opzionale sullo
  scenario **come chiave nel JSON di config** (esattamente come
  `climate_profile_id`: nessuna colonna FK, quindi la nuova tabella
  `market_profiles` è materializzata da `create_all` e non serve `ALTER TABLE`
  sulle tabelle esistenti), hydration in `scenario_builder`
  (`build_default_market_provider`), regole in `validation.py`, seed di un
  profilo "Italia" di default. Il `data` JSON contiene la superficie di prezzo
  precalcolata serializzata (`MarketPriceProvider.to_config_dict`) così il MC
  PV fa solo lookup.

- **20f — Sezione "Mercato Elettrico"** (pattern ThermalLab): lab orchestrator,
  `api/routes/market.py` (`/api/market`), `api/schemas/market.py`, pagina
  `frontend/src/pages/ElectricityMarket.svelte` (editor mix con pie/stacked-bar;
  editor trend con stacked-area della capacità sull'orizzonte; scenari
  gas/CO₂/coal; interconnessioni/storage; output: heatmap prezzo mese×ora, fan
  chart annuale, curva di durata, heatmap "chi fissa il prezzo") + voce navbar
  + rotta + `api.js` + exporters `{pdf,xlsx}_market.py`. Heatmap via componente
  canvas/CSS-grid leggero (nessuna libreria nuova, §3.5).

- **20g — Integrazione nello scenario + Dashboard**: nello step "Mercato
  elettrico" del wizard, scelta "modello prezzo semplice" vs "mercato simulato"
  (profilo salvato o inline) + toggle "ritiro dedicato (PMG)" e "il mercato
  guida anche l'acquisto". Dashboard: KPI ricavo immissione, bilancio
  energetico con export, contributo dell'export al profitto, superficie di
  prezzo usata.

**Nota di modellazione (limite noto)**: la superficie di prezzo è precalcolata
dal MC di mercato (con i *suoi* path meteo) e poi consultata dal MC PV (con i
*suoi* path meteo). Cattura la struttura media oraria/stagionale e l'effetto-
livello degli shock gas/CO₂, ma **non** la correlazione intra-path tra il meteo
della singola casa e il prezzo di sistema *nello stesso path*. La coppia piena
(meteo-casa ↔ prezzo-sistema su fattore comune) è una possibile "modalità
coupled" futura, fuori scope qui.

**Out of scope ora**: mercato infragiornaliero/MSD/bilanciamento; aste reali
GME; PUN vs prezzo zonale per zona (si usa un'unica zona); comunità energetiche
(CER); accise e oneri di sistema dettagliati per fascia (il retail è wholesale +
componenti aggregate); modalità "coupled" meteo↔prezzo.

---

## Dipendenze fra fasi

```
Fase 2 ─┐
        ├─→ Fase 3 ──→ Fase 4 ──┐
Fase 1 ─┘                       │
                                │
Fase 5 ────────────────────────┐│
Fase 8 (load profile rework) ──┤│
                                ││
Fase 7 (terminology) ──────────┐│
Fase 9 (simplified sizing)   ──┤│
                                │↓
                                Fase 6 (wizard finale) ◄─ tutte ─┘

Fase 10 (price preview) ── dipende solo da Fase 2 (già fatta)
Fase 11 (bonus, inflazione, export, rename Design) ── indipendente
Fase 12 (jobs, archive, filtri Dashboard) ── indipendente

Fase 13 (sync README/CLI/glossario)         ── indipendente
Fase 14 (geolocation + PVGIS + Open-Meteo)  ── indipendente
                                            │
Fase 15 (modello termico) ◄─ Fase 14 ──────┤    (Fase 14 alimenta
                            │                    la calibrazione)
                            ↓
                            ├─→ Fase 16 (modello elettrico opt-in)
                            └─→ Fase 17 (carico stocastico + HVAC)
                                          │
                                          ↓
                                Fase 17-bis (appliance discreti event-based)

Fase 17 (HVAC steady-state) ──→ Fase 18 (dinamica RC, T interna)
                                          │
                                          ↓
                                Fase 19 (laboratorio termico webapp)
```

Sequenza consigliata se si vuole valore precoce:
**13 → 14 → 15 → (16 ∥ 17) → 17-bis** — la 13 è cheap e ripulisce il
punto d'ingresso, la 14 sblocca due colli di bottiglia sotto (dati reali
per clima e solare), la 15 è il prerequisito comune di 16 e 17, la
17-bis estende la 17 con il modello a eventi per chi vuole valutare
strategie di scheduling intelligente (auto EV su PV, smart timer).

## Stato

> Questo è il *log vivente* dell'evoluzione del progetto. Aggiornare
> ogni volta che si entra o si esce da una fase (regola formalizzata in
> `CLAUDE.md §4`). Le date sono quelle effettive di inizio/fine sessione,
> non quelle "ideali" del piano.

### 🚧 In corso

**Fase 20 — Mercato elettrico sottostante (prezzo endogeno + ritiro dedicato)**
— avviata 2026-05-31. Slice completate: **20a** (port motore di mercato),
**20b** (trend di mix + superficie di prezzo cachabile), **20c** (cattura del
surplus PV come export/curtailment), **20d** (provider prezzo
`MarketPriceProvider` + integrazione del ricavo da immissione nel cashflow),
**20e** (persistenza `MarketProfileModel` + idratazione nello scenario).
Prossima slice: **20f** (sezione "Mercato Elettrico" + API + UI). Vedi blocco
*Fase 20* sopra per il piano completo delle 7 slice e le decisioni di design.

**20e completata (2026-05-31)**: nuovo `MarketProfileModel(name, description,
data JSON)` in `db/models.py` (tabella `market_profiles`, materializzata da
`create_all`); `MarketPriceProvider.to_config_dict`/`from_config_dict`
serializzano superficie + PMG/retail; `persistence/market_repo.py`
(`MarketProfileRepository` con CRUD + `load_market_provider`), esposto via il
facade `PersistenceService` (`self.market` + metodi delega);
`scenario_builder.build_default_market_provider` idrata il provider dal
`market_profile_id`/`market_profile_name` nel JSON di scenario (pattern
`climate_profile_id`, nessuna colonna FK); `application.run_analysis` passa il
provider al MC (byte-identico quando assente); check di tipo in
`validation.py`; `db/seeding.seed_market_profiles` semina il profilo "Italia
(mercato base)" (superficie 8 traiettorie × 20 anni, PMG 0.04 €/kWh) sia su DB
nuovo sia su DB esistente via `init_db` (idempotente).
`tests/test_market_profile_persistence.py` (15 test). Verificato end-to-end:
scenario con `market_profile_id` → ΔNPV positivo. Suite 565 verde.

**20d completata (2026-05-31)**: nuovo modulo `simulation/market_pricing.py`
con `MarketPriceProvider` (export = `max(wholesale, PMG)`,
`PMG(anno)=PMG_base·(1+infl)^anno` indicizzato riusando i fattori d'inflazione
della Fase 11, più tariffa retail opzionale `wholesale·(1+markup)+fissi`);
hook `PriceModel.get_price_hourly` (default piatto su 24h → byte-identico);
`MonteCarloSimulator` accetta un `market_price_provider` opzionale e ripiega il
ricavo da immissione nel cashflow come il bonus fiscale (→ NPV/IRR/break-even).
Nuovi campi `MonteCarloResults`: `monthly_export_kwh_paths`,
`monthly_export_eur_paths`, `df_export`, `export_revenue_total_mean_eur`,
`export_kwh_total_mean`. A mercato spento il run resta byte-identico.
`tests/test_market_pricing.py` (22 test). Suite 550 verde.

### ✅ Completate

**Fase 19-bis — Laboratorio termico: accuratezza economica e diagnostica** — chiusa 2026-05-30 (suite verde, 482 test backend; build frontend OK; verificata end-to-end nel browser).

Consegnato:
- **Accoppiamento `PriceModel`** (`simulation/thermal_lab.py`):
  `compare_house_variants(..., price_model=None)`. Con un modello prezzo il
  costo per path = `Σ_giorno kWh(g)·prezzo(anno_g, mese_g)/n_years` (prezzo
  per-mese vettorizzato via `month_of_year`), `reset_for_run` per path su uno
  stream RNG indipendente (`_PRICE_SEED_OFFSET`). Senza modello → scalare
  **byte-identico** alla Fase 19 (verificato). Test: GBM allarga la banda di
  costo >3× lo scalare, escalating deterministico alza il costo medio,
  riproducibilità da seed.
- **Split riscaldamento/raffrescamento**: `ThermalVariantResult` +
  `heating_kwh_annual_mean` / `cooling_kwh_annual_mean`, classificati per
  `T_out` vs i setpoint effettivi (`HvacController.setpoint_arrays`). Esatto in
  steady-state (`heating + cooling == totale`, verificato), approssimato in
  dinamico (documentato).
- **Finestra anteprima oraria**: `simulate_thermal_timeseries(..., start_day)`
  (0..364) con warmup AR(1) + ampiezza diurna corretta per la stagione.
- **API**: `ThermalLabCompareRequest.price` (blocco opzionale escalating/gbm/
  mean_reverting, costruito via `build_default_price_model`),
  `ThermalTimeseriesRequest.start_day`, campi risc./raffr. in
  `ThermalVariantResultSchema`. Route con helper `_build_price_model` /
  `_price_label`.
- **Export** Excel + PDF estesi con colonne risc./raffr. e l'etichetta del
  modello prezzo nell'header.
- **UI** (`ThermalLab.svelte`): selettore "Prezzo energia"
  (Fisso/Escalation/GBM/Mean-reverting) con campi condizionati e banda costo
  p05–p95 in tabella; colonna risc./raffr.; selettore stagione per l'anteprima
  oraria. Verificato live: GBM allarga la banda di costo, split mostrato,
  selettore stagione ricarica la serie, export PDF/Excel 200.
- **Test** `tests/test_phase19_thermal_lab.py`: +11 (price coupling, split,
  finestra stagionale, endpoint con blocco prezzo). Totale 40 nel file, 482
  nell'intera suite.

**Fase 19 — Laboratorio termico nella webapp** — chiusa 2026-05-29 (suite verde, 471 test backend; build frontend OK; verificata end-to-end nel browser).

**Fase 19 — Laboratorio termico nella webapp** — chiusa 2026-05-29 (suite verde, 471 test backend; build frontend OK; verificata end-to-end nel browser).

Consegnata in tre slice nella stessa giornata (backend-first, come Fase 18):
fondazione backend, UI Svelte, export report Excel/PDF.

**Slice 1 — fondazione backend:**
- **Setpoint a fasce orarie** (il seam già predisposto in Fase 18):
  `SetpointConfig` esteso con `heating_schedule_c` / `cooling_schedule_c`
  opzionali (24 valori hour-of-day, coerciti a tuple, invariante dead-band
  validata ora-per-ora). `HvacController._build_setpoint_arrays` li applica
  indicizzando per `index % 24`; assenza ⇒ scalare → **byte-identico**
  (verificato in test). Nuovo metodo pubblico
  `HvacController.setpoint_arrays(at_home_hourly)` per preview/lab.
- **`simulation/thermal_lab.py`** (nuovo): `HouseVariant`, `ThermalLabConfig`,
  `ThermalVariantResult`, `ThermalLabResult`,
  `compare_house_variants(model, config, n_paths, n_years, seed)` — lancia N
  path MC del solo sottosistema termico (ogni variante valutata sugli *stessi*
  path) e confronta più configurazioni di casa (preset isolamento o `UA`
  custom). Per variante: kWh/anno HVAC (mean+p05/p95), costo €/anno
  (× prezzo scalare), comfort breach h/anno, picco kW, T interna worst-case,
  serie giornaliera "anno tipico" calendar-aligned (kWh HVAC, T esterna
  condivisa, T interna rappresentativa in dinamico), giorni più gravosi
  heating/cooling. `simulate_thermal_timeseries(...)` + `ThermalTimeseriesResult`
  per la preview oraria (T esterna, T interna, P_elec, setpoint) di una
  singola config.
- **API**: `POST /api/thermal-lab/compare` e `POST /api/thermal-lab/timeseries`
  (`api/routes/thermal_lab.py` + `api/schemas/thermal_lab.py`), alimentati dal
  `ClimateProfileModel` (Fase 15) via `persistence.climate.load_thermal_model`.
  404 su profilo mancante, 400 su invarianti di dominio violate. Setpoint
  `±inf` (ore away senza setback) serializzati come `null`.
- **Test** `tests/test_phase19_thermal_lab.py` (24 test): schedule byte-identico
  vs scalare, night-setback riduce energia, validazione lunghezza/dead-band,
  monotonia energia vs isolamento, costo = kWh×prezzo, riproducibilità da seed,
  drift sotto setpoint con pompa sottodimensionata in dinamico, giorni più
  gravosi (clima freddo → solo heating; clima caldo → anche cooling), endpoint
  compare/timeseries (schema, 404, 400, setpoint null su ore away).

**Slice 2 — UI Svelte:**
- Nuova pagina **`frontend/src/pages/ThermalLab.svelte`** + rotta `/thermal-lab`
  + voce navbar "Lab termico". Form di configurazione (selettore profilo
  climatico, scelta varianti casa con preset + UA custom, superficie, pompa di
  calore COP/p_max, setpoint con setback notturno opzionale, presenza/assenza
  con setback, prezzo energia, parametri MC, toggle modello dinamico RC).
- Risultati: tabella KPI comparativa (UA, kWh/anno con p05–p95, costo €/anno,
  comfort breach evidenziato, picco kW, T interna min/max in dinamico) +
  **3 grafici sovrapposti** (consumi giornalieri/config con T esterna su asse
  secondario e marker dei giorni più gravosi ●/▲; barre costo/config con banda
  p05–p95 in tooltip; anteprima oraria setpoint vs T interna + P_elec).
- Verifica browser: layout 2-col (config sticky) ≥900px → 1-col sotto, nessun
  errore console, monotonia energia/costo e drift T interna confermati live.

**Slice 3 — Export report termico (Excel + PDF):**
- `output/exporters/xlsx_thermal_lab.py` (`build_thermal_lab_xlsx`): workbook
  3 fogli (Confronto KPI, Serie giornaliera, Temperatura interna in dinamico).
- `output/exporters/pdf_thermal_lab.py` (`build_thermal_lab_pdf`): report
  WeasyPrint + matplotlib (tabella KPI + grafici consumi/costo/T interna).
- Endpoint `POST /api/thermal-lab/compare/export.{xlsx,pdf}` (stessa request di
  `/compare`); route rifattorizzata con helper condivisi `_run_comparison` /
  `_result_to_response` / `_report_dict`.
- UI: pulsanti CSV / Excel / PDF nell'header della tabella KPI
  (`api.exportThermalLabXlsx` / `exportThermalLabPdf` via `downloadPost`) +
  PNG per grafico (nativo `ResultsChart`). Verificato end-to-end nel browser
  (Excel + PDF HTTP 200, content-type corretti) e via 5 test backend.

**Out of scope (refinement futuri, eventuale Fase 19-bis):** accoppiamento del
modello prezzo GBM stocastico (per ora prezzo scalare), editor schedule 24h
completo (per ora solo setback notturno), finestra estiva nell'anteprima oraria
(per ora parte da gennaio), split energia heating/cooling per ora.

**Fase 18 — Modalità dinamica RC della casa (temperatura interna)** — chiusa 2026-05-29.

Scope consegnato in questa iterazione: **libreria + KPI in API** (la sezione
webapp di confronto isolamenti resta tracciata come Fase 19, non iniziata).

Consegnato:
- **`simulation/thermal_load.py`**: `HvacController` rifattorizzato in dispatcher
  `_compute_steady_state` / `_compute_dynamic` (firma pubblica
  `compute_hourly_p_elec_kw` invariata). Nuova modalità dinamica RC a **Eulero
  implicito** (passo 1 h, incondizionatamente stabile) con controllore
  *deadbeat*; helper condiviso `_build_setpoint_arrays` (già schedule-ready per le
  fasce orarie della Fase 19). Invariante verificata: con `internal_gains_kw=0` e
  pompa non capata l'energia dinamica coincide con la steady-state.
- `HouseThermalConfig`: nuovo campo avanzato `internal_gains_kw` (default 0.0) +
  property `capacitance_kwh_per_c`. `ThermalLoadKPIs`: nuovi `t_in_min_c` /
  `t_in_max_c`. `aggregate_thermal_kpis` aggrega la T interna come **caso peggiore**
  (min dei min, max dei max sui path); le chiavi fluiscono nel dict pass-through
  `summary["thermal"]` senza modifiche di schema API (verificato end-to-end).
- **`simulation/energy_simulator.py`**: cache `last_indoor_temp_c` della serie
  oraria di T interna (per il futuro preview della Fase 19).
- **`tests/test_phase17_stochastic_load_and_hvac.py`**: nuova classe
  `TestPhase18DynamicRc` (8 test: invariante con steady-state, energia annua entro
  pochi %, drift sotto setpoint per casa `poor` sottodimensionata, stabilità a
  costante di tempo corta, dead-band, guadagni interni, aggregazione worst-case).
  Suite intera verde (442 test).
- **`notebooks/modelli_temperatura.ipynb`**: Sezione 3 aggiornata per usare la
  modalità dinamica reale della libreria (`dynamic=True` +
  `controller.last_indoor_temp_c`) al posto dell'integratore ricostruito a mano.

Nota ambientale: il venv aveva `tinycss2 1.4.0` incompatibile con `weasyprint 68.1`
(`ModuleNotFoundError: tinycss2.color5`), che bloccava la collection di 7 file di
test API/export; aggiornato a `tinycss2 1.5.1` (dev tooling, non in
`requirements.txt`).

**Fase 17-bis — Carico event-based con appliance discreti** — chiusa 2026-05-29.

Consegnato:
- **Nuovo modulo `sim_stochastic_pv/simulation/load_profiles/appliances.py`**:
  - `ApplianceEvent` (dataclass frozen) con i 6 campi obbligatori +
    `schedule_mode ∈ {naive_timer, smart_pv}` + `hour_of_day_weights`
    opzionale + `expected_kwh_annual()` per il readout predittivo
    lato wizard.
  - `APPLIANCE_PRESETS` con gli 8 preset realistici previsti dalla
    specifica (washing_machine, dishwasher, oven, dryer,
    ev_charger_slow, ev_charger_fast, induction_cooktop,
    dhw_heat_pump_cycle) calibrati su consumi tipici residenziali EU.
    `get_preset(name)` case-insensitive con KeyError esplicativo.
  - `ApplianceProfileConfig(enabled, smart_pv_default, appliances)`
    container del blocco scenario.
  - `EventBasedApplianceProfile(LoadProfile)` decoratore *additivo*:
    `reset_for_run` rasterizza l'intera path su un array
    `hourly_kw[n_years*12*30*24]` allineato al mock-calendar del
    simulatore. Per ogni (appliance, mese, anno):
    `N ~ Poisson(monthly_frequency[m])`, poi per ogni evento giorno
    uniforme su 30 giorni e ora di partenza categorica con pesi
    `allowed_hours * hour_of_day_weights * solar_shape (se smart_pv)`.
    Energia frazionaria preservata: durata `d` non-intera distribuisce
    `floor(d)` ore a piena potenza + 1 ora a peso `d - floor(d)`.
  - `AppliancesKPIs` (total_appliance_kwh_annual,
    appliance_kwh_annual_by_name dict, peak_simultaneous_kw,
    share_of_total_load_pct, smart_pv_self_consumption_pct).
  - `kpis_for_path(n_years, pv_hourly_kw)` calcola anche
    `smart_pv_self_consumption_pct = 100% × Σ min(appliance, pv) /
    Σ appliance` quando il PV array è disponibile.
  - `aggregate_appliances_kpis(list)` → dict con union dei nomi e
    media per-name (zero per i path che non hanno avuto eventi).
- **Wiring `EnergySystemSimulator`**:
  - `EnergySystemConfig.appliance_profile_config: Optional` (None →
    legacy byte-identico).
  - `__init__` istanzia `EventBasedApplianceProfile` quando
    `enabled && appliances`, passando `solar_model.hourly_shape` per
    abilitare smart_pv. Salva `last_appliances_kpis` cache.
  - `run_one_path`: dopo `load_profile.reset_for_run` chiama anche
    `appliance_profile.reset_for_run`. Nel ciclo orario somma
    `appliance_profile.get_hourly_load_kw(...)` al `p_load_kw`
    **dopo** la baseline (stocastica) e **prima** del contributo HVAC
    — ordine moltiplicativo-prima/additivo-dopo motivato in docstring.
  - Alla fine del path calcola `share_of_total_load_pct` usando
    `monthly_load_kwh.sum()` come denominatore (consistente con HVAC).
- **`MonteCarloResults`** esteso con `appliances_kpis_per_path` +
  `appliances_kpis_summary` (entrambi None in legacy).
- **`application.run_analysis`** espone `summary["appliances"]` (None
  se off).
- **`scenario_builder.py`**:
  - `_resolve_appliance_item(raw, smart_pv_default)`: parsing
    catalog + override per item (monthly_frequency_override, p_kw,
    duration_hours, allowed_hours, hour_of_day_weights, name);
    supporta anche `type="custom"` con specifica completa esplicita
    via `dataclasses.replace`.
  - `build_default_appliance_profile_config(scenario_data)` legge
    `load_profile.appliances.{enabled, smart_pv, items[]}`, propaga
    `smart_pv` come default a chi non specifica un proprio
    `schedule_mode`.
  - Integrato in `build_default_energy_config` accanto agli altri
    config Phase 17.
- **`validation._validate_appliances(raw)`** — enforce:
  - block dict + `items` lista non-vuota,
  - `type` ∈ catalog ∪ `{custom}`,
  - `schedule_mode` ∈ {naive_timer, smart_pv} quando presente,
  - `monthly_frequency_override` lunghezza 12 con valori ≥ 0,
  - per `type="custom"`: `p_kw > 0`, `duration_hours > 0`,
    `monthly_frequency` length 12 ≥ 0, `allowed_hours` subset di
    `range(24)` non-vuoto.
  - Import lazy del catalog per evitare loop circolare sim → validation.
- **Frontend Svelte (ScenarioBuilder step Carico)**:
  - Stato JS: `appliancesEnabled`, `appliancesSmartPv`,
    `APPLIANCE_CATALOG` (mirror lato JS degli 8 preset con etichette
    italiane), `appliancesSelection` (dict per-key { enabled,
    smart_pv_override }).
  - Reactive `appliancesPredictedKwh` + `appliancesPredictedTotal`:
    calcola la previsione `p_kw × duration × monthly_freq × 12` lato
    client e aggiorna istantaneamente la tabella riassuntiva al
    cambio di selezione — feedback immediato senza round-trip server.
  - Toggle gated "Appliance discreti (Phase 17-bis)" con sub-toggle
    "Modalità smart PV globale" + multi-select grid degli 8 preset
    (label italiana + p_kw + duration + frequenza nel sotto-titolo).
  - Tabella predittiva "kWh/anno previsti" inline con totale finale.
  - `buildPayload` aggiunge `load_profile.appliances.{enabled,
    smart_pv, items[]}` solo quando il toggle è attivo e almeno un
    item è selezionato.
- **Test** (`tests/test_phase17bis_appliances.py`): **32 nuovi test**:
  - `TestPhase17bisApplianceEventValidation` (5): negative p_kw,
    duration_hours zero, monthly_frequency wrong length, allowed_hour
    out of range, expected_kwh_annual formula.
  - `TestPhase17bisCatalog` (3): nomi unici, lookup
    case-insensitive, unknown preset.
  - `TestPhase17bisScheduler` (6): long-run kWh entro 2% su 60
    seed × 5 anni (Poisson convergente), zero-frequency → array
    nullo, eventi confinati in `allowed_hours`, smart_pv shifta il
    centroide verso le 12, errore smart_pv senza solar_shape,
    riproducibilità byte-identica per seed.
  - `TestPhase17bisConcurrency` (1): peak con 2 lavatrici > peak
    con 1 lavatrice + 0.5 kW (subadditivo per saturazione su
    `allowed_hours`).
  - `TestPhase17bisAggregation` (2): empty input, mean per-name.
  - `TestPhase17bisScenarioBuilder` (7): missing block, disabled,
    preset only, smart_pv_default propagato, override per-item,
    monthly_frequency_override, custom appliance round-trip.
  - `TestPhase17bisValidation` (4): unknown preset, schedule_mode
    invalido, monthly_frequency_override length sbagliata, custom
    senza required field.
  - `TestPhase17bisLegacyByteIdentity` (3): no block, disabled,
    end-to-end MC `final_gain_mean_eur` byte-identico tra "no block"
    e "enabled=false".
  - `TestPhase17bisEndToEnd` (1): abilitare appliance modifica
    `final_gain_mean_eur` e popola `summary.appliances` con
    `appliance_kwh_annual_by_name_mean` per ogni voce attivata.
  - Suite totale: **434/434 verde** (402 pre-Fase-17-bis + 32 nuovi).

Note operative:
- Per esercitare il modello live: step Carico → toggle "Appliance
  discreti" → check almeno un preset → la tabella "kWh/anno previsti"
  conferma le aspettative deterministiche; lanciare l'analisi popola
  `summary.appliances` con `total_appliance_kwh_annual_mean`,
  `peak_simultaneous_kw_mean`, e (se smart_pv) la
  `smart_pv_self_consumption_pct_mean`.
- Out of scope (potenziali Fasi 17-ter/quater):
  demand-response price-responsive (richiede prezzi orari, fuori
  Fase 2), weekday/weekend split nel `monthly_frequency`, ramp
  termici interni, coordinazione kW-contratto, EV V2H, DHW con
  serbatoio termico.

**Fase 17 — Carico stocastico con accoppiamento termico** — chiusa 2026-05-28.

Consegnato:
- **Nuovo modulo `sim_stochastic_pv/simulation/load_profiles/stochastic.py`**:
  - `StochasticLoadConfig(enabled, sigma_log, phi_intra_day)` dataclass
    con validazione (sigma_log ≥ 0, |phi| < 1).
  - `StochasticLoadProfile(base, config)` decorator. `reset_for_run`
    pre-genera l'intero path orario di moltiplicatori tramite
    `_sample_lognormal_ar1_path(n_hours, sigma_log, phi, rng)` —
    formula:
        z[h] = phi · z[h-1] + sigma_innov · w[h],
        eps[h] = exp(z[h] − σ²/2),
    con `sigma_innov = sigma_log · √(1−φ²)` che rende la varianza
    marginale di z esattamente `σ_log²` e la correzione di Itō tiene
    `E[eps]=1` (verificato empiricamente: mean=0.9987, std(log)=0.2007,
    lag-1=0.5000 su 100k samples). `get_hourly_load_kw` è O(1).
- **Nuovo modulo `sim_stochastic_pv/simulation/thermal_load.py`**:
  - 3 preset isolamento (`poor`=2.5, `standard`=1.5, `good`=0.8 W/°C/m²).
  - `HouseThermalConfig(floor_area_m2, insulation_preset,
    ua_w_per_c_per_m2 override, capacitance_kwh_per_c_per_m2)` con
    derived `ua_kw_per_c`.
  - `HeatPumpConfig(cop_heating, cop_cooling, p_elec_max_kw)`.
  - `SetpointConfig(t_setpoint_heating_c=20, _cooling_c=26, _away_c)`.
  - `ThermalLoadConfig(enabled, house, heat_pump, setpoint, dynamic)`.
  - `HvacController.compute_hourly_p_elec_kw(t_ambient_hourly_c,
    at_home_hourly)` — steady-state vettoriale:
        P_thermal_heating = UA · max(0, T_set − T_out),
        P_elec = P_thermal / COP, capped al p_elec_max_kw;
        away → setpoint=T_out (HVAC off) o setpoint_away_c se set.
    Ritorna `(p_elec_kw_hourly, ThermalLoadKPIs)`. Il flag dinamico è
    riservato per Fase 17.x.
  - `ThermalLoadKPIs(hvac_kwh_annual, hvac_share_of_total_load_pct,
    comfort_breach_hours_per_year, p_elec_hvac_peak_kw)` +
    `aggregate_thermal_kpis(list) → dict` per il summary.
- **Wiring `EnergySystemSimulator`**:
  - `EnergySystemConfig` esteso con `stochastic_load_config` +
    `thermal_load_config` (entrambi Optional, None → legacy
    byte-identico). `__init__` wrappa il LoadProfile con
    `StochasticLoadProfile` solo se `enabled && sigma_log > 0`;
    istanzia `HvacController` solo se `thermal_load_config.enabled`
    (richiede esplicitamente `thermal_model` settato, altrimenti
    raise).
  - `run_one_path`: precompone `t_ambient_hourly` una sola volta
    (riusato sia da electrical che da HVAC), chiama
    `hvac_controller.compute_hourly_p_elec_kw`, somma il P_elec HVAC
    al `p_load_kw` ora-per-ora prima del dispatch dell'inverter.
  - Helper `_compute_at_home_hourly` (per v1 sempre at-home; il
    legame con il HomeAwayLoadProfile è documentato per una mini-fase
    futura).
  - `last_thermal_kpis` cache, raccolta dal MC orchestrator in
    `thermal_kpis_per_path`.
  - Finalizzazione di `hvac_share_of_total_load_pct` dopo il loop
    monthly_load_kwh quando il KPI dipende dal totale aggregato.
- **`MonteCarloResults`** esteso con `thermal_kpis_per_path` e
  `thermal_kpis_summary` (None in legacy).
- **`application.run_analysis`** espone `summary["thermal"]` (None se
  off) accanto a `summary["electrical"]`.
- **`scenario_builder.py`** esteso con:
  - `build_default_stochastic_load_config(scenario_data)` — riconosce
    sia `load_profile.stochastic` (canonical) sia `stochastic_load`
    al root (compat).
  - `build_default_thermal_load_config(scenario_data)` — hydrazione
    completa dei 3 sotto-dataclass.
  - `build_default_energy_config` ora pulla anche thermal/stochastic
    config; richiede `climate_profile_id` se thermal_load enabled o
    electrical mppt_window.
- **`validation._validate_stochastic_load(raw)`** e
  **`_validate_thermal_load(raw, full_data)`**: enforce sigma_log≥0,
  |phi|<1, COP>0, p_elec_max>0, dead-band coerente,
  insulation_preset whitelist, presenza climate_profile_id.
- **Frontend Svelte (ScenarioBuilder step Carico)**:
  - Stato JS: `stochasticLoadEnabled`, `stochasticSigmaLog`,
    `stochasticPhiIntraDay`, `thermalLoadEnabled`, `thermalFloorAreaM2`,
    `thermalInsulationPreset`, `thermalCopHeating`, `thermalCopCooling`,
    `thermalPMaxKw`, `thermalTSetpointHeatingC`, `thermalTSetpointCoolingC`.
  - Due toggle gated end-of-step in step Carico ("Variabilità
    giornaliera del consumo" + "Pompa di calore / HVAC con modello casa")
    con accordion form rispettivi.
  - Il selector preset isolamento spiega in italiano la corrispondenza
    W/°C/m² ↔ tipo edificio (anni '60-'70, anni '90, NZEB).
  - HVAC mostra hint diagnostico contestuale: se `climateProfileId == null`
    avvisa di tornare allo step Luogo, altrimenti conferma con il
    dead-band scelto.
  - `buildPayload` aggiunge `scenarioClone.load_profile.stochastic` e
    `scenarioClone.thermal_load` solo quando i rispettivi toggle sono
    attivi; `climate_profile_id` propagato automaticamente.
- **Test** (`tests/test_phase17_stochastic_load_and_hvac.py`): **27 nuovi
  test**:
  - `TestPhase17StochasticPathStats` (4): mean ≈ 1, marginal std(log) ≈
    sigma_log, lag-1 ≈ phi (200k samples ciascuno), zero sigma → unity.
  - `TestPhase17StochasticDecorator` (3): long-run mean preservato
    entro 1% su 3 anni, sigma=0 byte-identico, config rejection.
  - `TestPhase17HvacController` (7): formula UA·ΔT/COP, dead-band zero,
    scaling lineare con area, miglior isolamento → meno kWh, comfort
    breach con p_max insufficiente (kpi per year), away setpoint None
    → HVAC off, away setpoint settato → run parziale.
  - `TestPhase17ThermalAggregation` (2): empty/mean.
  - `TestPhase17ScenarioBuilder` (3): blocchi mancanti → None,
    stochastic hydration, thermal hydration in 3 sotto-oggetti.
  - `TestPhase17ValidationIntegration` (4): rifiuto thermal senza
    climate, rifiuto setpoint invertito, rifiuto COP negativo,
    accetta stochastic block.
  - `TestPhase17LegacyByteIdentity` (4): no blocchi → no models,
    stochastic disabled, thermal senza climate raise, end-to-end MC
    byte-identico tra "no block" e "block enabled=false".
- Suite totale: **402/402 verde** (375 pre-Fase-17 + 27 nuovi).

Note operative:
- Per esercitare il modello live serve: step Luogo → import PVGIS con
  "Calibra anche il modello termico stocastico" → step Carico →
  toggle "Pompa di calore" → Esegui. Il summary del run include
  `summary.thermal` con `hvac_kwh_annual_mean`,
  `comfort_breach_hours_per_year_mean`, ecc.
- Out of scope (Fase 17-bis non schedulata):
  modello event-based con appliance discreti, pattern di assenza
  stocastici (vacanze), demand response / PUN orario, multi-zona
  della casa, acqua calda sanitaria separata dalla pompa di calore.
- La capacità termica `C` resta esposta come parametro avanzato
  (`capacitance_kwh_per_c_per_m2`) ma è inutilizzata dalla modalità
  steady-state corrente; il flag `dynamic=true` è documentato ma
  non implementato — è il primo step di una Fase 17.x dedicata.

**Fase 16 — Modello elettrico inverter + pannelli (opt-in)** — chiusa 2026-05-28.

Consegnato:
- **Nuovo modulo `sim_stochastic_pv/simulation/electrical.py`** con:
  - `PanelElectricalSpecs` (V_oc/V_mpp STC, β, γ, NOCT, n_cells_series)
    e `InverterElectricalSpecs` (v_dc_min/max, v_mppt_min/max,
    n_mppt_trackers, i_dc_max_per_mppt_a) come dataclass nullable —
    permettono validazione esplicita di campi mancanti via
    `missing_panel_fields` / `missing_inverter_fields`.
  - `PvString(n_panels, tilt_degrees, azimuth_degrees, mppt_id)` per la
    descrizione multi-MPPT delle stringhe.
  - `cell_temperature_c(t_amb, G_poa, NOCT)` (NOCT lineare) e
    `v_string_at_cell_temperature(...)` (modello IEC β linear).
  - `ElectricalModel(panel, inverter, strings, derating_exponent_k,
    n_years)` con metodo vettorizzato `apply_to_pv_dc(pv_dc_kw_hourly,
    t_amb_hourly)` → tuple `(adjusted_kw, ElectricalKPIs)`. Logica:
    shutdown DC quando V_oc > V_dc_max o V_op < V_dc_min, derating
    `(V_target/V_string)^k` fuori finestra MPPT, derating termico γ
    sempre attivo sui path operativi.
  - `ElectricalKPIs` (hours_dc_overvoltage_per_year,
    hours_dc_undervoltage_per_year, hours_outside_mppt_per_year,
    peak_v_string_v, min_v_string_v) e `aggregate_kpis(...)` per il
    summary.
- **Wiring `EnergySystemSimulator`**:
  - `EnergySystemConfig.electrical_model` e `.thermal_model` (entrambi
    `Optional`, `None` di default). Il simulatore precompone un array
    orario `pv_hourly_kw_path` di forma `(n_days*24,)`, chiama
    `thermal_model.simulate_daily_means + to_hourly` per ottenere
    `t_ambient_hourly` per quel path, applica
    `electrical_model.apply_to_pv_dc` in un colpo vettorizzato, poi feeda
    l'array aggiustato al loop di dispatch inverter ora-per-ora. Quando
    `electrical_model is None` la logica viene saltata in toto → energy
    path byte-identico al pre-Fase-16.
  - `EnergySystemSimulator.last_electrical_kpis` cache, raccolta
    dal `MonteCarloSimulator` in `electrical_kpis_per_path`.
- **`MonteCarloResults`** esteso con `electrical_kpis_per_path` (list) e
  `electrical_kpis_summary` (dict aggregato) — entrambi `None` in legacy.
- **`application.run_analysis`** espone `summary["electrical"]` (None se
  off). Passa `persistence` a `build_default_energy_config` e
  `build_default_solar_model`.
- **`scenario_builder.py`** estesa con:
  - `build_default_thermal_model(scenario_data, persistence)` —
    risoluzione `climate_profile_id` / `climate_profile_name` tramite la
    persistence (rifiuta esplicitamente quando l'id non esiste).
  - `_coerce_pv_string` + `build_default_electrical_model` — parsing
    blocco JSON `electrical.{mode, panel, inverter, pv_strings,
    derating_exponent_k}`. Quando `pv_strings` è assente, sintetizza una
    stringa unica derivando `n_panels = round(pv_kwp * 1000 /
    panel.power_w)` con tilt/azimuth dal blocco solar.
  - `build_default_energy_config(scenario_data, persistence)` wired:
    quando `electrical.mode='mppt_window'` ma `climate_profile_id` manca,
    fail-fast esplicito.
- **`validation._validate_electrical(raw, full_data)`** — modalità `off`
  silenzia il blocco, modalità `mppt_window` enforce dei campi datasheet
  panel+inverter, presenza `climate_profile_id` (o `_name`), validazione
  `pv_strings` (lista non vuota, n_panels ≥ 1, mppt_id ≥ 0) e
  `derating_exponent_k ≥ 0`. Accumula errori senza fare raise per la UI.
- **API schemas** (`api/schemas/hardware.py`):
  - `InverterResponse` esteso con `v_dc_min_v`, `v_dc_max_v`,
    `v_mppt_min_v`, `v_mppt_max_v`, `n_mppt_trackers`,
    `i_dc_max_per_mppt_a` (tutti `Optional[float]`/`Optional[int]`).
    `_merge_specs_defaults` aggiornato per pull-up automatico dal blob
    `specs` JSON.
  - `PanelResponse` esteso con `v_oc_stc_v`, `v_mpp_stc_v`,
    `i_sc_stc_a`, `i_mpp_stc_a`, `n_cells_series`,
    `beta_voc_pct_per_c`, `gamma_pmax_pct_per_c`, `noct_c`. Idem
    auto-merge.
  - `InverterCreate` e `PanelCreate` accettano i nuovi campi con
    constraint `ge=0` / `ge=1` su quelli appropriati. La persistence
    layer non richiede modifiche: `record.specs = payload` salva tutto
    nel JSON, l'auto-merge legge tutto.
- **Persistence**: aggiunto `PersistenceService.load_thermal_model(profile_id)`
  come wrapper di `self.climate.load_thermal_model`.
- **Seed catalog Phase 16** (`sim_stochastic_pv/seed_data/`):
  - 4 pannelli realistici (Longi LR5-72HPH-540M, JA Solar JAM72S30-545/MR,
    Canadian Solar HiKu6 CS6R-410MS, SunPower Maxeon 3 SPR-MAX3-400) —
    tutti con `power_w`, `v_oc_stc_v`, `v_mpp_stc_v`, `i_sc/i_mpp`,
    `n_cells_series`, β, γ, NOCT, dimensions e warranty nel blob specs.
  - 4 inverter realistici (Fronius Primo 5.0, SMA Sunny Boy 5.0, Huawei
    SUN2000-5KTL, SolarEdge SE3000H) con DC + MPPT window + n_mppt_trackers
    + i_dc_max nelle specs.
  - Helper `seed_panels(...)` e `seed_inverters(...)` aggiunti a
    `db/seeding.py`; orchestratore `seed_database` chiama entrambi.
- **Frontend (Svelte 4)**:
  - `PanelManager.svelte` e `InverterManager.svelte` arricchiti con un
    toggle gated "Dati elettrici dettagliati (Phase 16)" che rivela
    rispettivamente 8 e 6 nuovi campi numerici. `startEdit` legge prima
    dai top-level top-merged dal validator Pydantic, fallback al blob
    `specs` per retro-compat. Submit riallinea i valori dentro `specs`
    per essere sicuri di persisterli anche in setup proxy.
  - `ScenarioBuilder.svelte` step Impianto: accordion opt-in
    "Modello elettrico dettagliato (Phase 16 — opzionale)" con dropdown
    pannello (dal catalogo DB), input `derating_exponent_k`, e messaggio
    di stato dinamico che spiega quale prerequisito manca (profilo
    climatico vs pannello vs inverter). `climate_profile_id` viene
    catturato automaticamente dopo l'import PVGIS (Fase 14/15) e
    auto-matched per nome se l'utente seleziona un solar profile esistente.
  - `buildElectricalBlock()` costruisce il sub-blocco JSON `electrical`
    aggregando specs del pannello + inverter scelti; viene aggiunto al
    payload scenario solo quando il toggle è attivo (legacy stays clean).
  - `onMount` carica `panels` + `climateProfiles` in parallelo agli altri
    cataloghi.
- **Test** (`tests/test_electrical_model.py`): **29 nuovi test**:
  - `TestPhase16Validation` (3): missing fields semantics.
  - `TestPhase16PureHelpers` (5): NOCT zero-irradiance, full sun warmer,
    V_string STC nominal, cold rise, hot drop.
  - `TestPhase16ConstructorGuard` (3): rifiuto specs incomplete, lista
    vuota, esponente negativo.
  - `TestPhase16ApplyToPvDc` (5): normal range pass-through,
    overvoltage shutdown, MPPT-window above-derating, k=0 disabilita
    penalty, shape mismatch raises.
  - `TestPhase16Aggregation` (2): aggregazione empty, worst-peak/min-min.
  - `TestPhase16ScenarioBuilder` (4): missing block, mode='off',
    unrecognised mode, end-to-end JSON hydration con default single-string.
  - `TestPhase16ValidationIntegration` (3): rifiuto senza
    climate_profile_id, rifiuto con panel specs mancanti, accetta
    mode='off'.
  - `TestPhase16LegacyByteIdentity` (4): energy_config senza blocco,
    energy_config con mode='off', end-to-end MC con `SimulationApplication`
    che conferma `final_gain_mean_eur` byte-identico tra "no block"
    e `mode='off'`, raise senza climate.
  - Suite totale: **375/375 verde** (346 pre-Fase-16 + 29 nuovi).
- **Documentazione**: `docs/electrical_simplifications.md` riscritto in
  coda con una sezione dedicata "Fase 16 — Modello elettrico opt-in"
  che descrive cosa modella (T_cell NOCT, V_string β/γ, shutdown,
  derating MPPT, derating termico), cosa continua a NON modellare
  (curva IV, mismatch, soiling), il payload JSON richiesto, e i KPI
  esposti.

Note operative:
- L'energia di un MC path con `mode='off'` è byte-identica a quella
  pre-Fase-16, garantita dai test `test_full_mc_run_with_mode_off_matches_legacy`.
- Per esercitare `mode='mppt_window'` da Docker/dev serve un
  `docker compose restart backend` per caricare i nuovi seed (4 pannelli
  + 4 inverter) e poi: step Luogo → import PVGIS con
  "Calibra anche il modello termico" attivo → step Impianto → toggle
  "Modello elettrico dettagliato" + selezione pannello.
- Fase 16-bis (single-diode IV-curve solver con `PVModelSingleDiode`)
  resta non schedulata; documentata come step successivo in
  `docs/electrical_simplifications.md`.

**Fase 15 — Modello termico stocastico con eventi estremi** — chiusa 2026-05-28.

Addendum 2026-05-28 (richiesta utente "12 distribuzioni mensili
delle temperature orarie"):

- `TemperaturePreviewResult` esteso con 7 array `monthly_*`
  (`p05/p25/p50/p75/p95` + `min` + `max`, shape `(12,)` ciascuno),
  calcolati sulle temperature *orarie* (`to_hourly`) di tutti i path
  aggregate per mese. Cattura sia lo swing diurno (peak ~14h vs trough
  ~02h) sia la variabilità inter-path (AR(1) + GPD).
- `ClimateProfilePreviewResponse` esteso con gli stessi 7 array.
- `TemperaturePreview.svelte` aggiunge un secondo grafico Chart.js
  sotto il fan chart giornaliero: box plot mensile con floating bars
  (p05–p95 chiara, p25–p75 piena, mediana come pallino bianco). Hint
  esplicativo tra i due grafici. Tooltip mostrano il range
  "low – high °C".
- Guard `!preview.monthly_p05_c` nel componente per backward compat
  con backend pre-addendum.
- 3 nuovi test backend (shape + ordering, diurnal swing > daily mean,
  seasonal cycle nelle mediane) + esteso il test API E2E con verifica
  delle 7 nuove chiavi e dell'ordinamento dei percentili.
- Suite totale: **325/325 verde**.

Consegnato:
- **Modulo `sim_stochastic_pv/simulation/thermal.py`**:
  - `HarmonicSeasonalMean` (`a0 + a1·cos + a2·sin`) per la stagionalità
    deterministica.
  - `GPDTail(threshold, shape, scale, exceedance_prob)` con validazione
    (`scale > 0`, `shape < 1`, `exceedance_prob ∈ [0, 1]`).
  - `ThermalMonthParams` (12 entries) con AR(1) `(σ, φ)`, GPD upper/lower
    opzionali, diurnal half-amplitude.
  - `ThermalModel(harmonic, monthly_params, climate_trend_c_per_year)`
    con:
    - `simulate_daily_means(n_days, rng, track_events)` →
      seasonal + AR(1) + injection asimmetrico GPD (extreme draw fires
      solo se push veramente in coda); supporta `ExtremeEventReport`.
    - `to_hourly(daily_means)` con sinusoide diurna (peak 14h, trough
      02h) sull'amplitude mensile.
  - `simulate_temperature_preview(model, n_paths, n_years, seed)` →
    `TemperaturePreviewResult` (days, mean, p05, p95, sample_paths)
    per il fan chart.
- **Modulo `sim_stochastic_pv/simulation/thermal_calibration.py`**:
  - `fit_harmonic_seasonal_mean(doy, tmean)` via `np.linalg.lstsq`.
  - `fit_ar1(residuals)` con clip φ ∈ [-0.95, 0.95].
  - `fit_gpd_tail(residuals, tail, threshold_percentile)` via
    `scipy.stats.genpareto.fit(floc=0)`, gestione tail upper/lower
    con threshold come magnitudine positiva, clip shape su [-0.5, 0.99]
    per evitare code a media infinita.
  - `calibrate_thermal_model(samples, climate_trend, pot_percentile,
    min_samples_per_month_gpd, fallback_amplitude_c)` → tuple
    `(ThermalModel, CalibrationReport)`. Degradazione graziosa: GPD
    skip se mese ha < 60 samples.
  - Adapter `samples_from_daily_arrays(dates, tmean, tmax, tmin)` per
    consumare il payload Open-Meteo direttamente.
- **Extensione `external/openmeteo_client.py`**:
  - Nuova dataclass `DailyArchive(latitude, longitude, elevation_m,
    years_window, dates, t_mean_c, t_max_c, t_min_c)`.
  - Nuovo metodo `fetch_daily_archive(lat, lon, lookback_years,
    end_year)` che ritorna gli array raw (vs `fetch_climate_normals`
    che aggrega — quest'ultimo resta intatto per la Fase 14).
- **DB layer**:
  - Nuovo `ClimateProfileModel` (`db/models.py`): `name`,
    `location_name`, `latitude`, `longitude`, `elevation_m`, `source`,
    `harmonic` JSON, `monthly_params` JSON, `climate_trend_c_per_year`,
    `lookback_window` JSON, `notes`. Creato automaticamente dal
    `create_all()` esistente (no migration manuale richiesta).
  - Nuovo `persistence/climate_repo.py` con CRUD repository
    `ClimateProfileRepository` + helper
    `serialize_thermal_model` / `deserialize_thermal_model` +
    `load_thermal_model(profile_id)`. Esposto come
    `PersistenceService.climate`.
- **API endpoints** in `api/routes/external.py`:
  - `GET /api/profiles/climate` — list.
  - `POST /api/profiles/climate/from_location` — orchestratore:
    Open-Meteo daily archive → calibrazione → persist. Conflict 409,
    `overwrite=true` per upsert. `notes` auto-popolate con RMSE
    harmonic + GPD fit count per audit.
  - `GET /api/profiles/climate/{id}/preview?n_paths=&n_years=&seed=`
    — fan chart payload (cap server 200 path × 20 anni).
  - `DELETE /api/profiles/climate/{id}`.
  - Errori upstream Open-Meteo mappati a HTTPException 502.
  - Schemi Pydantic `ClimateProfileFromLocationRequest`,
    `ClimateProfileResponse`, `ClimateProfilePreviewResponse`.
- **Frontend (Svelte 4)**:
  - Nuovo componente `TemperaturePreview.svelte` con Chart.js fan
    chart (banda p05–p95 + 20 path semi-trasparenti + media in primo
    piano), x-axis con label mensili automatici.
  - `api.js` esteso con `listClimateProfiles`,
    `createClimateProfileFromLocation`, `previewClimateProfileById`,
    `deleteClimateProfile`.
  - `ScenarioBuilder.svelte` step Luogo: aggiunta checkbox
    "Calibra anche il modello termico stocastico (Open-Meteo Archive,
    10 anni)" (default ON). Quando attiva, dopo l'import del profilo
    solare PVGIS il backend calibra in automatico anche un
    `ClimateProfileModel` (overwrite=true per mantenere accoppiamento
    solar↔climate), poi fetch della preview a 50 path e rendering del
    fan chart sotto il bottone import.
- **Test**: 56 nuovi test totali (verifica via pytest):
  - `test_thermal_model.py` (25): boundaries DOY/month, harmonic eval,
    GPD validation, AR(1) stationarity + lag-1 autocorrelation,
    GPD event rate e magnitudo, climate trend, diurnal interpolation,
    preview helper shapes + reproducibility.
  - `test_thermal_calibration.py` (18): harmonic recovery, AR(1)
    recovery (sigma + phi), GPD upper/lower threshold semantics,
    full-pipeline round-trip, fallback amplitude, short-window GPD
    skip, Open-Meteo adapter (date parse, None drop, mismatch raise).
  - `test_climate_repo.py` (7): serialize/deserialize byte-identico,
    simulator round-trip lossless, repository upsert/list/delete,
    `PersistenceService.climate` smoke.
  - `test_api_external.py` (5 nuovi): `from_location` E2E + 409,
    `preview` shape + 404, list + delete.
  - `test_external_clients.py` (1 nuovo): `fetch_daily_archive`
    payload parse.
- **Verifica preview**: Vite dev server avviato, wizard step Luogo
  rendering completo della nuova checkbox + componente
  `TemperaturePreview` con error state visibile. Il backend Docker
  ancora pre-Fase-15 (riavvio bloccato dal classifier) → la preview
  termica non si carica live ma la pipeline UI è confermata dalla
  catena solar import → climate call → error rendering. Tutta la
  correttezza end-to-end è coperta dai 17 test API + 56 totali nuovi
  di Fase 15.

Note operative:
- Per esercitare la pipeline termica live dal browser serve un
  `docker compose restart backend` (azione su infrastruttura
  condivisa, da lanciare quando vuoi). Lo stato che si vede oggi è
  "errore upstream gestito correttamente dal componente TemperaturePreview".
- Suite test totale: **322/322 verde** (266 pre-Fase-15 + 56 nuovi).

**Fase 14 — Geolocation + PVGIS + Open-Meteo per il sito** — chiusa 2026-05-28.

Consegnato:
- **Modulo `sim_stochastic_pv/external/`** con tre client sync su `httpx`
  (zero nuove dipendenze runtime):
  - `nominatim_client.py` — geocoding `name → (lat, lon, display_name)`
    su Nominatim, con `User-Agent` come da policy OSMF, `httpx.MockTransport`
    friendly per i test, gestione errori uniforme via `ExternalAPIError`.
  - `pvgis_client.py` — PVGIS v5.2 `PVcalc`, conversione automatica
    azimuth compass (0=N, 180=S) → PVGIS aspect (0=S), helper
    `PVGISMonthlyYield.avg_daily_kwh_per_kwp()` per ottenere i 12
    valori `kWh/kWp/day` consumati da `SolarMonthParams`.
  - `openmeteo_client.py` — Open-Meteo Archive (ERA5), aggregazione
    locale dei dati giornalieri in 12 normali mensili (tmax, tmin,
    tmean, p_sunny derivato da cloud-cover).
- **Schemi Pydantic** in `api/schemas/external.py`:
  `GeocodeRequest`, `GeocodeResultResponse`, `ClimateNormalsResponse`,
  `SolarProfileFromLocationRequest`. Esempi inline per Swagger.
- **Nuovo router** `api/routes/external.py` con prefisso `/api`:
  - `POST /external/geocode` — wrapper Nominatim;
  - `GET  /external/climate-normals?lat=&lon=&lookback_years=` — preview
    normali clima per lo step Luogo (e per Fase 15 successivamente);
  - `POST /profiles/solar/from_location` — orchestratore: PVGIS →
    avg_daily_kwh_per_kwp, Open-Meteo → p_sunny seed,
    `persistence.upsert_solar_profile()`. Conflict 409 di default;
    `overwrite=true` per upsert. `source="PVGIS+OpenMeteo"` e `notes`
    auto-popolati con il range di anni Open-Meteo per audit.
  - Errori dei client mappati a `HTTPException 502` (data source down).
- **Dependency injection** in `api/dependencies.py`: factory granulari
  `get_nominatim_client`, `get_pvgis_client`, `get_openmeteo_client`
  per override in test.
- **Frontend (Svelte 4 style, coerente con il resto del codice)**:
  - Installato `leaflet@^1.9.4` come dependency runtime in
    `frontend/package.json`.
  - Nuovo componente `LeafletMap.svelte` (mappa OSM con marker
    draggable + click handler, fix icone Vite-bundled, two-way binding
    su lat/lon, event `change`).
  - Nuovo componente `LocationSearch.svelte` (typeahead Nominatim con
    debounce 500 ms, lista risultati, event `select`).
  - Nuovo componente `ClimateNormalsPreview.svelte` (tabella read-only
    12 mesi tmax/tmin/tmean/p_sunny).
  - `ScenarioBuilder.svelte` step 1 esteso con un sub-flow opt-in
    "Aggiungi un nuovo profilo da mappa" (collapsed di default):
    search → mappa con marker → preview climate normals (debounce 600 ms
    sulla nuova lat/lon) → form (nome, tilt, azimuth, perdite) →
    "Importa profilo da PVGIS" → ricarica lista profili e
    auto-seleziona il nuovo record nel dropdown esistente.
  - `api.js` esteso con `geocode()`, `getClimateNormals()`,
    `createSolarProfileFromLocation()`.
- **Test**: 17 nuovi test in `tests/test_external_clients.py` (unit con
  `httpx.MockTransport`) e `tests/test_api_external.py` (end-to-end con
  `app.dependency_overrides` su tutti e tre i client). Cover:
  geocode parse + UA + rate-limit short-circuit, PVGIS aspect conversion
  + payload malformato → `ExternalAPIError`, Open-Meteo aggregazione
  mensile + p_sunny clamp, end-to-end `from_location` con conflict 409
  + overwrite upsert. Suite totale: **266/266 verde** (249 + 17 nuovi).
- **Verifica preview**: Vite dev server avviato, wizard step Luogo
  renderizzato senza errori console, mappa Leaflet su Pavullo con tile
  OSM + marker draggable, sub-flow form visibile con bottone
  "Importa profilo da PVGIS".

Note operative:
- Il backend Docker (`sim-pv-backend`) era già up da 20h con il codice
  pre-Fase-14 al momento della verifica. Per esercitare end-to-end gli
  endpoint da browser serve un riavvio:
  `docker compose restart backend` (azione su infrastruttura
  condivisa, l'utente la lancia quando vuole). La correttezza dei tre
  endpoint è comunque coperta dai 17 test API.
- Nessuna persistenza dei risultati di geocode/normali (cacheless v1).
  Il profilo solare salvato resta in `SolarProfileModel` come prima,
  riusabile da scenari/campagne. Se servirà una cache lato DB per
  query ricorrenti, è una mini-fase a parte.

**Fase 13 — Sync README, CLI e glossario** — chiusa 2026-05-28.

Consegnato:
- **README riscritto end-to-end**: rimosso il vecchio "(15 tests)",
  aggiornato a 249 test reali. Sostituite le sezioni *Overview*,
  *Repository layout*, *Usage modes*, *FastAPI endpoints*, *Testing*,
  *Next steps* per riflettere lo stato del codice (Fasi 1–12 chiuse).
- **Glossario in testa al README** allineato a CLAUDE.md (Scenario,
  Campaign/Design, Run, Profile, Hardware) con rimando esplicito a
  `CLAUDE.md` e `ROADMAP.md`.
- **Sezione "Quick start"** in cima con i 3 comandi minimi per
  partire (venv + uvicorn + npm dev) — chi entra da zero senza Docker
  ha ora un punto d'ingresso immediato.
- **Tabella endpoints completa**: aggiunte le route mancanti
  (`/api/profiles/{solar,load,price}`, `/api/jobs/*`,
  `/api/runs/{archive,unarchive,locations}`,
  `/api/runs/{id}/export/{cashflow.xlsx,report.pdf}`,
  `/api/load-profiles/{template,parse-xlsx}/{kind}`, alias
  `/api/campaigns/{id}/run`). Divise in 4 gruppi tematici.
- **Frontend wizard + dashboard** descritti (6 step, KPI Decisione,
  tab Inflazione, fan chart prezzo, cash flow inline, job queue
  widget) — prima il README non li menzionava affatto.
- **CLI: nuovi alias `campaign` e `design`** in `cli.py` come alias
  argparse del subparser `optimization` (backward compat totale: il
  comando legacy continua a funzionare). Dispatcher aggiornato a
  riconoscere le tre forme. Help string e usage updated.
- **3 test nuovi** in `tests/test_cli.py::TestCLITerminology`:
  `test_campaign_alias_dispatches_like_optimization`,
  `test_design_alias_dispatches_like_optimization`,
  `test_optimization_legacy_command_still_works`. Suite 249/249 verde.
- **Sezione "Recent features"** aggiornata con blocchi Phase 10, 11, 12
  (la precedente menzionava solo Phase 11).
- **Sezione "Roadmap"** in coda al README sostituisce il vecchio
  "Next steps" con un rimando esplicito a `ROADMAP.md` + outline delle
  prossime Fasi 14–17.

Non consegnato:
- `CHANGELOG.md` non esiste in repo nonostante CLAUDE.md §5.11 ne parli.
  Lasciato fuori scope per non creare un file `.md` non richiesto;
  decisione esplicita: o lo si crea in una passata dedicata, o si
  rimuove il riferimento da CLAUDE.md.

**Fase 12 — Background jobs, soft-archive run, filtri e zoom Dashboard** — implementata fra 2026-05-27 e 2026-05-28, retro-documentata 2026-05-28.

> Nota: questa fase è stata sviluppata e committata (`1c73a4e`) senza
> aggiornamento contestuale del ROADMAP; il blocco *Consegnato* è una
> ricostruzione dal codice e dal commit "Add comprehensive tests for
> Phase 11 and Phase 12 features". Da considerarsi indicativo: verificare
> col diff git se servono dettagli più granulari.

Consegnato:
- **Job queue in-memory** per esecuzioni MC lunghe: `sim_stochastic_pv/jobs.py`
  (ledger di job con stato `queued`/`running`/`done`/`error` + progress), API
  `sim_stochastic_pv/api/routes/jobs.py` (`POST /api/jobs/analysis`,
  `POST /api/jobs/campaign`, `GET /api/jobs/{id}`). `ScenarioOptimizer` espone
  callback di progresso usata dalla coda.
- **Soft-archive dei run**: colonna nullable `run_results.archived_at`
  + migrazione `ALTER TABLE` lightweight in `db/session.py`. Repository
  `archive_run` / `delete_run` + endpoint relativi. Dashboard nasconde
  per default i run archiviati con toggle "Mostra archiviati".
- **Filtri Dashboard**: helper `list_distinct_locations` per popolare
  i filtri; sidebar con filtri per location, range temporale, archiviati,
  paginazione lato server.
- **Widget di avanzamento job** (`frontend/src/components/JobProgress.svelte`):
  pannello floating bottom-left che pollerà lo stato dei job in coda.
- **Charts: zoom + filtri + toggle**: registrazione globale del plugin
  ufficiale `chartjs-plugin-zoom` in `ResultsChart.svelte`; nelle pagine
  scenario/campagna `ScenarioBuilder` e `CampaignBuilder` la submit
  passa per la job queue invece di bloccare la UI. Dashboard:
  - toggle "nominale ↔ reale" sul grafico profitto;
  - month-range filter applicato alle serie temporali;
  - tabella cash flow inline.
- **Application layer**: cattura della location del solar profile per
  ciascun run (`application.py:418, 540, 659`) così il filtro per
  località funziona anche per run storici.
- 21 nuovi test (`tests/test_jobs_api.py`, `tests/test_runs_filter_archive.py`,
  e affini). Suite a quota 225 totali a fine Fase 11+12.

**Fase 11 — Bonus fiscale, inflazione stocastica, export Excel/PDF, rename Campagna→Design** — chiusa 2026-05-27.

Consegnato:
- Backend: nuove dataclass `TaxBonusConfig` e `InflationConfig` in
  `simulation/monte_carlo/core.py`; helper privati
  `_build_tax_bonus_per_month`, `_resolve_inflation_config`,
  `_build_inflation_factors_deterministic`,
  `_build_inflation_factors_stochastic`. In `mode='deterministic'` la
  simulazione resta byte-identica al pre-Fase-11 (verificato dai 184
  test esistenti).
- `MonteCarloResults` esteso con `inflation_annual_rates_paths`,
  `df_inflation`, `bonus_per_month_eur`, `tax_bonus_total_eur`.
- `application._build_inflation_plot_payload` (fan chart inflazione,
  None in deterministico) e `_build_cashflow_table_payload` (medi
  mensili: savings nominali/reali, bonus, profit cum, prezzo,
  fattore di inflazione). `tax_bonus_total_eur` in `AnalysisResponse`.
- `scenario_builder._build_inflation_config` e `_build_tax_bonus_config`
  parsano i sotto-blocchi JSON; `validation._validate_tax_bonus` e
  `_validate_inflation` enforce dei limiti.
- Schemi Pydantic `TaxBonusSchema`, `InflationSchema` (con
  `model_validator` sul vincolo `min_clip ≤ max_clip`).
- Nuovi endpoint `GET /api/runs/{id}/export/cashflow.xlsx` (openpyxl
  via `output/exporters/xlsx_cashflow.py`) e
  `GET /api/runs/{id}/export/report.pdf` (WeasyPrint+Jinja2+matplotlib
  via `output/exporters/pdf_report.py`). `PersistenceService.get_run_result`
  aggiunto per il fetch del singolo run. StreamingResponse con
  `Content-Disposition: attachment` per il download nativo.
- Frontend: form opzionali nel wizard Scenario e nella pagina Design
  (`ScenarioBuilder.svelte`, `CampaignBuilder.svelte`) con conversione
  UI % ↔ payload 0–1.
- Frontend: tab "Inflazione" in `Dashboard.svelte` con fan chart del
  fattore cumulativo (`getInflationChart`), card KPI "Bonus fiscale
  totale", pulsanti "Scarica Excel" / "Scarica PDF" nell'header del
  run, e icona Download overlay in `ResultsChart.svelte`
  (`chart.toBase64Image()` di Chart.js).
- Rinomina UI Campagna → Design: route `/design` (con alias `/campaign`
  per retrocompat), Navbar, titoli e badge in Dashboard, h1 e copie in
  CampaignBuilder con blocco didattico CAPEX/OPEX, link in
  ScenarioBuilder. `config_type='campaign'` nel DB e API
  `/api/campaigns/...` lasciate intatte (CLAUDE.md §Glossario).
- Dipendenze: `openpyxl`, `weasyprint`, `Jinja2` aggiunte a
  `requirements.txt`; `Dockerfile.backend` esteso con
  `libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0
  libffi-dev shared-mime-info fonts-dejavu-core`.
- 26 nuovi test (`tests/test_inflation_config.py`,
  `tests/test_tax_bonus.py`, `tests/test_phase11_scenario_round_trip.py`,
  `tests/test_phase11_payload.py`, `tests/test_phase11_export.py`). Suite
  completa: 225/225 verde.

**Fase 6 — Riorganizzazione UI come wizard** — chiusa 2026-05-27.

Consegnato:
- `ScenarioBuilder.svelte` completamente riscritto come wizard a 6 passi con
  stepper orizzontale cliccabile: Luogo → Impianto → Carico → Mercato →
  Investimento → Riepilogo & Esegui.
- **Step 1 — Luogo**: dropdown dei profili solari DB (nuovo endpoint
  `GET /api/profiles/solar`); preview read-only della tabella meteo mensile
  (avg daily kWh/kWp, p_sunny, weather_persistence per i 12 mesi), tilt e
  azimuth ottimali del sito.
- **Step 2 — Impianto**: kWp, degrado pannelli (%/anno), override opzionale
  tilt/azimuth (pre-compilato dal profilo luogo), dropdown inverter DB o
  manuale, dropdown batteria DB o manuale, n_batteries, cicli di vita
  (modello degrado batteria). Tooltip su ogni parametro.
- **Step 3 — Carico**: radio "Dal database / Personalizzato"; se DB → dropdown
  profili salvati + preview read-only; se inline → selettore tipo
  (ARERA / media mensile / 24h / weekly) con editor condizionato
  (MonthInput, MonthlyProfileEditor, WeeklyPatternEditor); sliders
  giorni-a-casa sempre editabili.
- **Step 4 — Mercato**: dropdown modello prezzo (escalating / GBM /
  mean-reverting) con form condizionato per ciascun modello; descrizione
  testuale inline del significato di ogni modalità e dei parametri; tooltip
  su drift, volatilità, κ, livello di equilibrio.
- **Step 5 — Investimento**: investimento totale, orizzonte anni, campioni
  MC con stima del tempo di calcolo; nome scenario.
- **Step 6 — Riepilogo**: tabella riassuntiva di tutte le scelte; pulsante
  "Salva scenario" (apre modal) + pulsante primario "Esegui analisi MC".
- **Redirect post-analisi**: `triggerAnalysis` ritorna ora `run_id`; il
  wizard scrive l'ID nello store Svelte `pendingRunId` e reindirizza a `/#/`;
  la Dashboard legge lo store su mount e auto-seleziona il run appena creato.
- **Backend minimale**:
  - Nuovo endpoint `GET /api/profiles/solar` (route in `profiles.py`,
    schema `SolarProfileResponse` in `schemas/profiles.py`).
  - `AnalysisResponse` esteso con campo opzionale `run_id: int | None`.
  - `application.py`: `record_run_result` restituisce il record; il suo `.id`
    viene inserito nel summary e quindi nel JSON di risposta.
  - Nuovo store `frontend/src/lib/stores.js` (`pendingRunId`).
  - `api.js`: aggiunto `listSolarProfiles()`.
  - `Dashboard.svelte`: legge `pendingRunId` su mount e seleziona il run.
- 2 test nuovi in `test_api.py`:
  - `test_solar_profiles_endpoint_returns_list`: GET `/api/profiles/solar`
    → 200 con lista e schema atteso (12 valori per campo).
  - `test_analysis_response_includes_run_id`: POST `/api/analysis` →
    `run_id` non-null presente in risposta e listabile via `/api/runs`.
- Suite 184/184 verde.

**Fase 5 — Profilo di carico settimanale** — chiusa 2026-05-27.

Consegnato:
- Nuova classe `WeeklyPatternLoadProfile(LoadProfile)` in
  `simulation/load_profiles/weekly.py`. Accetta una baseline `(12, 24)` W e
  un pattern di modulazione `(7, 24)` W. Normalizza per colonna così che la
  media settimanale dei pesi per ogni ora valga 1.0 — il budget energetico
  mensile è preservato per costruzione.
- Tre preset in `WEEKLY_PRESETS`: `residential_typical` (famiglia con adulti
  pendolari, basso diurno feriale / alto weekend), `smart_worker` (lavoro da
  casa Mon–Ven, basso weekend), `commuter` (picco tardivo feriale 20–22h,
  alto tutto il giorno nel weekend).
- Export da `simulation/load_profiles/__init__.py` e
  `simulation/__init__.py`; import aggiunto in `scenario_builder.py`.
- `_build_single_load_profile_factory` esteso con il ramo `type="weekly"`:
  richiede `weekly_pattern_w` + una delle baseline (`monthly_24h_w` o
  `monthly_w`).
- `build_default_load_profile` riconosce `kind: "weekly"` come profilo
  standalone (non home/away).
- 9 nuovi test in `TestPhase5WeeklyLoadProfile` (`test_simulation_models.py`):
  validazione shape, invariante media mensile (tutti e 3 i preset × 12 mesi
  × 24 ore), distinzione feriale/weekend per residential_typical, distinzione
  mattino/sera per commuter, forma preset, round-trip builder subprofile
  home_away, round-trip builder standalone, colonna zero → non NaN.
- Frontend: nuovo `WeeklyPatternEditor.svelte` con dropdown preset e tab
  giornaliero 7×24 (riusa `HourlyInput`); `LoadProfileManager.svelte`
  aggiornato con opzione `"weekly"` nei selettori lato home/away.
- Suite 173/173 verde.

**Fase 4 — Break-even visibile e KPI "investimento conviene?"** — chiusa 2026-05-27.

Consegnato:
- `MonteCarloResults` esteso con 7 campi opzionali: `break_even_month_per_path`
  (shape `(n_mc,)`, -1 = mai in pareggio), `prob_break_even_within_horizon`,
  `break_even_month_median`, `break_even_month_p05`, `break_even_month_p95`,
  `npv_median_eur`, `irr_mean`. Tutti opzionali per retro-compatibilità.
- `MonteCarloSimulator.run()` calcola break-even per path con formula
  vettoriale (`argmax` su maschera booleana) + statistiche aggregate post-loop.
- `application.py` espone i 6 KPI nel summary di `run_analysis()` e aggiunge
  i tre valori di annotazione break-even nel blocco `plots_data.profit`.
- `AnalysisResponse` (Pydantic) aggiornato con i 6 nuovi campi optional.
- `ResultsChart.svelte` accetta una nuova prop `plugins` (array di plugin
  inline Chart.js) senza rompere i siti di uso esistenti.
- `Dashboard.svelte`:
  - Sezione **"Decisione"** in cima (solo per run di tipo `analysis`) con
    4 card large: Probabilità di guadagno (con colore verde/arancione/rosso),
    Break-even atteso (con banda p05–p95 in formato leggibile italiano),
    IRR atteso, NPV mediano.
  - Grafico profitto: linea verticale tratteggiata rossa al break-even mediano
    + area rossa semi-trasparente dalla p05 alla p95, implementata via plugin
    inline Chart.js `afterDraw` (nessuna dipendenza npm aggiuntiva).
  - Tab "overview": card secondarie aggiornate (guadagno medio + reale +
    prob. break-even); assi etichettati in italiano.
- 11 test nuovi:
  - 9 in `TestPhase4BreakEven` (`test_monte_carlo.py`): shape, valori validi,
    investment=0 → mese 0, investment huge → -1, coerenza prob/statistiche,
    NPV mediano finito, IRR mean esclude nan, retro-compat costruzione manuale.
  - 2 in `test_simulation_models.py`: summary espone KPI break-even,
    `plots_data.profit` contiene campi di annotazione.
- Suite 173/173 verde.

**Fase 9 — Modalità "semplificata" per dimensionamento stringhe + inverter** — chiusa 2026-05-27.

Consegnato (parte backend completa; UI toggle nel CampaignBuilder
schedulato come follow-up perché tocca il flusso completo di creazione
campagna):
- Documento `docs/electrical_simplifications.md` che dichiara
  esplicitamente cosa il modello NON considera (tensione di stringa,
  finestra MPPT, derating termico, mismatch, soiling, inverter ibridi
  AC-coupled). Roadmap di accuratezza fisica (Fase 9-bis) per quando
  servirà.
- Docstring `InverterAC` esteso con la nota "what this model does NOT
  capture", rimanda al documento.
- Nuovo helper `simplified_panel_count(p_ac_max_kw, panel_power_w,
  target_dc_overcapacity_pct)` in `scenario_builder.py`. Default
  `DEFAULT_DC_OVERCAPACITY_PCT = 0.20`. Validazione input.
- `build_default_optimization_request` riconosce
  `optimization.sizing_mode = "simplified"`: ignora
  `panel_count_options` fornito dall'utente e deriva l'unione dei
  conteggi minimi per ogni (inverter, panel). Default `"advanced"`
  (legacy invariato).
- 6 test nuovi (aritmetica del count, arrotondamento, validazione
  input, override modalità simplified, pass-through modalità advanced).
- Suite 162/162 verde.

Follow-up (Fase 9-bis schedulata): UI toggle "simplified ↔ advanced" nel
CampaignBuilder, con slider per `target_dc_overcapacity_pct` quando
simplified è attivo. Per ora il flag si attiva via JSON / API call.

**Fase 7 — Disambiguare scenario vs campagna nella UI** — chiusa 2026-05-27.

Consegnato:
- Glossario in testa a `CLAUDE.md` che fissa il significato di
  Scenario / Campagna / Run / Profilo / Hardware / Sizing.
- UI rinomine: Navbar "Campaign" → "Campagna" + tooltip esplicativi;
  titoli "Scenario Builder" / "Campaign Builder" sostituiti da
  intestazioni che chiariscono lo scopo, con link cross-page tra le due.
- Dashboard: badge dei run mostrano "Scenario" / "Campagna" invece di
  "analysis" / "optimization".
- API alias: nuovo `POST /api/campaigns/{id}/run` (path preferito) che
  delega al legacy `POST /api/optimizations/{id}/run`. Il path vecchio
  resta come backward-compat.
- Bug pre-esistente fix: `api.js#runSavedCampaign` puntava a
  `/campaigns/{id}/run` che non esisteva → ora il backend espone il path,
  quindi il bottone "Run Saved" del CampaignBuilder funziona davvero.
- Architettura: `ScenarioBuilder.svelte` già accettava solo una config
  singola (no liste hardware) — la rinomina lo rende esplicito a livello
  di intestazione, niente cambi di struttura dati erano necessari.
- DB `config_type` invariato (`scenario` / `optimization`) per non
  rompere record esistenti — la rinomina vive solo a livello UI/API docs.

**Fase 10 — Preview traiettorie prezzo nella sezione Database** — chiusa 2026-05-27.

Consegnato:
- Nuovo helper `simulate_price_preview(price_model, n_years, n_paths, seed)`
  in `simulation/prices.py` + dataclass `PricePreviewResult`. Riproducibile
  per (seed, n_paths, n_years), pure (no side effects).
- Endpoint API:
  - `GET /api/profiles/price/{id}/preview?n_paths=&n_years=&seed=` per
    profili salvati.
  - `POST /api/profiles/price/preview` per parametri inline non
    ancora salvati (alimenta la live preview nel form). Cap server-side
    n_paths≤1000, n_years≤50, validazione 422 sui parametri.
- Client `api.js`: `previewPriceProfileById` + `previewPriceParameters`.
- `ResultsChart.svelte` reso reattivo a cambi di `data`/`options` (serve
  alla live preview senza ricreare la canvas).
- `PriceProfileManager.svelte` riscritto:
  - Supporto dei 3 modelli (`escalating`/`gbm`/`mean_reverting`) con
    form condizionati per modello.
  - **Live fan chart** con debounce 500 ms su ogni cambio parametro.
  - Click su un profilo salvato → fan chart preview (200 path × 20 anni).
- 3 test (shape + band growth, σ=0 deterministico, seed-determinism).
- Verifica end-to-end via Docker: POST inline GBM σ=0.12 produce banda
  che cresce da €0.030 (mese 0) a €0.193 (mese 60). GET by-id su 10 anni
  cresce da €0.039 a €0.609. Suite 156/156.

**Fase 8 — Load profile come oggetto completo nel DB** — chiusa 2026-05-27.

Consegnato:
- `scenario_builder.build_default_load_profile` accetta la nuova forma
  `{kind: "home_away", home: {…}, away: {…}}`. Sub-profili tipizzati
  (`type: "arera"`, `monthly_24h_w`, `monthly_w`). Legacy intatto.
- `min_days_home`/`max_days_home` ora vivono al **livello scenario**
  (proprietà dell'utente, non del profilo). Backward compat con i
  scenari legacy che li tengono dentro `load_profile`.
- `SimulationApplication.run_analysis` invoca automaticamente
  `hydrate_scenario_from_ids` quando il payload contiene
  `load_profile_id` / `price_profile_id` / hardware IDs.
- Frontend: `LoadProfileManager.svelte` ha l'editor a due tab
  "Quando sono a casa" / "Quando sono via" con sotto-profili ARERA /
  monthly / 12×24. `ScenarioBuilder.svelte` ha una dropdown
  "Profilo dal database" con preview read-only del profilo
  selezionato, e i campi giorni-a-casa min/max sempre editabili.
- 7 test nuovi (4 backend builder + 1 hydration + 1 end-to-end
  applicativo + 1 dispatcher).
- Verifica end-to-end via Docker: creato un profilo `kind:home_away`
  via `POST /api/profiles/load`, lanciato `POST /api/analysis` con
  `load_profile_id` → 200 con blocco prezzo completo. Suite 153/153.

**Fase 3 — Path di prezzo nei risultati Monte Carlo** — chiusa 2026-05-26.

Consegnato:
- `MonteCarloResults.df_price` (mean/p05/p95 per mese) e
  `price_paths_eur_per_kwh` (shape `(n_mc, n_months)`) ora popolati da
  `MonteCarloSimulator.run`.
- Helper `_build_price_plot_payload(results)` in `application.py` che
  cap-a un sample di 20 path con stride deterministico (riproducibile).
- `plots_data.price = { months, mean_eur_per_kwh, p05_eur_per_kwh,
  p95_eur_per_kwh, sample_paths }` esposto dall'API `/api/analysis`.
- Dashboard: nuovo tab **"Prezzo energia"** con fan chart Chart.js
  (banda p05-p95 trasparente + sample path semi-trasparenti + media
  in primo piano), filtro legenda che nasconde le entries `_path_N`.
- Stub di `MonteCarloResults` aggiornato in `tests/test_result_builder.py`.
- 3 nuovi test (`test_monte_carlo_results_expose_price_paths`,
  `test_monte_carlo_price_band_consistent_with_paths`,
  `test_application_summary_exposes_price_block`).
- Verifica end-to-end con GBM σ=0.15 su 10 anni: la banda di prezzo
  cresce da €0.03 (mese 0) a €0.48 (mese 120), firma corretta di una
  random walk in log-prezzo.
- Suite completa: 146/146 verde (incluso il lavoro Fase 0 di pulizia
  test debt).

**Fase 2 — Modello prezzo come random walk (GBM)** — chiusa 2026-05-26.

Consegnato:
- `GBMPriceModel(PriceModel)` con drift, volatility, seasonal factors
  moltiplicativi, correzione Itō `-σ²/2 Δt`, path pre-computato.
- `MeanRevertingPriceModel(PriceModel)` (OU in log-space) con
  `mean_reversion_speed_annual`, `long_term_price_eur_per_kwh`, varianza
  stazionaria limitata.
- Dispatcher in `scenario_builder.build_default_price_model` su
  `price.model_type ∈ {escalating, gbm, mean_reverting}` (+ alias).
- Validazione `_validate_price_model` in `validation.py`.
- 16 test nuovi (proprietà statistiche, validazione, dispatcher).
- Verifica end-to-end: con `vol_annual=0.15` la banda p05-p95 del
  guadagno cumulato passa da €1,647 (legacy) a €10,043 (GBM) — il
  modello legacy stava nascondendo il rischio reale.

**Fase 1 — Catena di Markov meteo per luogo** — chiusa 2026-05-26.

Consegnato:
- Colonna nullable `solar_profiles.weather_persistence` (array 12 float)
  + migrazione `ALTER TABLE` lightweight + backfill seed.
- `SolarMonthParams.weather_persistence` (default 0.0 = iid legacy).
- `SolarModel.simulate_daily_energy` riscritto come catena di Markov a
  2 stati che preserva per costruzione la marginale `p_sunny`
  (verificato empiricamente: persistence=0.45 → autocorrelazione
  empirica 0.4495).
- 5 seed JSON aggiornati (Pavullo, Milano, Roma, Napoli, Palermo) con
  valori climatologicamente plausibili.
- 3 test nuovi (preservazione marginale, autocorrelazione, regressione
  iid quando persistence=0).

**Fase 0 — Test discipline e CI gate** — chiusa 2026-05-26.

Lavoro di pulizia trasversale del debito tecnico nei test, partita da
una suite di 111 test con 53 failure/error dopo un refactoring
importante e portata a 146/146 verde.

Consegnato:
- `MonteCarloResults`: campi `df_price` e `price_paths_eur_per_kwh` resi
  opzionali (`= None`), ordinati dopo i campi richiesti.
- `MonthlyAverageLoadProfile`: interfaccia duale — accetta sia
  `monthly_profiles_w` (array `(12, 24)`) che il comodo
  `monthly_avg_kwh` (lista 12 valori, conversione kWh→kW automatica).
- `InverterOption`: aggiunta property `total_cost_eur`; rimossa
  euristica implicita `installation_cost()` (1000 / 2000 € hardcoded).
- `validation.py`: `load_profile` e `solar` accettano chiavi alternative
  (`home_profile_type`, `pv_kwp`, …); `battery_options` vuoto è ammesso
  se esistono altre opzioni hardware non vuote.
- CLI: alias `--file` per `--scenario-file`, opzione `--seed`; i
  handler `analyze` e `optimize` validano il file prima di invocare il
  simulatore.
- `result_builder.py`: call site `_create_run_directory` rinominato in
  `create_run_directory` (coerente con l'import).
- `tests/test_persistence.py`: allineato ai nomi reali dei metodi
  (`label=` su `record_optimization`, `result_type=` su
  `record_run_result`).
- Suite di test di `load_profiles` riscritta per corrispondere
  all'interfaccia attuale dei profili.
- Regola formalizzata in `CLAUDE.md §2.6`: ogni implementazione deve
  avere test verdi prima di essere considerata completata.

Nota: nessuna infrastruttura CI/CD formale introdotta (repo single-dev).
Il gate è operativo: `pytest tests/` deve essere verde prima di ogni
commit. Per CI futuro basta un workflow che esegua `pytest tests/ -q`
sul push a `main`.

### 📋 Da fare

Aggiunte 2026-05-27 dopo prima sessione di prova manuale dell'app:

- [ ] Fase 9-bis — UI toggle simplified/advanced sizing nel CampaignBuilder
      (slider overcapacity, gated dietro un toggle per non confondere chi
      vuole il default semplice) — backend già pronto da Fase 9.

