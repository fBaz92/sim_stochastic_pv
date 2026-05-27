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
```

## Stato

> Questo è il *log vivente* dell'evoluzione del progetto. Aggiornare
> ogni volta che si entra o si esce da una fase (regola formalizzata in
> `CLAUDE.md §4`). Le date sono quelle effettive di inizio/fine sessione,
> non quelle "ideali" del piano.

### 🚧 In corso

Nessuna fase attivamente in corso.

### ✅ Completate

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
