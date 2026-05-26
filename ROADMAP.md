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

## Dipendenze fra fasi

```
Fase 2 ─┐
        ├─→ Fase 3 ──→ Fase 4
Fase 1 ─┘                   ↑
                            │
Fase 5 ─────────────────────┤
                            │
Fase 6 ◄────── tutte ───────┘
```

## Stato

- [x] Fase 1 — in corso (al 2026-05-26)
- [ ] Fase 2
- [ ] Fase 3
- [ ] Fase 4
- [ ] Fase 5
- [ ] Fase 6
