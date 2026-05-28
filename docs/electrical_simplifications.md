# Semplificazioni elettriche del simulatore

Questo documento elenca **cosa il modello attualmente NON considera** sul lato
elettrico/fotovoltaico, perché è stato scelto questo livello di astrazione, e
in che ordine sarebbe ragionevole rilassare le approssimazioni in futuro.

L'obiettivo del simulatore è la **valutazione economica** di un impianto
residenziale: massima trasparenza sui parametri che muovono davvero il
guadagno (kWp installato, profilo di carico, prezzo dell'energia, dimensione
batteria), modello fisico minimo ma onesto sul resto.

> **Aggiornamento 2026-05-28 (Fase 16 chiusa)** — la "Fase 9-bis"
> originariamente schedulata in coda a questo documento è stata implementata
> come **Fase 16**: l'utente può adesso opt-in al modello elettrico
> dettagliato (finestra MPPT, derating termico cella, shutdown V_dc) tramite
> il blocco JSON `electrical.mode='mppt_window'`. Sezione dedicata in coda
> al documento. Quando il blocco è assente o `mode='off'`, il simulatore
> mantiene il comportamento byte-identico pre-Fase-16 — i punti elencati
> qui sotto si applicano alla modalità predefinita.

---

## Cosa NON è modellato (al 2026-05-27)

### 1. Tensione di stringa e finestra MPPT dell'inverter

`InverterAC` (`sim_stochastic_pv/simulation/inverter.py`) espone soltanto due
parametri:

- `p_ac_max_kw` — limite di potenza AC in uscita,
- `p_dc_max_kw` — cap opzionale in ingresso DC.

**Non c'è alcun calcolo della tensione DC della stringa.** Nessun valore di
`v_oc`, `v_mpp`, `v_min_mppt`, `v_max_mppt` viene letto, propagato o
controllato. Di conseguenza:

- Il parametro `n_panels_per_string` di `CampaignBuilder` influenza **solo
  il CAPEX** (più pannelli = più costo): il modello energetico vede solo la
  potenza DC complessiva del campo (`n_panels × power_w × orientation_factor
  × degradation_factor`).
- L'inverter non viene mai "spento" perché la stringa esce dalla sua
  finestra MPPT.
- Effetti di temperatura sulla tensione di circuito aperto / di lavoro
  non sono presi in conto.

### 2. Modello fisico single-diode dei pannelli

La classe `PVModelSingleDiode` (`sim_stochastic_pv/simulation/pv_model.py`)
**esiste** ma non è collegata al simulatore. È un modello solo dei parametri
elettrici di un pannello (Iph, I0, Rs, Rsh, ecc.), pronto per future
integrazioni quando arriveranno dei datasheet veri nel DB pannelli.

### 3. Mismatch di stringa, ombreggiamenti parziali, soiling

- Pannelli leggermente diversi nella stessa stringa → no effetto modellato.
- Ombre parziali (ostacolo che copre un solo pannello) → no, il campo è
  trattato come un blocco unico modulato dal `SolarMonthParams` e dai
  fattori di orientamento (`compute_orientation_factor`).
- Soiling (sporco/polvere) → non modellato; in alternativa l'utente può
  includere una piccola perdita nel `degradation_per_year`.

### 4. Derating termico

La produzione mensile `avg_daily_kwh_per_kwp` è già "calibrata" sulla
realtà locale (PVGIS o equivalente), quindi include implicitamente il
derating termico medio. **Non c'è** un calcolo orario di temperatura del
pannello → coefficiente di temperatura → potenza istantanea.

### 5. Inverter ibridi e accoppiamento AC/DC della batteria

Il modello tratta l'inverter come un dispatcher con limiti AC e DC.
L'accoppiamento batteria-inverter è uno solo (DC), non c'è distinzione tra
inverter ibridi DC-coupled e sistemi AC-coupled con un secondo inverter
per la batteria.

---

## Modalità di dimensionamento

A partire da Fase 9 il simulatore espone due modalità di dimensionamento
del campo pannelli rispetto all'inverter:

### Modalità "simplified" (default)

L'utente non sceglie `n_panels_per_string`. Indica invece:

- **`target_dc_overcapacity_pct`** (default `0.20` = +20%): quanta
  sovrappotenza DC vuole rispetto al limite AC dell'inverter.

Il numero minimo di pannelli viene calcolato come:

```
n_panels = ceil(p_ac_max_kw * (1 + target_dc_overcapacity_pct) * 1000 / power_w)
```

`n_panels_per_string` è impostato implicitamente a `n_panels` (tutto su una
sola stringa), coerentemente con l'**assenza** di logica MPPT: separare in
più stringhe non avrebbe alcun effetto sul calcolo energetico.

### Modalità "advanced"

L'utente specifica direttamente `n_panels_per_string` (e quindi il numero di
pannelli totali nella campagna). Modalità pensata per quando il simulatore
verrà esteso con il calcolo MPPT, in modo che lo sweep su numero di pannelli
per stringa abbia un effetto fisico misurabile.

In assenza di logica MPPT, *advanced* e *simplified* danno gli stessi risultati
energetici a parità di `n_panels` totali — differiscono solo nel CAPEX (perché
il numero di pannelli può essere diverso).

---

## Roadmap di accuratezza fisica

Quando l'utente vorrà più realismo:

- **Fase 16 — chiusa 2026-05-28**: modello elettrico dettagliato opt-in
  (vedi sezione "Fase 16" qui sotto). Sostituisce la Fase 9-bis che era
  segnata come "futura non schedulata" in questo documento.
- Step successivo (Fase 16-bis, non schedulata): collegare
  `PVModelSingleDiode` al simulatore per la curva IV completa, sostituendo
  l'approssimazione lineare β/γ con un solver implicito per accuratezza
  fisica del kWh prodotto in condizioni estreme (mismatch, soiling).
- Step ulteriore: modelli di soiling stagionale (perdita ~3-5%/anno
  recuperabile con manutenzione).

Per ora, la priorità del progetto resta sull'incertezza **economica**
(prezzo energia random walk, persistenza meteo, profilo di carico
realistico), non sulla precisione **fisica** del singolo kWh prodotto.

---

## Fase 16 — Modello elettrico opt-in (chiusa 2026-05-28)

A partire dalla Fase 16 il simulatore può girare in due regimi:

| `electrical.mode` | Comportamento |
|---|---|
| `"off"` (default) o blocco assente | Comportamento byte-identico pre-Fase-16: tutto quanto elencato in "Cosa NON è modellato" sopra resta valido. |
| `"mppt_window"` | Il simulatore calcola per ogni ora `T_cell`, `V_string` open-circuit e operativo, applica derating MPPT-window e spegne l'inverter quando `V_oc > V_dc_max`. |

### Quello che la modalità `mppt_window` modella

1. **Cell temperature**: `T_cell = T_amb + (NOCT − 20)/800 · G_poa` con
   irradianza proxy dalla potenza DC istantanea.
2. **String voltage**: `V_string = N · V_panel(T_cell)`, con
   `V_panel(T) = V_stc · (1 + β/100 · (T − 25))` per V_oc e V_mpp.
3. **Shutdown DC**: quando `V_oc > V_dc_max` o `V_operativa < V_dc_min`
   l'inverter viene spento — la potenza oraria scende a 0 e l'ora è
   contata in `hours_dc_overvoltage_per_year` (o `_undervoltage`).
4. **Derating MPPT-window**: quando `V_operativa` esce dalla finestra
   MPPT (`v_mppt_min`–`v_mppt_max`) la potenza viene moltiplicata per
   `(V_target / V_string)^k` con `k=0.5` di default (esposto come
   `electrical.derating_exponent_k`).
5. **Derating termico**: `P × (1 + γ/100 · (T_cell − 25))` applicato
   in modo moltiplicativo su tutto l'orario produttivo.

### Quello che la modalità `mppt_window` continua a NON modellare

- **Curva IV completa**: il modello usa l'approssimazione lineare β/γ del
  datasheet, non risolve l'equazione single-diode. Per quello servirà
  Fase 16-bis. `PVModelSingleDiode` resta scollegato dal simulatore.
- **Mismatch di stringa**: tutte le stringhe sono trattate come uniformi.
  Diversità di moduli nella stessa stringa (Imax → Imin) non è modellata.
- **Inverter ibridi DC/AC-coupled separati**: la batteria resta lato DC
  dello stesso inverter, come in modalità `off`.
- **Soiling, ombreggiamenti parziali**: invariati rispetto al regime base.
- **Multi-MPPT vero**: il modello accetta più stringhe via
  `electrical.pv_strings[].mppt_id`, ma calcola un V_string
  "rappresentativo" pesato sui pannelli — un solo inverter con due
  stringhe uguali su due MPPT tracker dà gli stessi numeri di uno con
  una stringa unica della stessa dimensione totale.

### Requisiti per attivare `mppt_window`

Lo scenario JSON deve contenere:

```jsonc
{
  // ... resto dello scenario invariato ...
  "climate_profile_id": 7,                     // Phase 15, fonte di T_ambient
  "electrical": {
    "mode": "mppt_window",
    "panel": {                                 // dal datasheet del pannello
      "power_w": 540.0,
      "v_oc_stc_v": 49.5,
      "v_mpp_stc_v": 41.5,
      "n_cells_series": 144,
      "beta_voc_pct_per_c": -0.27,
      "gamma_pmax_pct_per_c": -0.34,
      "noct_c": 45.0
    },
    "inverter": {                              // dal datasheet dell'inverter
      "v_dc_min_v": 80.0,
      "v_dc_max_v": 1000.0,
      "v_mppt_min_v": 240.0,
      "v_mppt_max_v": 800.0,
      "n_mppt_trackers": 2
    },
    "pv_strings": [                            // opzionale; default = stringa unica
      {"n_panels": 10, "tilt_degrees": 35, "azimuth_degrees": 180, "mppt_id": 0}
    ],
    "derating_exponent_k": 0.5                  // opzionale, default 0.5
  }
}
```

`validation.py` rifiuta lo scenario con messaggi puntuali se manca uno
qualsiasi dei campi obbligatori (`v_oc_stc_v`, `v_mpp_stc_v`, …) o se
`climate_profile_id` non è presente.

### KPI esposti in `summary.electrical`

Quando il modello è attivo, il summary API espone:

- `hours_dc_overvoltage_per_year_mean` — media sui path delle ore/anno
  in cui l'inverter è stato spento per overvoltage. KPI di rischio
  hardware: >5 indica dimensionamento aggressivo della stringa.
- `hours_dc_undervoltage_per_year_mean` — analogo sulla soglia inferiore.
- `hours_outside_mppt_per_year_mean` — ore/anno in cui il tracker era
  operativo ma fuori finestra MPPT (penalizzate dal derating soft).
- `peak_v_string_v` — V_oc massimo osservato (caso peggiore tra i path).
  Diagnostico, deve restare comodamente sotto `v_dc_max_v`.
- `min_v_string_v` — V_operativo minimo osservato.

### Catalogo pannelli/inverter con dati elettrici

Il seed catalog Phase 16 include quattro pannelli (Longi LR5-72HPH-540M,
JA Solar JAM72S30-545/MR, Canadian Solar HiKu6 CS6R-410MS, SunPower
Maxeon 3 SPR-MAX3-400) e quattro inverter (Fronius Primo 5.0, SMA Sunny
Boy 5.0, Huawei SUN2000-5KTL, SolarEdge SE3000H) tutti con specifiche
elettriche complete nel blob `specs`. Database già popolati possono
estendere voci esistenti tramite il toggle "Dati elettrici dettagliati"
in `InverterManager` e `PanelManager` lato UI.
