# Semplificazioni elettriche del simulatore

Questo documento elenca **cosa il modello attualmente NON considera** sul lato
elettrico/fotovoltaico, perché è stato scelto questo livello di astrazione, e
in che ordine sarebbe ragionevole rilassare le approssimazioni in futuro.

L'obiettivo del simulatore è la **valutazione economica** di un impianto
residenziale: massima trasparenza sui parametri che muovono davvero il
guadagno (kWp installato, profilo di carico, prezzo dell'energia, dimensione
batteria), modello fisico minimo ma onesto sul resto.

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

- **Fase 9-bis** (futura, non schedulata): collegare `PVModelSingleDiode`
  al simulatore, leggere `v_oc`, `v_mpp` dai datasheet pannello, valutare
  la stringa rispetto alla finestra MPPT dell'inverter, applicare
  spegnimento/derating quando la stringa esce dal range.
- Step successivo: derating termico ora-per-ora con un modello T_cell
  semplice (NOCT + irradianza).
- Step ulteriore: modelli di soiling stagionale (perdita ~3-5%/anno
  recuperabile con manutenzione).

Per ora, la priorità del progetto resta sull'incertezza **economica**
(prezzo energia random walk, persistenza meteo, profilo di carico
realistico), non sulla precisione **fisica** del singolo kWh prodotto.
