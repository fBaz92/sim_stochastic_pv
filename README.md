## Simulatore Stocastico PV

Questo progetto esegue simulazioni stocastiche (Monte Carlo) per valutare impianti fotovoltaici domestici con diverse combinazioni di inverter, pannelli e batterie. Il flusso principale consente di ottimizzare la configurazione confrontando decine di scenari e generando report dettagliati.

### Requisiti

- Python 3.11+ (testato)
- Virtualenv o strumento equivalente

### Setup veloce

```bash
python -m venv venv
source venv/bin/activate  # oppure .\venv\Scripts\activate su Windows
pip install -r requirements.txt
```

> Nota: la repo include uno `requirements.txt` implicito nella virtualenv (`venv`). Se manca, rigenera le dipendenze usando `pip freeze > requirements.txt` dopo l’installazione.

### Esecuzione analisi singola

```bash
source venv/bin/activate
python main_analysis.py
```

Il comando:

- Simula il sistema base definito in `scenario_setup.py`
- Esegue 100 percorsi Monte Carlo (parametro modificabile in `build_default_economic_config`)
- Salva grafici + rapporto testuale dentro `results/<timestamp>_home_away_default/`

### Ottimizzazione scenari

```bash
source venv/bin/activate
python main_optimization.py
```

Questo script:

1. Genera tutte le combinazioni definite in `build_default_optimization_request`
2. Esegue le simulazioni Monte Carlo per ogni configurazione
3. Seleziona i migliori scenari (top 10 + copertura per inverter e batteria)
4. Produce grafici comparativi (profitto, distribuzioni, break-even)
5. Salva report dettagliati per:
   - Scenario con migliore break-even
   - Scenario con guadagno finale massimo

Tutti gli output vengono raccolti sotto `results/<timestamp>_home_away_default_batch/`.

### Configurazione scenari

Modifica `scenario_setup.py` per cambiare:

- Parametri dei pannelli (`PanelOption`)
- Opzioni inverter (`InverterOption`)
- Batterie disponibili (`BatteryOption`)
- Numero di Monte Carlo per scenario (`build_default_economic_config(n_mc=...)`)
- Profili di carico e modello solare

### Note sui grafici

- I grafici dei profitti mostrano:
  - Barra superiore con gli anni simulati
  - Punto di break-even per ogni scenario (se raggiunto) con etichetta “BE anno X”
  - Guadagno massimo con etichetta del valore in euro

### Risoluzione problemi comuni

- **Matplotlib non può scrivere la cache**: impostare `export MPLCONFIGDIR=$(pwd)/.cache/matplotlib`
- **Fontconfig non scrive cache**: esportare `XDG_CACHE_HOME` verso una directory scrivibile
- **Simulazione lenta**: ridurre `n_mc` o limitare `panel_count_options` in `scenario_setup.py`

### Struttura cartelle principale

- `sim_stochastic_pv/`: libreria con simulatori, modelli e reporting
- `main_analysis.py`: esegue un singolo scenario
- `main_optimization.py`: gira il batch di ottimizzazione
- `results/`: output dei run (grafici, csv, report)
- `scenario_setup.py`: parametri e richieste di ottimizzazione

### Contribuire / Estendere

1. Fork o nuova branch
2. Aggiorna setup e parametri in `scenario_setup.py`
3. Aggiungi test o script di validazione se introduci nuove logiche
4. Apri una PR descrivendo le modifiche

Buone simulazioni!
# sim_stochastic_pv
