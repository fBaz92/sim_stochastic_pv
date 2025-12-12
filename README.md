## Simulatore Stocastico PV

Simulatore Monte Carlo per sistemi fotovoltaici domestici. Il progetto offre:

- motore di simulazione riusabile via libreria Python;
- linea di comando con due modalità (analisi singola e ottimizzazione batch), che salva i risultati nella cartella `results/`;
- API FastAPI per integrare l’engine in una futura web app;
- database (SQLite di default, PostgreSQL opzionale) per storicizzare componenti, scenari e run;
- suite di test automatizzati.

---

## Requisiti

- Python 3.11+
- Pip/virtualenv
- Dipendenze opzionali:
  - PostgreSQL (se si vuole usare un RDBMS esterno)
  - `uvicorn` per avviare l’API

---

## Installazione

```bash
git clone <repo>
cd sim_stochastic_pv
python -m venv venv
source venv/bin/activate  # oppure .\venv\Scripts\activate su Windows
pip install -r requirements.txt  # oppure `pip install -e .` se usi un setup personalizzato
```

Se `requirements.txt` non è aggiornato, rigeneralo con `pip freeze > requirements.txt` dopo aver installato i pacchetti desiderati (`fastapi`, `uvicorn`, `sqlalchemy`, `pytest`, ecc.).

---

## Configurazione

1. **Database**
   - Genera un file `.env` nella root del progetto.
   - Per PostgreSQL definisci `POSTGRES_DSN`, ad esempio:
     ```
     POSTGRES_DSN=postgresql+psycopg://user:password@host:5432/database
     ```
   - In assenza di `POSTGRES_DSN`, l’app creerà automaticamente un SQLite locale (`sim_pv.db`).

2. **Directory risultati**
   - I report (grafici, CSV, testi) vengono salvati in `results/` **solo quando avvii il comando da CLI**.
   - L’API non genera output su disco: salva soltanto i riassunti nel database.

3. **Cache Matplotlib/Fontconfig (solo se richiesto dall’ambiente)**
   - Se ricevi errori su directory non scrivibili, esporta:
     ```bash
     export MPLCONFIGDIR=$(pwd)/.cache/matplotlib
     export XDG_CACHE_HOME=$(pwd)/.cache
     ```

---

## Modalità di utilizzo

### 1. Libreria Python

Puoi importare direttamente i componenti principali:

```python
from sim_stochastic_pv import SimulationApplication, ResultBuilder

app = SimulationApplication(save_outputs=False)
summary = app.run_analysis()
```

### 2. Linea di comando

```bash
source venv/bin/activate
python -m sim_stochastic_pv.cli analyze        # Analisi singolo scenario
python -m sim_stochastic_pv.cli optimize       # Ottimizzazione batch
python -m sim_stochastic_pv.cli analyze --no-save   # Esegui senza salvare file
```

- `analyze`: esegue una simulazione Monte Carlo dello scenario base definito in `scenario_setup.py`.
- `optimize`: genera tutte le combinazioni di componenti, esegue le simulazioni e produce report per lo scenario con break-even più rapido e quello con guadagno finale maggiore.
- Entrambi i comandi popolano il database (componenti, scenari, risultati) e, salvo `--no-save`, scrivono i report in `results/<timestamp>_<scenario>`.

### 3. API FastAPI

Avvia l’API con uvicorn:

```bash
source venv/bin/activate
uvicorn api_main:app --reload
```

Endpoints principali (tutti sotto `/api`):

| Metodo | Endpoint       | Descrizione                                  |
|--------|----------------|----------------------------------------------|
| POST   | `/analysis`    | Avvia e restituisce un’analisi singola       |
| POST   | `/optimization`| Avvia l’ottimizzazione multi-scenario        |
| GET    | `/runs`        | Elenca gli ultimi run salvati nel database   |

Esempio chiamata `curl`:

```bash
curl -X POST http://localhost:8000/api/analysis -H "Content-Type: application/json" -d '{"n_mc": 50, "seed": 123}'
```

L’API restituisce solo JSON; nessun file viene scritto su disco.

---

## Architettura principale

- `sim_stochastic_pv/application.py`: orchestratore condiviso da CLI e API.
- `sim_stochastic_pv/result_builder.py`: genera report e grafici per l’utilizzo da linea di comando.
- `sim_stochastic_pv/persistence.py`: interfaccia per salvare componenti, scenari, ottimizzazioni e risultati.
- `sim_stochastic_pv/api/`: app FastAPI, schemi Pydantic e router.
- `sim_stochastic_pv/cli.py`: entry point per la CLI.
- `tests/`: suite pytest.

---

## Testing

La suite automatizzata copre configurazione, persistenza, orchestratore, result builder e API.

```bash
source venv/bin/activate
pytest
```

I test usano SQLite in-memory e directory temporanee, quindi non toccano il database reale né la cartella `results/`.

Per aggiungere nuovi test:

- inserisci le fixture in `tests/conftest.py`;
- documenta ogni test con una docstring che descriva il comportamento verificato;
- assicurati che nuovi moduli/funzioni abbiano docstring chiare (obbligatorio nel progetto).

---

## Personalizzazione scenari

Modifica `scenario_setup.py` per cambiare:

- elenco di inverter/pannelli/batterie (con eventuali datasheet);
- profili di carico (`LoadScenarioBlueprint`);
- modello solare e parametri economici;
- numero di percorsi Monte Carlo (`build_default_economic_config(n_mc=...)`).

Ogni modifica sarà automaticamente riflessa sia nella CLI sia nell’API.

---

## Contribuire

1. Crea una branch o un fork.
2. Aggiungi/aggiorna codice con relative docstring.
3. Esegui `pytest` prima di aprire una PR.
4. Documenta eventuali nuovi comandi/API nel README.

Buone simulazioni!
