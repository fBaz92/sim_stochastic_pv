# CLAUDE.md

Documento operativo per chiunque (umano o agente) lavori su questo repository.
Letto prima di ogni sessione, aggiornato quando le regole del progetto cambiano.

---

## Glossario rapido

Termini ricorrenti e cosa significano *in questa codebase*. Tenere allineata
la terminologia in codice, docstring, schemi API, e UI.

| Termine | Significato | Note |
|---|---|---|
| **Scenario** | Analisi economica di **una** configurazione hardware specifica (uno scelto: inverter, batteria, kWp, profilo di carico, modello di prezzo). L'unica stocasticità è il Monte Carlo (meteo, carico, prezzo). | DB: `SavedConfigurationModel.config_type = "scenario"`. UI: "Scenario". |
| **Campagna** | Esplorazione **di design**: un set di scenari su griglie di alternative (più inverter, più kWp, più batterie) per individuare la configurazione ottimale. | DB: `SavedConfigurationModel.config_type = "optimization"` (nome storico, da rinominare *solo* a livello UI/API docs). UI: "Campagna". |
| **Run** | Singola esecuzione del Monte Carlo per uno scenario o una campagna. Persistita in `RunResultRecord`. | Tipi: `analysis` (singolo scenario), `optimization` (campagna). |
| **Profilo (load/price/solar)** | Oggetto DB riutilizzabile riferito via ID in scenari/campagne. | Es. `LoadProfileModel.kind = "home_away"`. |
| **Hardware** | Inverter, pannello, batteria — entità DB con specs. | `InverterModel`, `PanelModel`, `BatteryModel`. |
| **Modalità simplified vs advanced (sizing)** | (Fase 9) modalità di calcolo n. pannelli/stringhe. Simplified = "minimo numero di pannelli per dare DC overcap %"; advanced = utente specifica `n_panels_per_string`. | Default simplified. |

> **Importante**: i `config_type` lato DB (`scenario` / `optimization`) restano
> i nomi storici per non rompere i record già salvati. La rinomina vive solo
> a livello UI/API-docs/dizionario utente.

## 1. Obiettivo del progetto

`sim_stochastic_pv` è un **toolkit Monte Carlo per la valutazione economica
di impianti fotovoltaici residenziali con accumulo**, pensato per rispondere
a una domanda concreta:

> *"Mi conviene investire X mila euro in un impianto PV + batteria, dato il
> mio luogo di installazione, il mio profilo di consumo, e l'incertezza
> del prezzo dell'energia elettrica nei prossimi 20 anni?"*

Per farlo il sistema combina:

- **Modello di produzione PV stocastica** parametrizzato per luogo
  geografico (catena di Markov sunny/cloudy, degrado, orientamento pannelli).
- **Modello di carico** orario, con varianti deterministiche e stocastiche.
- **Modello di prezzo dell'energia** con escalation e volatilità (a regime
  finale: random walk geometrica).
- **Simulatore Monte Carlo** che combina i tre sopra su N path indipendenti.
- **KPI finanziari** (NPV, IRR, profitto cumulato, probabilità di break-even,
  tempo atteso di rientro).

Il prodotto si esprime in **tre superfici**:

1. **Libreria Python** importabile (`sim_stochastic_pv.*`).
2. **CLI** (`python -m sim_stochastic_pv.cli ...`).
3. **Web app**: backend FastAPI + frontend Svelte (servita via Docker).

La web app è la superficie principale per l'utente finale: deve permettere
di definire **luogo, impianto, carico, mercato elettrico, investimento**
in un flusso unico e organico, e di leggere i risultati (probabilità di
guadagno, curva di profitto con break-even, traiettorie di prezzo simulate)
in una dashboard altrettanto unica.

---

## 2. Principi di design — non negoziabili

### 2.1 Orientamento agli oggetti, manutenibilità prima di tutto

Tutto il codice nuovo deve essere scritto in **classi e funzioni piccole,
con responsabilità singola e ben definita**. Il codice qui è destinato a
crescere per anni: ogni scorciatoia oggi è un debito tecnico domani.

- **Una classe = un concetto.** Se una classe inizia a fare due cose
  (es. parametri + simulazione), spezzala.
- **Dataclass per i dati, classi normali per il comportamento.**
  `SolarMonthParams` (dataclass) tiene i numeri, `SolarModel` (classe) fa
  girare il modello. Non mescolare.
- **Composizione, non eredità profonda.** L'ereditarietà è ammessa solo
  per definire interfacce (es. `LoadProfile` astratto). Niente gerarchie
  a più di un livello senza ragione tecnica esplicita.
- **Niente moduli mega-file.** Soglia indicativa: 600 righe per file.
  Oltre, è tempo di spezzare.
- **Niente import circolari**, neppure mascherati da import lazy interni
  a una funzione, a meno che siano l'unica via percorribile (in tal caso
  documenta perché in un commento).

### 2.2 Docstring verbose obbligatori

**Ogni** classe, metodo, funzione, e dataclass pubblica deve avere un
docstring in stile Google esteso. Anche le funzioni "ovvie".

Il docstring deve contenere, in ordine:

1. **Riga di sintesi** (una sola frase, fino a 80 caratteri).
2. **Descrizione estesa** del comportamento, della semantica, del *perché*
   esiste, di quali assunzioni fa, di quali invarianti rispetta.
3. **`Args:`** — per ciascun argomento: tipo, range tipico, unità di misura,
   valore di default, vincoli.
4. **`Returns:`** — tipo, shape (se array), unità, range tipico.
5. **`Raises:`** — solo le eccezioni che davvero solleva (no `Exception`
   generiche).
6. **`Example:`** — almeno uno snippet eseguibile che mostri l'uso tipico.
   Per modelli numerici, includi anche i valori attesi commentati.
7. **`Notes:`** — assunzioni implicite, limiti del modello, riferimenti a
   metodi correlati, gotcha noti.

I docstring sono **per il lettore futuro** (incluso un agente AI che entra
nella codebase a freddo): devono permettere di capire come usare il pezzo
senza leggere il corpo. Se servono formule, scrivile in modo leggibile in
ASCII o LaTeX-like.

**No docstring stringato.** Tre righe vanno bene solo per metodi privati
banali (es. `_clip_to_range`). Tutto il resto è verbose.

### 2.3 Tipizzazione completa

- Annotazioni di tipo su **ogni** parametro e valore di ritorno.
- Usa `from __future__ import annotations` in tutti i file.
- Tipi precisi: `np.ndarray` va bene, ma documenta shape e dtype nel
  docstring. Per Pydantic, schema completi con esempi.
- `Optional[X]` solo quando `None` ha un significato semantico esplicito,
  non come "non ho voglia di pensare al default".

### 2.4 Validazione ai confini, fiducia all'interno

- **Validazione esplicita** al confine fra mondo esterno e modello
  (CLI parser, API Pydantic schemas, JSON loader).
- All'interno del modello le funzioni si fidano dei tipi: niente `assert`
  difensivi nel cuore del simulatore.
- Le eccezioni devono essere informative: messaggio + valori incriminati
  + suggerimento di correzione quando possibile.

### 2.5 Niente magia, niente comportamenti impliciti

- I numeri magici stanno in **costanti modulari con nome**, mai inline.
- I default vanno **dichiarati nel firmatario della funzione**, non
  iniettati da `get()` su dict casuali.
- Configurazione: una sola fonte di verità (`config.py` per env vars,
  schemi Pydantic per i payload, JSON di scenario per i casi).

### 2.6 Test obbligatori per il codice nuovo

**Regola fondamentale: una feature non è "fatta" finché `pytest tests/ -q`
non passa completamente, suite verde. Scrivere i test è parte integrante
dell'implementazione, non un'aggiunta opzionale.**

- Ogni nuovo modulo `sim_stochastic_pv/<area>/<modulo>.py` ha un
  corrispondente `tests/test_<modulo>.py`.
- Test deterministici (sempre `np.random.default_rng(seed)`).
- Per i modelli stocastici, test sulle **proprietà statistiche**
  (media, varianza, autocorrelazione lag-1) con tolleranze esplicite,
  non sui singoli valori.
- Test fast: nessun test deve durare > 5 secondi. Per Monte Carlo
  pesante, usa `n_mc` piccolo e un seed fisso.
- **Mai lasciare test rossi su `main`.** Se un refactoring rompe test
  esistenti, sistemare i test (o il codice) prima di considerare la
  sessione conclusa. Il branch principale è sempre verde.
- **I test devono testare l'interfaccia reale**, non un'interfaccia
  immaginata. Se il test passa argomenti o chiama metodi che non
  esistono nell'implementazione, il test è sbagliato — va allineato
  all'implementazione attuale (o viceversa, se l'implementazione è
  cambiata senza intenzione).
- Quando si modifica una classe/funzione pubblica, controllare subito
  se i test esistenti la coprono ancora: firma, nomi dei parametri,
  valori di ritorno.

---

## 3. Convenzioni di repository

### 3.1 Struttura

```
sim_stochastic_pv/
  simulation/        # Cuore fisico/economico (solar, battery, prices, MC, opt)
  application.py     # Orchestratore alto livello condiviso CLI/API
  scenario_builder.py# Da JSON/DB → oggetti del simulatore
  cli.py             # CLI argparse
  validation.py      # Validazione post-idratazione
  api/               # FastAPI: app, routes, schemas
  persistence/       # CRUD repository pattern (uno per aggregato)
  db/                # SQLAlchemy models + session + seeding
  output/            # ResultBuilder (CSV, plot, report)
  seed_data/         # JSON di default (solar profiles, ecc.)
frontend/            # Vite + Svelte
tests/               # pytest
```

### 3.2 Naming

- **Snake_case** per Python, **camelCase** per JS/Svelte.
- Classi: `PascalCase`, suffissi descrittivi (`*Model` per ORM,
  `*Repository` per persistence, `*Simulator` per logica MC,
  `*Profile` per profili dati, `*Config`/`*Specs` per dataclass).
- Niente abbreviazioni inventate. `kwp` e `kwh` ok (notazione standard).
- File: minuscolo + underscore, breve ma esplicito.

### 3.3 Lingua

- Codice, identificatori, commenti, docstring: **inglese**.
- Messaggi CLI/UI rivolti all'utente: **italiano** (è l'audience del
  prodotto).
- Documenti `.md` (README, CLAUDE, ROADMAP): **italiano** o inglese
  a seconda di chi è il lettore principale, ma coerenti dentro il
  singolo documento.

### 3.4 Git

- Branch principale `main`, sempre verde (pytest pulito).
- Commit message in inglese, formato breve `area: cosa cambia` o convenzione
  di tipo Conventional Commits. Niente boilerplate, niente emoji.
- Non amendare commit pubblici. Non force-push su `main`.
- **Mai** committare se non esplicitamente richiesto dall'utente.

### 3.5 Dipendenze

- Runtime: solo ciò che è in `requirements.txt`.
- Niente librerie nuove senza giustificazione (è già pesante con
  numpy/pandas/scipy/sqlalchemy/fastapi/matplotlib/tqdm/pydantic).
- Se serve una libreria scientifica, preferisci `scipy` a un pacchetto
  one-shot.

---

## 4. ROADMAP.md è un documento vivente

`ROADMAP.md` **non è un piano scritto una volta e abbandonato**. È un
artefatto di tracking dello stato dell'implementazione. Le sue regole:

- **All'inizio di una fase**: marca la fase come `🚧 in corso` nella sezione
  *Stato*, indicando data di avvio. Se cambi scope rispetto alla descrizione
  originale, aggiorna anche il blocco *Deliverable* della fase per
  riflettere quello che davvero stai facendo.
- **Durante la fase**: se scopri vincoli o opportunità che cambiano il
  design (es. "il modello prezzo deve esporre anche un sample di path
  per il fan chart"), aggiorna il blocco *Deliverable* prima di scrivere
  il codice — così il prossimo che legge non si chiede perché il codice
  fa più di quello che la roadmap promette.
- **A fine fase**: marca `✅ completata` con data e un riepilogo di una
  riga su cosa è stato consegnato (nome dei moduli/classi nuove,
  modifiche di schema DB, numero di test aggiunti, evidenza end-to-end
  raccolta). Sposta i task aperti che sono stati spostati nelle fasi
  successive in modo che restino tracciati.
- **Quando aggiungi una fase nuova**: numerala con il prossimo intero
  libero e descrivi `Problema → Deliverable → Out of scope` come per le
  esistenti. Aggiorna la sezione *Dipendenze fra fasi* se serve.

La verità del progetto vive in `ROADMAP.md`. Se ti accorgi che il codice
è più avanti o più indietro della roadmap, **la roadmap è il bug**:
aggiornala prima di continuare.

## 5. Come aggiungere una feature — workflow standard

1. **Leggi `ROADMAP.md`** e individua la fase di riferimento. Se non
   esiste, aggiungine una (vedi §4) prima di scrivere il codice.
2. **Marca la fase come in corso** nella sezione *Stato* del ROADMAP.
3. **Estendi il modello di dominio** in `simulation/` o `db/models.py`
   (con docstring verbosi, test associati).
4. **Estendi `scenario_builder.py`** se la feature è esposta come parametro
   di scenario.
5. **Estendi gli schema Pydantic** in `api/schemas/` per esporla via API.
6. **Estendi le route** in `api/routes/` se servono nuovi endpoint, oppure
   arricchisci il payload di quelle esistenti.
7. **Estendi il frontend** in `frontend/src/` (form input nel
   ScenarioBuilder, visualizzazione in Dashboard).
8. **Aggiorna `validation.py`** se introduci nuovi vincoli.
9. **Aggiorna i seed JSON** in `seed_data/` se la feature ha valori
   di default per luogo/profilo.
10. **Test**: aggiungi un caso in `tests/`. Tutto verde.
11. **Documenta**: aggiorna `README.md` se cambia l'interfaccia utente,
    aggiorna `CHANGELOG.md` con la voce della modifica.
12. **Chiudi la fase nel ROADMAP**: marca `✅ completata` con un riepilogo
    (vedi §4).

### 5.1 Migrazioni database

Il progetto usa SQLite/PostgreSQL via SQLAlchemy con `create_all()` in
[db/session.py:init_db](sim_stochastic_pv/db/session.py). Non c'è Alembic.

Per aggiungere colonne **nullable** a tabelle esistenti:

- Aggiungi la colonna al model SQLAlchemy.
- In `init_db()`, dopo `create_all`, esegui un PRAGMA check (SQLite) o
  `information_schema` query (Postgres) e un `ALTER TABLE ADD COLUMN`
  manuale se la colonna manca.
- I record già seedati vanno aggiornati: estendi `seed_*` per upsertare
  i campi nuovi anche su righe esistenti.

Per modifiche **breaking** (rinomina, NOT NULL nuovi, vincoli):
chiedi all'utente prima — può essere il momento giusto per introdurre
Alembic.

---

## 6. Cosa non fare

- **Non** introdurre fallback impliciti che mascherano errori
  (es. `try/except: return None`). Lascia esplodere e gestisci all'esterno.
- **Non** rompere la retro-compatibilità degli scenari JSON senza un
  meccanismo di migrazione documentato.
- **Non** scrivere codice senza docstring "per ora, lo aggiungiamo dopo".
  Si scrive il docstring **prima** o **insieme** al corpo.
- **Non** spostare logica nelle route FastAPI. Le route delegano al
  service layer (`application.py`, `persistence/`). Le route sono
  *solo* parsing, validazione, response shaping.
- **Non** salvare path assoluti, segreti, o database personali nel repo.
  `sim_pv.db` è già esplicitamente ignorato → lascialo perdere.
- **Non** introdurre feature flag o backward-compat shim quando puoi
  semplicemente cambiare il codice e aggiornare i call site (questa è
  una codebase con un solo utente, non c'è da gestire utenti legacy).
- **Non** rifare la UI con un altro framework. È Svelte. Punto.

---

## 7. Quando in dubbio

- **Sulla decisione**: chiedi all'utente. Meglio una domanda corta che
  un commit da rifare.
- **Sull'API esistente**: leggi il docstring del metodo. Se non è chiaro,
  il docstring va sistemato — quello è il bug.
- **Su un'assunzione del modello fisico/economico**: vale la regola
  "il codice non spiega, il docstring spiega". Aggiungi nelle Notes.
- **Su una scelta architetturale**: privilegia la chiarezza
  (più classi piccole) rispetto alla brevità (una classe che fa tutto).
