<script>
    /**
     * WeeklyPatternEditor — Phase 5
     *
     * Editor per una matrice (7, 24) di pesi relativi che descrive il pattern
     * di consumo settimanale.  Giorno 0 = Lunedì, giorno 6 = Domenica.
     *
     * Le colonne sono i pesi "raw" (valori positivi, in W o unità arbitrarie
     * — normalizzati internamente dal backend).  Il componente offre:
     *
     *   - Dropdown "preset" per selezionare un pattern predefinito.
     *   - Tab per giorno della settimana (navigazione tra i 7 profili 24h).
     *   - HourlyInput per editare il pattern del giorno selezionato.
     *   - Pulsante "Copia a tutti i giorni" per propagare il pattern corrente.
     *
     * Props:
     *   values  — array 7×24 di float (bind:values), modificato in-place
     *             tramite Svelte reactivity.
     *
     * I preset corrispondono a quelli definiti in
     * `sim_stochastic_pv/simulation/load_profiles/weekly.py`.
     */
    import HourlyInput from "./HourlyInput.svelte";

    export let values = Array.from({ length: 7 }, () => Array(24).fill(100));

    const DAYS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"];

    let selectedDay = 0;

    // -------------------------------------------------------------------
    // Preset patterns — mirror of Python WEEKLY_PRESETS
    // (relative weights, NOT normalised; same structure as the backend)
    // -------------------------------------------------------------------

    const PRESETS = {
        residential_typical: (() => {
            const WD = [ 5, 4, 3, 3, 4,15,40,55,35,25,22,25,28,25,22,22,25,55,70,65,55,45,35,22];
            const WE = [ 5, 4, 3, 3, 4,12,25,40,55,62,65,68,65,60,60,62,62,68,72,68,60,52,42,28];
            return [...Array(5).fill(WD), ...Array(2).fill(WE)];
        })(),
        smart_worker: (() => {
            const WD = [ 5, 4, 3, 3, 4,15,35,55,65,62,58,65,70,62,58,58,62,72,80,72,65,55,42,25];
            const WE = [ 5, 4, 3, 3, 4,12,25,40,50,55,58,60,58,55,55,58,58,62,68,62,55,48,38,22];
            return [...Array(5).fill(WD), ...Array(2).fill(WE)];
        })(),
        commuter: (() => {
            const WD = [ 5, 4, 3, 3, 4,12,30,55,22,15,12,15,18,15,12,12,18,38,60,78,75,60,45,28];
            const WE = [ 5, 4, 3, 3, 4,12,28,42,58,65,70,72,68,65,60,62,62,68,78,72,60,52,40,28];
            return [...Array(5).fill(WD), ...Array(2).fill(WE)];
        })(),
    };

    let selectedPreset = "residential_typical";

    function applyPreset() {
        // Deep-copy the preset rows so mutations don't affect the constant.
        values = PRESETS[selectedPreset].map((row) => [...row]);
    }

    $: currentDayValues = values[selectedDay]?.slice() ?? Array(24).fill(0);

    function updateDayValues(updatedValues) {
        values = values.map((row, i) =>
            i === selectedDay ? [...updatedValues] : [...row]
        );
    }

    function handleHourlyChange(event) {
        updateDayValues(event.detail);
    }

    function copyToAllDays() {
        const template = currentDayValues.slice();
        values = values.map(() => template.slice());
    }
</script>

<div class="card weekly-editor">
    <div class="editor-header">
        <h3>Pattern settimanale (7 × 24 h)</h3>
        <div class="header-actions">
            <select
                class="select select-sm"
                bind:value={selectedPreset}
                on:change={applyPreset}
                title="Applica un preset come punto di partenza"
            >
                <option value="residential_typical">Famiglia tipo (lavoro fuori)</option>
                <option value="smart_worker">Smart worker (lavoro da casa)</option>
                <option value="commuter">Pendolare (rientro serale)</option>
            </select>
            <button
                type="button"
                class="btn btn-outline btn-sm"
                on:click={copyToAllDays}
                title="Propaga il pattern del giorno selezionato agli altri 6"
            >
                Copia a tutti
            </button>
        </div>
    </div>

    <p class="hint">
        I valori sono pesi relativi (in W o unità proporzionali); il backend li
        normalizza colonna per colonna in modo da preservare la media mensile di
        riferimento.
    </p>

    <div class="day-selector">
        {#each DAYS as d, i}
            <button
                type="button"
                class="day-btn"
                class:active={selectedDay === i}
                class:weekend={i >= 5}
                on:click={() => (selectedDay = i)}
            >
                {d}
            </button>
        {/each}
    </div>

    <div class="editor-content">
        <HourlyInput
            label={`${DAYS[selectedDay]} — pattern orario (pesi)`}
            values={currentDayValues}
            on:change={handleHourlyChange}
        />
    </div>
</div>

<style>
    .weekly-editor {
        padding: 1rem;
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border);
    }
    .editor-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .editor-header h3 {
        font-size: 1rem;
        margin: 0;
    }
    .header-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    .select-sm {
        padding: 0.2rem 0.5rem;
        font-size: 0.85rem;
    }
    .hint {
        font-size: 0.8rem;
        color: var(--color-text-secondary);
        margin: 0 0 0.75rem;
    }
    .day-selector {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--color-border);
        padding-bottom: 0.5rem;
    }
    .day-btn {
        background: none;
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        cursor: pointer;
        color: var(--color-text-secondary);
        transition: all 0.15s;
    }
    .day-btn:hover {
        background: var(--color-bg-tertiary);
        color: var(--color-text-primary);
        border-color: var(--color-border-hover);
    }
    .day-btn.active {
        background: var(--color-accent);
        color: white;
        font-weight: 600;
        border-color: var(--color-accent);
    }
    .day-btn.weekend {
        color: var(--color-primary, #0d6efd);
    }
    .day-btn.weekend.active {
        color: white;
    }
</style>
