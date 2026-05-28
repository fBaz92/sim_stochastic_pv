<script>
    /**
     * LocationsManager — Database tab for geographic profiles.
     *
     * A "location" in this codebase is materialised as two paired records:
     *
     *   - **SolarProfileModel** — monthly PVGIS production + Open-Meteo
     *     ``p_sunny`` for the Markov sunny/cloudy chain.
     *   - **ClimateProfileModel** — calibrated stochastic thermal model
     *     (Phase 15) for temperature-dependent derating / HVAC load.
     *
     * This component lets the user manage both lists from the Database
     * section: rename, delete, and add a brand-new location via the same
     * geocoding + map + PVGIS import sub-flow used in the Scenario
     * wizard's "Luogo" step.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";
    import LeafletMap from "../LeafletMap.svelte";
    import LocationSearch from "../LocationSearch.svelte";
    import ClimateNormalsPreview from "../ClimateNormalsPreview.svelte";

    let solarProfiles = [];
    let climateProfiles = [];
    let loadingSolar = false;
    let loadingClimate = false;
    let message = "";

    // Add-new flow state
    let showAdd = false;
    let pickedLat = 44.336;
    let pickedLon = 10.831;
    let pickedDisplayName = "";
    let importTilt = 35.0;
    let importAzimuth = 180.0;
    let importLossPct = 14.0;
    let importLookbackYears = 10;
    let importName = "";
    let alsoCalibrateThermal = true;
    let importing = false;
    let importError = null;
    let climateNormals = null;
    let climateLoading = false;
    let climateError = null;
    let climateDebounceTimer = null;

    // Rename / delete state — keyed by `${kind}-${id}` to keep solar and
    // climate rows independent even when their IDs collide.
    let renamingKey = null;
    let renameDraft = "";
    let deleteConfirmKey = null;

    async function loadAll() {
        loadingSolar = true;
        loadingClimate = true;
        message = "";
        try {
            const [sp, cp] = await Promise.all([
                api.listSolarProfiles(),
                api.listClimateProfiles(),
            ]);
            solarProfiles = sp;
            climateProfiles = cp;
        } catch (e) {
            console.error(e);
            message = "Errore nel caricamento delle posizioni: " + e.message;
        } finally {
            loadingSolar = false;
            loadingClimate = false;
        }
    }

    function rowKey(kind, id) {
        return `${kind}-${id}`;
    }

    function startRename(kind, item) {
        renamingKey = rowKey(kind, item.id);
        renameDraft = item.name;
        deleteConfirmKey = null;
    }

    function cancelRename() {
        renamingKey = null;
        renameDraft = "";
    }

    async function commitRenameSolar(item) {
        const name = renameDraft.trim();
        if (!name || name === item.name) return cancelRename();
        try {
            await api.updateSolarProfile(item.id, { name });
            await loadAll();
            cancelRename();
        } catch (e) {
            alert("Errore nella rinomina: " + e.message);
        }
    }

    async function commitRenameClimate(item) {
        const name = renameDraft.trim();
        if (!name || name === item.name) return cancelRename();
        try {
            await api.updateClimateProfile(item.id, { name });
            await loadAll();
            cancelRename();
        } catch (e) {
            alert("Errore nella rinomina: " + e.message);
        }
    }

    async function deleteSolar(id) {
        try {
            await api.deleteSolarProfile(id);
            deleteConfirmKey = null;
            await loadAll();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    async function deleteClimate(id) {
        try {
            await api.deleteClimateProfile(id);
            deleteConfirmKey = null;
            await loadAll();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    // ── Add-new flow ──────────────────────────────────────────────────
    function onLocationPicked(event) {
        const r = event.detail;
        pickedLat = r.latitude;
        pickedLon = r.longitude;
        pickedDisplayName = r.display_name;
        const shortName = (r.display_name.split(",")[0] || "Luogo").trim();
        if (!importName) importName = shortName;
        scheduleClimateFetch();
    }

    function onMapChange(event) {
        pickedLat = event.detail.lat;
        pickedLon = event.detail.lon;
        scheduleClimateFetch();
    }

    function scheduleClimateFetch() {
        clearTimeout(climateDebounceTimer);
        climateDebounceTimer = setTimeout(fetchClimateNormals, 600);
    }

    async function fetchClimateNormals() {
        climateLoading = true;
        climateError = null;
        try {
            climateNormals = await api.getClimateNormals(pickedLat, pickedLon, {
                lookback_years: importLookbackYears,
            });
        } catch (err) {
            climateError = err.message || "Errore di rete";
            climateNormals = null;
        } finally {
            climateLoading = false;
        }
    }

    function defaultImportNameFromCoords() {
        const lat = pickedLat.toFixed(2);
        const lon = pickedLon.toFixed(2);
        return `Pos_${lat}_${lon}`.replace(/-/g, "S").replace(/\./g, "_");
    }

    async function importProfileFromLocation() {
        const effectiveName = (importName && importName.trim())
            || defaultImportNameFromCoords();
        importName = effectiveName;
        importError = null;
        importing = true;
        const locationLabel = pickedDisplayName
            || `${pickedLat.toFixed(3)}°, ${pickedLon.toFixed(3)}°`;
        try {
            await api.createSolarProfileFromLocation({
                name: effectiveName,
                location_name: locationLabel,
                latitude: pickedLat,
                longitude: pickedLon,
                tilt_degrees: importTilt,
                azimuth_degrees: importAzimuth,
                loss_pct: importLossPct,
                lookback_years: importLookbackYears,
                overwrite: false,
            });
            if (alsoCalibrateThermal) {
                await api.createClimateProfileFromLocation({
                    name: effectiveName,
                    location_name: locationLabel,
                    latitude: pickedLat,
                    longitude: pickedLon,
                    lookback_years: importLookbackYears,
                    climate_trend_c_per_year: 0.0,
                    overwrite: true,
                });
            }
            await loadAll();
            resetAddForm();
        } catch (err) {
            importError = err.message || "Errore durante l'import PVGIS.";
        } finally {
            importing = false;
        }
    }

    function resetAddForm() {
        showAdd = false;
        importName = "";
        pickedDisplayName = "";
        climateNormals = null;
        climateError = null;
        importError = null;
    }

    onMount(loadAll);
</script>

<div>
    <div class="header-actions">
        <h2>Posizioni geografiche</h2>
        <button class="btn btn-primary" on:click={() => {
            if (showAdd) {
                resetAddForm();
            } else {
                showAdd = true;
                scheduleClimateFetch();
            }
        }}>
            {showAdd ? "Annulla" : "+ Aggiungi posizione"}
        </button>
    </div>
    <p class="hint">
        Ogni posizione genera un profilo solare (PVGIS) e, in opzione, un
        profilo climatico stocastico (Open-Meteo). I profili sono riusabili
        da qualsiasi scenario o design.
    </p>

    {#if message}
        <p class="error">{message}</p>
    {/if}

    {#if showAdd}
        <div class="card add-card">
            <h3>Nuova posizione</h3>
            <div class="add-grid">
                <div>
                    <LocationSearch on:select={onLocationPicked} />
                    <div class="form-group">
                        <label class="label" for="loc-name">Nome breve *</label>
                        <input
                            id="loc-name"
                            class="input"
                            bind:value={importName}
                            placeholder="es. Pavullo"
                        />
                        <p class="hint small">
                            Lasciato vuoto verrà generato automaticamente da
                            lat/lon (es. <code>Pos_44_34_10_83</code>).
                        </p>
                    </div>
                    <div class="grid-2col">
                        <div class="form-group">
                            <label class="label" for="loc-tilt">Tilt pannelli (°)</label>
                            <input
                                id="loc-tilt"
                                class="input"
                                type="number"
                                step="1"
                                min="0"
                                max="90"
                                bind:value={importTilt}
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="loc-azimuth">Azimuth (°, 180 = sud)</label>
                            <input
                                id="loc-azimuth"
                                class="input"
                                type="number"
                                step="1"
                                min="0"
                                max="360"
                                bind:value={importAzimuth}
                            />
                        </div>
                    </div>
                    <div class="grid-2col">
                        <div class="form-group">
                            <label class="label" for="loc-loss">Perdite di sistema (%)</label>
                            <input
                                id="loc-loss"
                                class="input"
                                type="number"
                                step="0.5"
                                min="0"
                                bind:value={importLossPct}
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="loc-lookback">Anni di storico</label>
                            <input
                                id="loc-lookback"
                                class="input"
                                type="number"
                                step="1"
                                min="1"
                                max="30"
                                bind:value={importLookbackYears}
                            />
                        </div>
                    </div>
                    <label class="checkbox-label">
                        <input type="checkbox" bind:checked={alsoCalibrateThermal} />
                        Calibra anche il profilo climatico stocastico (consigliato)
                    </label>
                </div>
                <div>
                    <LeafletMap
                        bind:lat={pickedLat}
                        bind:lon={pickedLon}
                        on:change={onMapChange}
                    />
                    <p class="coords">
                        {pickedLat.toFixed(4)}°, {pickedLon.toFixed(4)}°
                        {#if pickedDisplayName} · {pickedDisplayName}{/if}
                    </p>
                    <ClimateNormalsPreview
                        data={climateNormals}
                        loading={climateLoading}
                        error={climateError}
                    />
                </div>
            </div>

            {#if importError}
                <p class="error">{importError}</p>
            {/if}

            <div class="form-actions">
                <button
                    type="button"
                    class="btn btn-ghost"
                    on:click={resetAddForm}
                >Annulla</button>
                <button
                    type="button"
                    class="btn btn-primary"
                    disabled={importing}
                    on:click={importProfileFromLocation}
                >
                    {importing ? "Importazione in corso…" : "Importa profilo"}
                </button>
            </div>
        </div>
    {/if}

    <!-- Solar profiles list ------------------------------------------------ -->
    <h3 class="section-title">Profili solari (PVGIS)</h3>
    {#if loadingSolar}
        <p>Caricamento…</p>
    {:else if solarProfiles.length === 0}
        <p class="empty">Nessun profilo solare salvato.</p>
    {:else}
        <div class="list">
            {#each solarProfiles as item (item.id)}
                <div class="card item-card">
                    <div class="item-body">
                        {#if renamingKey === rowKey('solar', item.id)}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRenameSolar(item)}
                            >
                                <input
                                    class="input"
                                    bind:value={renameDraft}
                                    autofocus
                                    required
                                />
                                <button type="submit" class="btn btn-sm btn-primary">Salva</button>
                                <button
                                    type="button"
                                    class="btn btn-sm btn-ghost"
                                    on:click={cancelRename}
                                >Annulla</button>
                            </form>
                        {:else}
                            <h4>{item.name}</h4>
                        {/if}
                        <p class="meta">
                            {item.location_name}
                            · {item.latitude.toFixed(3)}°, {item.longitude.toFixed(3)}°
                            · tilt {item.optimal_tilt_degrees}°
                            · azimuth {item.optimal_azimuth_degrees}°
                            {#if item.source} · {item.source}{/if}
                        </p>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmKey === rowKey('solar', item.id)}
                            <span class="confirm-label">Eliminare?</span>
                            <button
                                class="btn btn-sm btn-danger"
                                on:click={() => deleteSolar(item.id)}
                            >Sì</button>
                            <button
                                class="btn btn-sm btn-ghost"
                                on:click={() => (deleteConfirmKey = null)}
                            >No</button>
                        {:else if renamingKey !== rowKey('solar', item.id)}
                            <button
                                class="btn btn-sm btn-ghost"
                                title="Rinomina"
                                on:click={() => startRename('solar', item)}
                            >✏️</button>
                            <button
                                class="btn btn-sm btn-ghost btn-del"
                                title="Elimina"
                                on:click={() => (deleteConfirmKey = rowKey('solar', item.id))}
                            >🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}

    <!-- Climate profiles list ---------------------------------------------- -->
    <h3 class="section-title">Profili climatici (termici)</h3>
    {#if loadingClimate}
        <p>Caricamento…</p>
    {:else if climateProfiles.length === 0}
        <p class="empty">Nessun profilo climatico salvato.</p>
    {:else}
        <div class="list">
            {#each climateProfiles as item (item.id)}
                <div class="card item-card">
                    <div class="item-body">
                        {#if renamingKey === rowKey('climate', item.id)}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRenameClimate(item)}
                            >
                                <input
                                    class="input"
                                    bind:value={renameDraft}
                                    autofocus
                                    required
                                />
                                <button type="submit" class="btn btn-sm btn-primary">Salva</button>
                                <button
                                    type="button"
                                    class="btn btn-sm btn-ghost"
                                    on:click={cancelRename}
                                >Annulla</button>
                            </form>
                        {:else}
                            <h4>{item.name}</h4>
                        {/if}
                        <p class="meta">
                            {item.location_name}
                            · {item.latitude.toFixed(3)}°, {item.longitude.toFixed(3)}°
                            {#if item.lookback_window}
                                · {item.lookback_window.start_year}–{item.lookback_window.end_year}
                            {/if}
                            {#if item.source} · {item.source}{/if}
                        </p>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmKey === rowKey('climate', item.id)}
                            <span class="confirm-label">Eliminare?</span>
                            <button
                                class="btn btn-sm btn-danger"
                                on:click={() => deleteClimate(item.id)}
                            >Sì</button>
                            <button
                                class="btn btn-sm btn-ghost"
                                on:click={() => (deleteConfirmKey = null)}
                            >No</button>
                        {:else if renamingKey !== rowKey('climate', item.id)}
                            <button
                                class="btn btn-sm btn-ghost"
                                title="Rinomina"
                                on:click={() => startRename('climate', item)}
                            >✏️</button>
                            <button
                                class="btn btn-sm btn-ghost btn-del"
                                title="Elimina"
                                on:click={() => (deleteConfirmKey = rowKey('climate', item.id))}
                            >🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .header-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .hint {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
        margin: 0 0 1.5rem;
        max-width: 65ch;
    }
    .hint.small { font-size: 0.82rem; margin-top: 0.25rem; }
    .add-card {
        padding: 1.5rem;
        margin-bottom: 2rem;
        background-color: var(--color-bg-tertiary);
    }
    .add-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
    }
    @media (max-width: 900px) {
        .add-grid { grid-template-columns: 1fr; }
    }
    .grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .checkbox-label { display: flex; align-items: center; gap: 0.5rem; }
    .coords {
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: var(--color-text-secondary);
    }
    .form-actions {
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        justify-content: flex-end;
    }
    .section-title {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .list { display: flex; flex-direction: column; gap: 0.75rem; }
    .item-card {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem;
        height: auto;
    }
    .item-body { flex: 1; min-width: 0; }
    .item-body h4 { margin: 0; }
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.85rem;
        margin: 0.25rem 0 0;
    }
    .item-actions { display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0; }
    .rename-form { display: flex; gap: 0.4rem; align-items: center; }
    .rename-form .input { min-width: 16rem; }
    .confirm-label {
        font-size: 0.82rem;
        color: var(--color-danger, #dc3545);
        font-weight: 500;
    }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .error { color: var(--color-danger, #dc3545); }
</style>
