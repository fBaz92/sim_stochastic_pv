<script>
    /**
     * LocationsManager — Database tab for installation sites.
     *
     * A location is the durable anchor of a site's datasets: the PVGIS
     * solar profile and the calibrated stochastic climate profile hang
     * off it via `location_id`. This component shows one card per site
     * with the status of each linked profile (present with freshness
     * date, or missing with a download action), and an add-flow that
     * saves address + profiles with a single transactional call
     * (`POST /api/locations/import` — per-component errors are explicit).
     *
     * Profiles created before locations existed (no `location_id`) are
     * listed in a separate "non collegati" section so nothing becomes
     * invisible after the migration.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";
    import LeafletMap from "../LeafletMap.svelte";
    import LocationSearch from "../LocationSearch.svelte";
    import ClimateNormalsPreview from "../ClimateNormalsPreview.svelte";

    let locations = [];
    let orphanSolar = [];
    let orphanClimate = [];
    let loading = false;
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
    let importWarnings = [];
    let climateNormals = null;
    let climateLoading = false;
    let climateError = null;
    let climateDebounceTimer = null;

    // Per-card busy flag for re-download actions, keyed by location id.
    let refreshingId = null;

    // Rename / delete state — keyed by `${kind}-${id}` so location, solar
    // and climate rows stay independent even when their IDs collide.
    let renamingKey = null;
    let renameDraft = "";
    let deleteConfirmKey = null;
    let deleteProfilesToo = false;

    async function loadAll() {
        loading = true;
        message = "";
        try {
            const [locs, sp, cp] = await Promise.all([
                api.listLocations(),
                api.listSolarProfiles(),
                api.listClimateProfiles(),
            ]);
            locations = locs;
            orphanSolar = sp.filter((p) => p.location_id == null);
            orphanClimate = cp.filter((p) => p.location_id == null);
        } catch (e) {
            console.error(e);
            message = "Errore nel caricamento delle posizioni: " + e.message;
        } finally {
            loading = false;
        }
    }

    function rowKey(kind, id) {
        return `${kind}-${id}`;
    }

    function formatDate(iso) {
        if (!iso) return "";
        try {
            return new Date(iso).toLocaleDateString("it-IT");
        } catch {
            return iso;
        }
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

    async function commitRename(kind, item) {
        const name = renameDraft.trim();
        if (!name || name === item.name) return cancelRename();
        try {
            if (kind === "location") await api.updateLocation(item.id, { name });
            else if (kind === "solar") await api.updateSolarProfile(item.id, { name });
            else await api.updateClimateProfile(item.id, { name });
            await loadAll();
            cancelRename();
        } catch (e) {
            alert("Errore nella rinomina: " + e.message);
        }
    }

    async function deleteLocation(loc) {
        try {
            await api.deleteLocation(loc.id, { deleteProfiles: deleteProfilesToo });
            deleteConfirmKey = null;
            deleteProfilesToo = false;
            await loadAll();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    async function deleteOrphan(kind, id) {
        try {
            if (kind === "solar") await api.deleteSolarProfile(id);
            else await api.deleteClimateProfile(id);
            deleteConfirmKey = null;
            await loadAll();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    /**
     * (Re)download one component for an existing location, reusing its
     * coordinates and — when present — the tilt/azimuth of the linked
     * solar profile. Errors come back in the response body, explicitly.
     */
    async function refreshComponent(loc, { solar, climate }) {
        refreshingId = loc.id;
        message = "";
        try {
            const solarProfile = loc.solar_profiles[0];
            const res = await api.importLocation({
                name: loc.name,
                display_name: loc.display_name || undefined,
                latitude: loc.latitude,
                longitude: loc.longitude,
                include_solar: solar,
                include_climate: climate,
                tilt_degrees: solarProfile?.optimal_tilt_degrees ?? importTilt,
                azimuth_degrees: solarProfile?.optimal_azimuth_degrees ?? importAzimuth,
                loss_pct: importLossPct,
                lookback_years: importLookbackYears,
            });
            const errors = [res.solar_error, res.climate_error].filter(Boolean);
            if (errors.length) message = errors.join(" — ");
            await loadAll();
        } catch (e) {
            message = "Errore nell'aggiornamento: " + e.message;
        } finally {
            refreshingId = null;
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

    async function importNewLocation() {
        const effectiveName = (importName && importName.trim())
            || defaultImportNameFromCoords();
        importName = effectiveName;
        importError = null;
        importWarnings = [];
        importing = true;
        try {
            const res = await api.importLocation({
                name: effectiveName,
                display_name: pickedDisplayName || undefined,
                latitude: pickedLat,
                longitude: pickedLon,
                include_solar: true,
                include_climate: alsoCalibrateThermal,
                tilt_degrees: importTilt,
                azimuth_degrees: importAzimuth,
                loss_pct: importLossPct,
                lookback_years: importLookbackYears,
            });
            const warnings = [res.solar_error, res.climate_error].filter(Boolean);
            await loadAll();
            if (warnings.length) {
                // Site saved, but one component failed: keep the form open
                // so the user sees what happened and can retry.
                importWarnings = warnings;
            } else {
                resetAddForm();
            }
        } catch (err) {
            importError = err.message || "Errore durante l'import.";
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
        importWarnings = [];
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
        Ogni posizione salva l'indirizzo e ancora i suoi dati: il profilo
        solare (PVGIS) e il profilo climatico stocastico (Open-Meteo).
        Indirizzo e profili sono salvati con un'unica operazione; gli errori
        di download sono sempre riportati componente per componente.
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
            {#if importWarnings.length}
                <div class="warning">
                    <strong>Posizione salvata</strong>, ma con problemi:
                    <ul>
                        {#each importWarnings as w}
                            <li>{w}</li>
                        {/each}
                    </ul>
                    Puoi riprovare il download dalla scheda della posizione.
                </div>
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
                    on:click={importNewLocation}
                >
                    {importing ? "Importazione in corso…" : "Salva e scarica dati"}
                </button>
            </div>
        </div>
    {/if}

    <!-- Locations list ------------------------------------------------------ -->
    {#if loading}
        <p>Caricamento…</p>
    {:else if locations.length === 0}
        <p class="empty">Nessuna posizione salvata.</p>
    {:else}
        <div class="list">
            {#each locations as loc (loc.id)}
                <div class="card item-card location-card">
                    <div class="item-body">
                        {#if renamingKey === rowKey('location', loc.id)}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRename('location', loc)}
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
                            <h4>{loc.name}</h4>
                        {/if}
                        <p class="meta">
                            {#if loc.display_name}{loc.display_name} · {/if}
                            {loc.latitude.toFixed(3)}°, {loc.longitude.toFixed(3)}°
                            {#if loc.elevation_m != null} · {Math.round(loc.elevation_m)} m s.l.m.{/if}
                        </p>
                        <div class="profile-badges">
                            {#if loc.solar_profiles.length > 0}
                                <span class="badge ok" title={loc.solar_profiles[0].source}>
                                    ☀ Solare · agg. {formatDate(loc.solar_profiles[0].updated_at)}
                                    · tilt {loc.solar_profiles[0].optimal_tilt_degrees}°
                                </span>
                                <button
                                    class="btn btn-xs btn-ghost"
                                    disabled={refreshingId === loc.id}
                                    on:click={() => refreshComponent(loc, { solar: true, climate: false })}
                                >↻</button>
                            {:else}
                                <span class="badge missing">☀ Solare mancante</span>
                                <button
                                    class="btn btn-xs btn-primary"
                                    disabled={refreshingId === loc.id}
                                    on:click={() => refreshComponent(loc, { solar: true, climate: false })}
                                >Scarica</button>
                            {/if}
                            {#if loc.climate_profiles.length > 0}
                                <span class="badge ok" title={loc.climate_profiles[0].source}>
                                    🌡 Clima
                                    {#if loc.climate_profiles[0].lookback_window}
                                        · {loc.climate_profiles[0].lookback_window.start_year}–{loc.climate_profiles[0].lookback_window.end_year}
                                    {/if}
                                    · agg. {formatDate(loc.climate_profiles[0].updated_at)}
                                </span>
                                <button
                                    class="btn btn-xs btn-ghost"
                                    disabled={refreshingId === loc.id}
                                    on:click={() => refreshComponent(loc, { solar: false, climate: true })}
                                >↻</button>
                            {:else}
                                <span class="badge missing">🌡 Clima mancante</span>
                                <button
                                    class="btn btn-xs btn-primary"
                                    disabled={refreshingId === loc.id}
                                    on:click={() => refreshComponent(loc, { solar: false, climate: true })}
                                >Calibra</button>
                            {/if}
                            {#if refreshingId === loc.id}
                                <span class="badge">aggiornamento…</span>
                            {/if}
                        </div>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmKey === rowKey('location', loc.id)}
                            <div class="delete-confirm">
                                <span class="confirm-label">Eliminare la posizione?</span>
                                <label class="checkbox-label small">
                                    <input type="checkbox" bind:checked={deleteProfilesToo} />
                                    elimina anche i profili collegati
                                </label>
                                <div>
                                    <button
                                        class="btn btn-sm btn-danger"
                                        on:click={() => deleteLocation(loc)}
                                    >Sì</button>
                                    <button
                                        class="btn btn-sm btn-ghost"
                                        on:click={() => { deleteConfirmKey = null; deleteProfilesToo = false; }}
                                    >No</button>
                                </div>
                            </div>
                        {:else if renamingKey !== rowKey('location', loc.id)}
                            <button
                                class="btn btn-sm btn-ghost"
                                title="Rinomina"
                                on:click={() => startRename('location', loc)}
                            >✏️</button>
                            <button
                                class="btn btn-sm btn-ghost btn-del"
                                title="Elimina"
                                on:click={() => (deleteConfirmKey = rowKey('location', loc.id))}
                            >🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}

    <!-- Orphan profiles (created before locations existed) ------------------- -->
    {#if orphanSolar.length > 0 || orphanClimate.length > 0}
        <h3 class="section-title">Profili non collegati a una posizione</h3>
        <p class="hint">
            Creati prima dell'introduzione delle posizioni, o scollegati da
            una posizione eliminata. Restano usabili da scenari e design.
        </p>
        <div class="list">
            {#each orphanSolar as item (rowKey('solar', item.id))}
                <div class="card item-card">
                    <div class="item-body">
                        {#if renamingKey === rowKey('solar', item.id)}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRename('solar', item)}
                            >
                                <input class="input" bind:value={renameDraft} autofocus required />
                                <button type="submit" class="btn btn-sm btn-primary">Salva</button>
                                <button type="button" class="btn btn-sm btn-ghost" on:click={cancelRename}>Annulla</button>
                            </form>
                        {:else}
                            <h4>☀ {item.name}</h4>
                        {/if}
                        <p class="meta">
                            {item.location_name}
                            · {item.latitude.toFixed(3)}°, {item.longitude.toFixed(3)}°
                            · tilt {item.optimal_tilt_degrees}°
                            {#if item.source} · {item.source}{/if}
                        </p>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmKey === rowKey('solar', item.id)}
                            <span class="confirm-label">Eliminare?</span>
                            <button class="btn btn-sm btn-danger" on:click={() => deleteOrphan('solar', item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost" on:click={() => (deleteConfirmKey = null)}>No</button>
                        {:else if renamingKey !== rowKey('solar', item.id)}
                            <button class="btn btn-sm btn-ghost" title="Rinomina" on:click={() => startRename('solar', item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina" on:click={() => (deleteConfirmKey = rowKey('solar', item.id))}>🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
            {#each orphanClimate as item (rowKey('climate', item.id))}
                <div class="card item-card">
                    <div class="item-body">
                        {#if renamingKey === rowKey('climate', item.id)}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRename('climate', item)}
                            >
                                <input class="input" bind:value={renameDraft} autofocus required />
                                <button type="submit" class="btn btn-sm btn-primary">Salva</button>
                                <button type="button" class="btn btn-sm btn-ghost" on:click={cancelRename}>Annulla</button>
                            </form>
                        {:else}
                            <h4>🌡 {item.name}</h4>
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
                            <button class="btn btn-sm btn-danger" on:click={() => deleteOrphan('climate', item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost" on:click={() => (deleteConfirmKey = null)}>No</button>
                        {:else if renamingKey !== rowKey('climate', item.id)}
                            <button class="btn btn-sm btn-ghost" title="Rinomina" on:click={() => startRename('climate', item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina" on:click={() => (deleteConfirmKey = rowKey('climate', item.id))}>🗑️</button>
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
    .checkbox-label.small { font-size: 0.82rem; }
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
        margin-bottom: 0.5rem;
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
    .profile-badges {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.4rem;
        margin-top: 0.5rem;
    }
    .badge {
        font-size: 0.78rem;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background: var(--color-bg-tertiary, #f0f0f0);
        color: var(--color-text-secondary);
        border: 1px solid var(--color-border, #ddd);
    }
    .badge.ok {
        border-color: var(--color-success, #28a745);
        color: var(--color-success, #28a745);
        background: transparent;
    }
    .badge.missing {
        border-color: var(--color-warning, #ffc107);
        color: var(--color-warning-text, #b8860b);
        background: transparent;
    }
    .btn-xs {
        font-size: 0.75rem;
        padding: 0.1rem 0.45rem;
        line-height: 1.4;
    }
    .item-actions { display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0; }
    .delete-confirm {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.35rem;
    }
    .rename-form { display: flex; gap: 0.4rem; align-items: center; }
    .rename-form .input { min-width: 16rem; }
    .confirm-label {
        font-size: 0.82rem;
        color: var(--color-danger, #dc3545);
        font-weight: 500;
    }
    .warning {
        margin-top: 1rem;
        padding: 0.75rem 1rem;
        border: 1px solid var(--color-warning, #ffc107);
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .warning ul { margin: 0.4rem 0; padding-left: 1.2rem; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .error { color: var(--color-danger, #dc3545); }
</style>
