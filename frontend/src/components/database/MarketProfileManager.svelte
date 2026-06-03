<script>
    /**
     * MarketProfileManager — Database section editor for electricity-market
     * profiles.
     *
     * A market profile encapsulates "how the gas/CO₂/coal market behaves": the
     * generation mix, capacity trends, fuel scenarios, the resulting wholesale
     * price surface, and the dedicated-withdrawal valuation parameters (PMG +
     * optional retail tariff). Scenarios reference a saved profile by id in
     * their "Mercato elettrico" step to value PV export and (optionally) drive
     * the purchase price.
     *
     * The full designer lives here (not just a link to the lab): the same
     * config form (:file:`MarketConfigEditor.svelte`) plus the PMG/retail
     * fields, a live "Anteprima" (annual fan chart + price heatmap via
     * ``POST /api/market/run``), and create / edit / delete over
     * ``/api/market/profiles`` (the save endpoint upserts by name, so editing a
     * profile is a re-save under the same name).
     */
    import { onMount, tick } from "svelte";
    import { api } from "../../api.js";
    import ResultsChart from "../ResultsChart.svelte";
    import Heatmap from "../Heatmap.svelte";
    import MarketConfigEditor from "../MarketConfigEditor.svelte";
    import { MONTHS, HOURS, buildFanConfig } from "../../lib/marketCharts.js";

    let items = [];
    let showForm = false;
    /** ID being edited; null = creating a new profile. */
    let editingId = null;
    /** ID pending delete confirmation. */
    let deleteConfirmId = null;
    /** Forces a fresh MarketConfigEditor mount (defaults) on each open. */
    let editorKey = 0;
    let marketEditor;

    // Form fields owned here (the config sub-form lives in MarketConfigEditor).
    let name = "";
    let description = "";
    let pmg = 0.04;
    let retailEnabled = false;
    let markupPct = 80;
    let fixed = 0.1;

    // Preview state.
    let result = null;
    let previewing = false;
    let previewError = null;

    // Save state.
    let saving = false;
    let formError = null;

    async function load() {
        try {
            items = await api.listMarketProfiles();
        } catch (e) {
            items = [];
        }
    }

    function resetFormFields() {
        name = "";
        description = "";
        pmg = 0.04;
        retailEnabled = false;
        markupPct = 80;
        fixed = 0.1;
        result = null;
        previewError = null;
        formError = null;
    }

    function startCreate() {
        editingId = null;
        deleteConfirmId = null;
        resetFormFields();
        editorKey += 1; // fresh editor → Italian base-mix defaults
        showForm = true;
    }

    async function startEdit(item) {
        deleteConfirmId = null;
        formError = null;
        previewError = null;
        result = null;
        try {
            const p = await api.getMarketProfile(item.id);
            editingId = p.id;
            name = p.name;
            description = p.description || "";
            pmg = p.pmg_base_eur_per_kwh ?? 0.04;
            if (p.retail_markup_fraction != null) {
                retailEnabled = true;
                markupPct = Math.round(p.retail_markup_fraction * 100);
                fixed = p.retail_fixed_components_eur_per_kwh ?? 0.1;
            } else {
                retailEnabled = false;
                markupPct = 80;
                fixed = 0.1;
            }
            editorKey += 1;
            showForm = true;
            await tick(); // wait for the editor to mount before loading config
            marketEditor?.setConfig(p.config || {});
            document
                .getElementById("market-profile-form")
                ?.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (e) {
            formError = e.message;
        }
    }

    function cancelForm() {
        showForm = false;
        editingId = null;
        result = null;
    }

    async function preview() {
        if (!marketEditor || previewing) return;
        previewing = true;
        previewError = null;
        try {
            result = await api.runMarketLab(marketEditor.getConfig());
        } catch (e) {
            previewError = e.message;
            result = null;
        } finally {
            previewing = false;
        }
    }

    async function save() {
        formError = null;
        if (!name.trim()) {
            formError = "Inserisci un nome per il profilo.";
            return;
        }
        if (!marketEditor) return;
        saving = true;
        try {
            const payload = {
                name: name.trim(),
                description: description.trim() || null,
                config: marketEditor.getConfig(),
                pmg_base_eur_per_kwh: Number(pmg),
            };
            if (retailEnabled) {
                payload.retail_markup_fraction = Number(markupPct) / 100;
                payload.retail_fixed_components_eur_per_kwh = Number(fixed);
            }
            await api.saveMarketProfile(payload);
            cancelForm();
            await load();
        } catch (e) {
            formError = e.message;
        } finally {
            saving = false;
        }
    }

    async function handleDelete(id) {
        try {
            await api.deleteMarketProfile(id);
            deleteConfirmId = null;
            if (editingId === id) cancelForm();
            await load();
        } catch (e) {
            formError = e.message;
        }
    }

    onMount(load);

    $: fanConfig = result ? buildFanConfig(result) : null;
</script>

<div class="manager">
    <div class="toolbar">
        <h2>Profili di mercato</h2>
        <button
            class="btn btn-primary"
            on:click={() => (showForm && editingId === null ? cancelForm() : startCreate())}
        >
            {showForm && editingId === null ? "Annulla" : "+ Aggiungi"}
        </button>
    </div>

    <p class="intro">
        Definisci il mercato elettrico sottostante — mix di generazione, trend di
        capacità, scenari gas/CO₂/carbone — e salvalo come profilo riutilizzabile.
        Lo scenario lo aggancia nello step "Mercato elettrico" per valorizzare
        l'immissione (ritiro dedicato) e, opzionalmente, guidare l'acquisto.
    </p>

    {#if showForm}
        <div id="market-profile-form" class="card form-card">
            <h3>{editingId ? "Modifica profilo di mercato" : "Nuovo profilo di mercato"}</h3>

            <div class="form-grid">
                <!-- Config editor + identity/valuation fields -->
                <div class="editor-col">
                    {#key editorKey}
                        <MarketConfigEditor bind:this={marketEditor} on:displaychange={preview} />
                    {/key}

                    <div class="divider"></div>
                    <div class="section-title">Identità e valorizzazione</div>
                    <div class="form-group">
                        <label class="label" for="mp-name">Nome profilo</label>
                        <input id="mp-name" class="input" type="text" bind:value={name} placeholder="Es. Italia crisi gas" />
                    </div>
                    <div class="form-group">
                        <label class="label" for="mp-desc">Descrizione (opzionale)</label>
                        <input id="mp-desc" class="input" type="text" bind:value={description} placeholder="Note libere" />
                    </div>
                    <div class="form-group">
                        <label class="label" for="mp-pmg">PMG — prezzo minimo garantito (€/kWh)</label>
                        <input id="mp-pmg" class="input" type="number" min="0" step="0.005" bind:value={pmg} />
                    </div>
                    <div class="form-group">
                        <label class="checkbox-label">
                            <input type="checkbox" bind:checked={retailEnabled} />
                            Configura tariffa retail (per «il mercato guida l'acquisto»)
                        </label>
                    </div>
                    {#if retailEnabled}
                        <div class="grid-mini">
                            <div class="form-group">
                                <label class="label" for="mp-markup">Markup %</label>
                                <input id="mp-markup" class="input" type="number" min="0" step="5" bind:value={markupPct} />
                            </div>
                            <div class="form-group">
                                <label class="label" for="mp-fixed">Oneri fissi €/kWh</label>
                                <input id="mp-fixed" class="input" type="number" min="0" step="0.01" bind:value={fixed} />
                            </div>
                        </div>
                    {/if}

                    <div class="form-actions">
                        <button type="button" class="btn btn-ghost" on:click={cancelForm}>Annulla</button>
                        <button type="button" class="btn btn-outline" on:click={preview} disabled={previewing}>
                            {previewing ? "Anteprima…" : "Anteprima"}
                        </button>
                        <button type="button" class="btn btn-primary" on:click={save} disabled={saving}>
                            {saving ? "Salvataggio…" : editingId ? "Aggiorna" : "Salva profilo"}
                        </button>
                    </div>
                    {#if formError}<p class="error-msg">{formError}</p>{/if}
                </div>

                <!-- Live preview -->
                <div class="preview-col">
                    {#if result}
                        <div class="section-title">
                            Anteprima — prezzo medio anno {result.display_year}:
                            <strong>{result.mean_price_eur_per_kwh.toFixed(4)} €/kWh</strong>
                        </div>
                        <div class="chart-wrap">
                            <ResultsChart type={fanConfig.type} data={fanConfig.data} options={fanConfig.options} downloadFilename="prezzo_annuale" />
                        </div>
                        <div class="section-title hm-title">Prezzo all'ingrosso (mese × ora) — anno {result.display_year}</div>
                        <Heatmap
                            matrix={result.price_heatmap_eur_per_kwh}
                            rowLabels={MONTHS}
                            colLabels={HOURS}
                            unit="€/kWh"
                            valueDigits={3}
                            colLabelEvery={3}
                        />
                    {:else if previewing}
                        <p class="text-meta">Calcolo dell'anteprima in corso…</p>
                    {:else}
                        <p class="text-meta">
                            Premi <strong>Anteprima</strong> per simulare il mercato e
                            vedere prezzo medio annuale e heatmap mese × ora prima di
                            salvare.
                        </p>
                        {#if previewError}<p class="error-msg">{previewError}</p>{/if}
                    {/if}
                </div>
            </div>
        </div>
    {/if}

    {#if items.length === 0}
        <p class="empty">Nessun profilo di mercato salvato. Aggiungine uno.</p>
    {:else}
        <div class="list">
            {#each items as item (item.id)}
                <div class="card item-card" class:editing={editingId === item.id}>
                    <div class="item-body">
                        <h3>{item.name}</h3>
                        {#if item.description}<p class="meta">{item.description}</p>{/if}
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmId === item.id}
                            <span class="confirm-label">Eliminare?</span>
                            <button class="btn btn-sm btn-danger" on:click={() => handleDelete(item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost" on:click={() => (deleteConfirmId = null)}>No</button>
                        {:else}
                            <button class="btn btn-sm btn-ghost" title="Modifica" on:click={() => startEdit(item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina" on:click={() => { deleteConfirmId = item.id; }}>🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
    .intro { color: var(--color-text-secondary); font-size: 0.9rem; max-width: 75ch; margin-bottom: 1.5rem; }
    .form-card { padding: 1.5rem; margin-bottom: 2rem; }
    .form-grid { display: grid; grid-template-columns: 420px 1fr; gap: 1.5rem; align-items: start; }
    .editor-col { min-width: 0; }
    .preview-col { min-width: 0; }
    .chart-wrap { height: 300px; }
    .hm-title { margin-top: 1rem; }
    .form-actions { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: flex-end; flex-wrap: wrap; }
    .item-card { padding: 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
    .item-card.editing { border-color: var(--color-accent, #0d6efd); outline: 2px solid var(--color-accent, #0d6efd); outline-offset: 2px; }
    .item-body { flex: 1; }
    .item-actions { display: flex; align-items: center; gap: 0.4rem; flex-shrink: 0; }
    .confirm-label { font-size: 0.82rem; color: var(--color-danger, #dc3545); font-weight: 500; }
    .meta { color: var(--color-text-secondary); font-size: 0.9rem; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .error-msg { color: #dc2626; font-size: 0.85rem; margin-top: 0.5rem; }
    @media (max-width: 900px) {
        .form-grid { grid-template-columns: 1fr; }
    }
</style>
