<script>
    import { onMount } from "svelte";
    import { api } from "../api.js";
    import ResultsChart from "../components/ResultsChart.svelte";
    import Heatmap from "../components/Heatmap.svelte";
    import MarketConfigEditor from "../components/MarketConfigEditor.svelte";
    import {
        TECHS,
        colorFor,
        MONTHS,
        HOURS,
        buildFanConfig,
        buildFuelConfig,
        buildDurationConfig,
        buildCapacityConfig,
    } from "../lib/marketCharts.js";

    // Bound reference to the shared config editor (owns the mix/fuel/sim form).
    let marketEditor;

    // ── Run / result state ────────────────────────────────────────────────
    let result = null;
    let running = false;
    let runError = null;
    let exporting = null; // 'xlsx' | 'pdf' | null

    // ── Save-profile state ──────────────────────────────────────────────────
    let saveName = "";
    let savePmg = 0.04;
    let saveRetailEnabled = false; // configure a retail tariff on the profile
    let saveMarkupPct = 80; // % markup on wholesale
    let saveFixed = 0.1; // €/kWh flat retail components (taxes/grid fees)
    let saving = false;
    let saveMsg = "";
    let saveError = null;
    let savedProfiles = [];

    async function run() {
        if (!marketEditor) return;
        running = true;
        runError = null;
        try {
            result = await api.runMarketLab(marketEditor.getConfig());
        } catch (e) {
            runError = e.message;
            result = null;
        } finally {
            running = false;
        }
    }

    async function doExport(kind) {
        if (exporting || !marketEditor) return;
        exporting = kind;
        try {
            const payload = marketEditor.getConfig();
            if (kind === "xlsx") await api.exportMarketXlsx(payload);
            else await api.exportMarketPdf(payload);
        } catch (e) {
            runError = e.message;
        } finally {
            exporting = null;
        }
    }

    async function loadProfiles() {
        try {
            savedProfiles = await api.listMarketProfiles();
        } catch (e) {
            savedProfiles = [];
        }
    }

    async function saveProfile() {
        saveError = null;
        saveMsg = "";
        if (!saveName.trim()) {
            saveError = "Inserisci un nome per il profilo.";
            return;
        }
        saving = true;
        try {
            const payload = {
                name: saveName.trim(),
                description: null,
                config: marketEditor.getConfig(),
                pmg_base_eur_per_kwh: Number(savePmg),
            };
            if (saveRetailEnabled) {
                payload.retail_markup_fraction = Number(saveMarkupPct) / 100;
                payload.retail_fixed_components_eur_per_kwh = Number(saveFixed);
            }
            const r = await api.saveMarketProfile(payload);
            saveMsg = `Profilo "${r.name}" salvato (id ${r.id}).`;
            await loadProfiles();
        } catch (e) {
            saveError = e.message;
        } finally {
            saving = false;
        }
    }

    async function deleteProfile(id) {
        try {
            await api.deleteMarketProfile(id);
            await loadProfiles();
        } catch (e) {
            saveError = e.message;
        }
    }

    // Load a saved profile back into the editor (config + PMG/retail), then
    // recompute. Reads only the keys it recognises so the lighter seeded
    // profile loads too (the rest keeps the current defaults).
    async function loadProfile(id) {
        saveError = null;
        saveMsg = "";
        try {
            const p = await api.getMarketProfile(id);
            marketEditor.setConfig(p.config || {});
            savePmg = p.pmg_base_eur_per_kwh ?? savePmg;
            if (p.retail_markup_fraction != null) {
                saveRetailEnabled = true;
                saveMarkupPct = Math.round(p.retail_markup_fraction * 100);
                saveFixed = p.retail_fixed_components_eur_per_kwh ?? saveFixed;
            } else {
                saveRetailEnabled = false;
            }
            saveName = p.name;
            saveMsg = `Profilo "${p.name}" caricato nell'editor.`;
            await run();
        } catch (e) {
            saveError = e.message;
        }
    }

    onMount(async () => {
        await loadProfiles();
        await run();
    });

    // ── Chart configs ──────────────────────────────────────────────────────
    $: fanConfig = result ? buildFanConfig(result) : null;
    $: fuelConfig = result ? buildFuelConfig(result) : null;
    $: durationConfig = result ? buildDurationConfig(result) : null;
    $: capacityConfig = result ? buildCapacityConfig(result) : null;
    $: setterColors = result ? result.price_setter_techs.map((t, i) => colorFor(t, i)) : [];
    $: shareRows = result
        ? Object.entries(result.price_setter_share_year)
              .sort((a, b) => b[1] - a[1])
              .map(([tech, frac]) => ({ tech, pct: (frac * 100).toFixed(1) }))
        : [];
</script>

<div class="container">
    <h1 class="page-title">Mercato elettrico</h1>
    <p class="page-subtitle text-meta">
        Progetta il mercato elettrico sottostante — mix di generazione, trend di
        capacità, scenari gas/CO₂ — e leggi il prezzo all'ingrosso orario, la sua
        evoluzione, la curva di durata e chi fissa il prezzo. Salva il risultato
        come profilo di mercato da usare nello scenario (ritiro dedicato).
    </p>

    <div class="lab-grid">
        <!-- ── Config card ─────────────────────────────────────────────── -->
        <div class="card config-card">
            <MarketConfigEditor bind:this={marketEditor} on:displaychange={run} />

            <button class="btn btn-primary run-btn" on:click={run} disabled={running}>
                {running ? "Calcolo in corso…" : "Calcola mercato"}
            </button>
            {#if runError}
                <p class="error-msg">{runError}</p>
            {/if}

            <div class="divider"></div>
            <div class="section-title">Salva come profilo</div>
            <div class="form-group">
                <label class="label" for="saveName">Nome profilo</label>
                <input id="saveName" class="input" type="text" bind:value={saveName} placeholder="Es. Italia crisi gas" />
            </div>
            <div class="form-group">
                <label class="label" for="savePmg">PMG (€/kWh)</label>
                <input id="savePmg" class="input" type="number" min="0" step="0.005" bind:value={savePmg} />
            </div>
            <div class="form-group">
                <label class="checkbox-label">
                    <input type="checkbox" bind:checked={saveRetailEnabled} />
                    Configura tariffa retail (per «il mercato guida l'acquisto»)
                </label>
            </div>
            {#if saveRetailEnabled}
                <div class="grid-mini">
                    <div class="form-group">
                        <label class="label" for="saveMarkup">Markup %</label>
                        <input id="saveMarkup" class="input" type="number" min="0" step="5" bind:value={saveMarkupPct} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="saveFixed">Oneri fissi €/kWh</label>
                        <input id="saveFixed" class="input" type="number" min="0" step="0.01" bind:value={saveFixed} />
                    </div>
                </div>
            {/if}
            <button class="btn btn-outline btn-sm" on:click={saveProfile} disabled={saving}>
                {saving ? "Salvataggio…" : "Salva profilo"}
            </button>
            {#if saveMsg}<p class="ok-msg">{saveMsg}</p>{/if}
            {#if saveError}<p class="error-msg">{saveError}</p>{/if}

            {#if savedProfiles.length}
                <p class="hint profile-list-title">Profili salvati — clicca per caricare nell'editor</p>
                <ul class="profile-list">
                    {#each savedProfiles as p}
                        <li>
                            <button
                                class="profile-name-btn"
                                on:click={() => loadProfile(p.id)}
                                title={p.description || "Carica questo profilo nell'editor"}
                            >{p.name}</button>
                            <button class="link-btn" on:click={() => deleteProfile(p.id)} title="Elimina">×</button>
                        </li>
                    {/each}
                </ul>
            {/if}
        </div>

        <!-- ── Results column ──────────────────────────────────────────── -->
        <div class="results-col">
            {#if result}
                <div class="card">
                    <div class="header-actions">
                        <div>
                            <div class="section-title">Sintesi</div>
                            <p class="text-meta">
                                Prezzo medio anno {result.display_year}:
                                <strong>{result.mean_price_eur_per_kwh.toFixed(4)} €/kWh</strong>
                                · {result.n_trajectories} traiettorie · {result.n_runs} run
                            </p>
                        </div>
                        <div class="export-actions">
                            <button class="btn btn-outline btn-sm" on:click={() => doExport("xlsx")} disabled={exporting}>
                                {exporting === "xlsx" ? "…" : "Excel"}
                            </button>
                            <button class="btn btn-outline btn-sm" on:click={() => doExport("pdf")} disabled={exporting}>
                                {exporting === "pdf" ? "…" : "PDF"}
                            </button>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">Prezzo all'ingrosso (mese × ora) — anno {result.display_year}</div>
                    <Heatmap
                        matrix={result.price_heatmap_eur_per_kwh}
                        rowLabels={MONTHS}
                        colLabels={HOURS}
                        unit="€/kWh"
                        valueDigits={3}
                        colLabelEvery={3}
                    />
                </div>

                <div class="card">
                    <div class="section-title">Prezzo medio annuale (banda p05–p95)</div>
                    <div class="chart-wrap">
                        <ResultsChart type={fanConfig.type} data={fanConfig.data} options={fanConfig.options} downloadFilename="prezzo_annuale" />
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">Curva di durata del prezzo — anno {result.display_year}</div>
                    <div class="chart-wrap">
                        <ResultsChart type={durationConfig.type} data={durationConfig.data} options={durationConfig.options} downloadFilename="curva_durata" />
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">Capacità installata per tecnologia</div>
                    <div class="chart-wrap">
                        <ResultsChart type={capacityConfig.type} data={capacityConfig.data} options={capacityConfig.options} downloadFilename="mix_capacita" />
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">Prezzi combustibili — gas e CO₂</div>
                    <p class="text-meta">Livello medio del prezzo (mean-reversion) per anno: gas in €/MWh termici, CO₂ in €/tonnellata. Sono i driver del prezzo all'ingrosso.</p>
                    <div class="chart-wrap">
                        <ResultsChart type={fuelConfig.type} data={fuelConfig.data} options={fuelConfig.options} downloadFilename="prezzi_combustibili" />
                    </div>
                </div>

                <div class="card">
                    <div class="section-title">Chi fissa il prezzo (mese × ora) — anno {result.display_year}</div>
                    {#if result.price_setter_techs.length}
                        <Heatmap
                            mode="categorical"
                            matrix={result.price_setter_dominant}
                            rowLabels={MONTHS}
                            colLabels={HOURS}
                            categories={result.price_setter_techs}
                            categoryColors={setterColors}
                            colLabelEvery={3}
                        />
                        <table class="share-table">
                            <thead><tr><th>Tecnologia</th><th>Quota dell'anno</th></tr></thead>
                            <tbody>
                                {#each shareRows as r}
                                    <tr><td>{r.tech}</td><td>{r.pct}%</td></tr>
                                {/each}
                            </tbody>
                        </table>
                    {:else}
                        <p class="text-meta">Nessun dato disponibile sul prezzo marginale.</p>
                    {/if}
                </div>
            {:else if running}
                <div class="card"><p class="text-meta">Calcolo del mercato in corso…</p></div>
            {:else}
                <div class="card"><p class="text-meta">Configura il mix e premi "Calcola mercato".</p></div>
            {/if}
        </div>
    </div>
</div>

<style>
    .page-subtitle {
        max-width: 70ch;
        margin-bottom: 1.5rem;
    }
    .lab-grid {
        display: grid;
        grid-template-columns: 420px 1fr;
        gap: 1.5rem;
        align-items: start;
    }
    /* min-width:0 lets each grid item shrink to its track and clip its own
       overflow, so the config card never bleeds over the results column. */
    .config-card {
        position: sticky;
        top: 1rem;
        min-width: 0;
        overflow: hidden;
    }
    .results-col {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        min-width: 0;
    }
    .chart-wrap {
        height: 320px;
    }
    .hint {
        font-size: 0.75rem;
        color: var(--color-text-secondary, #6b7280);
        line-height: 1.4;
        margin: 0.5rem 0 0 0;
    }
    .profile-list-title {
        margin-top: 0.75rem;
        font-weight: 600;
    }
    .profile-name-btn {
        background: none;
        border: none;
        padding: 0;
        color: var(--color-accent, #3b82f6);
        cursor: pointer;
        font-size: 0.85rem;
        text-align: left;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .profile-name-btn:hover {
        text-decoration: underline;
    }
    .run-btn {
        width: 100%;
        margin-top: 1rem;
    }
    .header-actions {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .export-actions {
        display: flex;
        gap: 0.5rem;
    }
    .error-msg {
        color: #dc2626;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .ok-msg {
        color: #16a34a;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .profile-list {
        list-style: none;
        padding: 0;
        margin: 0.75rem 0 0 0;
        font-size: 0.85rem;
    }
    .profile-list li {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 0;
        border-bottom: 1px solid var(--color-border, #e5e7eb);
    }
    .link-btn {
        background: none;
        border: none;
        color: #dc2626;
        cursor: pointer;
        font-size: 1rem;
        line-height: 1;
    }
    .share-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.75rem;
        font-size: 0.85rem;
    }
    .share-table th, .share-table td {
        text-align: left;
        padding: 3px 6px;
        border-bottom: 1px solid var(--color-border, #e5e7eb);
    }
    @media (max-width: 900px) {
        .lab-grid {
            grid-template-columns: 1fr;
        }
        .config-card {
            position: static;
        }
    }
</style>
