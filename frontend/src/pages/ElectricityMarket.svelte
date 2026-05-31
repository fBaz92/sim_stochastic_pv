<script>
    import { onMount } from "svelte";
    import { api } from "../api.js";
    import ResultsChart from "../components/ResultsChart.svelte";
    import Heatmap from "../components/Heatmap.svelte";

    // ── Technology display config (label + colour) ────────────────────────
    const TECHS = [
        { key: "gas", label: "Gas", color: "#ef4444" },
        { key: "coal", label: "Carbone", color: "#6b7280" },
        { key: "nuclear", label: "Nucleare", color: "#8b5cf6" },
        { key: "wind", label: "Eolico", color: "#10b981" },
        { key: "solar", label: "Solare", color: "#f59e0b" },
        { key: "hydro_mustrun", label: "Idro (must-run)", color: "#3b82f6" },
    ];
    const TECH_COLOR = {
        gas: "#ef4444", coal: "#6b7280", nuclear: "#8b5cf6",
        wind: "#10b981", solar: "#f59e0b", hydro_mustrun: "#3b82f6",
        import: "#ec4899",
    };
    const FALLBACK = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#6b7280", "#ec4899"];
    const colorFor = (tech, i) => TECH_COLOR[tech] ?? FALLBACK[i % FALLBACK.length];

    const MONTHS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"];
    const HOURS = Array.from({ length: 24 }, (_, h) => String(h));

    // ── Form state (defaults = Italian base mix) ──────────────────────────
    let techState = {
        gas: { cap: 45, growth: 0, stepYear: "", stepCap: "" },
        coal: { cap: 0, growth: 0, stepYear: "", stepCap: "" },
        nuclear: { cap: 0, growth: 0, stepYear: "", stepCap: "" },
        wind: { cap: 13, growth: 4, stepYear: "", stepCap: "" },
        solar: { cap: 30, growth: 6, stepYear: "", stepCap: "" },
        hydro_mustrun: { cap: 8, growth: 0, stepYear: "", stepCap: "" },
    };
    let gasScenario = "base";
    let co2Scenario = "base";
    let coalScenario = "";
    let gasDrift = 0.0;
    let co2Drift = 0.0;
    let nYears = 20;
    let nTrajectories = 8;
    let nRuns = 6;
    let displayYear = 0;
    let seed = 42;

    // ── Run / result state ────────────────────────────────────────────────
    let result = null;
    let running = false;
    let runError = null;
    let exporting = null; // 'xlsx' | 'pdf' | null

    // ── Save-profile state ──────────────────────────────────────────────────
    let saveName = "";
    let savePmg = 0.04;
    let saving = false;
    let saveMsg = "";
    let saveError = null;
    let savedProfiles = [];

    function buildPayload() {
        const capacities_gw = {};
        const capacity_trends = {};
        for (const t of TECHS) {
            const s = techState[t.key];
            capacities_gw[t.key] = Number(s.cap);
            const trend = { annual_growth_pct: Number(s.growth) };
            if (s.stepYear !== "" && s.stepCap !== "") {
                trend.step_year = Number(s.stepYear);
                trend.step_capacity_gw = Number(s.stepCap);
            }
            capacity_trends[t.key] = trend;
        }
        return {
            capacities_gw,
            capacity_trends,
            gas_scenario: gasScenario,
            co2_scenario: co2Scenario || null,
            coal_scenario: coalScenario || null,
            gas_mu_drift_annual: Number(gasDrift),
            co2_mu_drift_annual: Number(co2Drift),
            n_years: Number(nYears),
            n_trajectories: Number(nTrajectories),
            n_runs: Number(nRuns),
            seed: Number(seed),
            display_year: Number(displayYear),
        };
    }

    async function run() {
        running = true;
        runError = null;
        try {
            result = await api.runMarketLab(buildPayload());
        } catch (e) {
            runError = e.message;
            result = null;
        } finally {
            running = false;
        }
    }

    async function doExport(kind) {
        if (exporting) return;
        exporting = kind;
        try {
            const payload = buildPayload();
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
            const r = await api.saveMarketProfile({
                name: saveName.trim(),
                description: null,
                config: buildPayload(),
                pmg_base_eur_per_kwh: Number(savePmg),
            });
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

    onMount(async () => {
        await loadProfiles();
        await run();
    });

    // ── Chart configs ──────────────────────────────────────────────────────
    function buildFanConfig(res) {
        return {
            type: "line",
            data: {
                labels: res.years,
                datasets: [
                    {
                        label: "p05",
                        data: res.annual_price_p05_eur_per_kwh,
                        borderColor: "transparent",
                        pointRadius: 0,
                        fill: false,
                    },
                    {
                        label: "p05–p95",
                        data: res.annual_price_p95_eur_per_kwh,
                        borderColor: "transparent",
                        backgroundColor: "rgba(59,130,246,0.15)",
                        pointRadius: 0,
                        fill: "-1",
                    },
                    {
                        label: "media",
                        data: res.annual_price_mean_eur_per_kwh,
                        borderColor: "#1d4ed8",
                        backgroundColor: "#1d4ed8",
                        pointRadius: 2,
                        fill: false,
                    },
                ],
            },
            options: {
                plugins: { legend: { display: true } },
                scales: {
                    x: { title: { display: true, text: "Anno" } },
                    y: { title: { display: true, text: "€/kWh" } },
                },
            },
        };
    }

    function buildDurationConfig(res) {
        return {
            type: "line",
            data: {
                labels: res.duration_curve_x.map((v) => (v * 100).toFixed(0)),
                datasets: [
                    {
                        label: "Prezzo €/kWh",
                        data: res.duration_curve_price_eur_per_kwh,
                        borderColor: "#0d9488",
                        backgroundColor: "rgba(13,148,136,0.15)",
                        pointRadius: 0,
                        fill: true,
                    },
                ],
            },
            options: {
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: "% delle ore dell'anno" } },
                    y: { title: { display: true, text: "€/kWh" } },
                },
            },
        };
    }

    function buildCapacityConfig(res) {
        return {
            type: "line",
            data: {
                labels: res.years,
                datasets: TECHS.filter((t) => res.capacity_by_year_gw[t.key]).map((t) => ({
                    label: t.label,
                    data: res.capacity_by_year_gw[t.key],
                    borderColor: t.color,
                    backgroundColor: t.color + "cc",
                    pointRadius: 0,
                    fill: true,
                })),
            },
            options: {
                plugins: { legend: { display: true } },
                scales: {
                    x: { title: { display: true, text: "Anno" } },
                    y: { stacked: true, title: { display: true, text: "GW installati" } },
                },
            },
        };
    }

    $: fanConfig = result ? buildFanConfig(result) : null;
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
            <div class="section-title">Mix e trend di capacità</div>
            <table class="mix-table">
                <thead>
                    <tr>
                        <th>Tecnologia</th>
                        <th>GW</th>
                        <th>%/anno</th>
                        <th>Anno step</th>
                        <th>GW step</th>
                    </tr>
                </thead>
                <tbody>
                    {#each TECHS as t}
                        <tr>
                            <td><span class="dot" style="background:{t.color}"></span>{t.label}</td>
                            <td><input class="input mini" type="number" min="0" step="0.5" bind:value={techState[t.key].cap} /></td>
                            <td><input class="input mini" type="number" step="0.5" bind:value={techState[t.key].growth} /></td>
                            <td><input class="input mini" type="number" min="0" bind:value={techState[t.key].stepYear} placeholder="—" /></td>
                            <td><input class="input mini" type="number" min="0" step="0.5" bind:value={techState[t.key].stepCap} placeholder="—" /></td>
                        </tr>
                    {/each}
                </tbody>
            </table>

            <div class="divider"></div>
            <div class="section-title">Scenari combustibili</div>
            <div class="grid-mini">
                <div class="form-group">
                    <label class="label" for="gas">Scenario gas</label>
                    <select id="gas" class="select" bind:value={gasScenario}>
                        <option value="base">Base</option>
                        <option value="tension">Tensione</option>
                        <option value="crisis">Crisi</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="co2">Scenario CO₂</label>
                    <select id="co2" class="select" bind:value={co2Scenario}>
                        <option value="">Default motore</option>
                        <option value="base">Base</option>
                        <option value="low">Basso</option>
                        <option value="high">Alto</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="coal">Scenario carbone</label>
                    <select id="coal" class="select" bind:value={coalScenario}>
                        <option value="">Default motore</option>
                        <option value="base">Base</option>
                        <option value="tension">Tensione</option>
                        <option value="crisis">Crisi</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="gasDrift">Drift gas %/anno</label>
                    <input id="gasDrift" class="input" type="number" step="0.01" bind:value={gasDrift} />
                </div>
            </div>

            <div class="divider"></div>
            <div class="section-title">Simulazione</div>
            <div class="grid-mini">
                <div class="form-group">
                    <label class="label" for="nYears">Anni orizzonte</label>
                    <input id="nYears" class="input" type="number" min="1" max="30" bind:value={nYears} />
                </div>
                <div class="form-group">
                    <label class="label" for="displayYear">Anno mostrato</label>
                    <input id="displayYear" class="input" type="number" min="0" bind:value={displayYear} />
                </div>
                <div class="form-group">
                    <label class="label" for="nTraj">Traiettorie</label>
                    <input id="nTraj" class="input" type="number" min="1" max="50" bind:value={nTrajectories} />
                </div>
                <div class="form-group">
                    <label class="label" for="nRuns">Run (chi fissa prezzo)</label>
                    <input id="nRuns" class="input" type="number" min="1" max="50" bind:value={nRuns} />
                </div>
            </div>

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
            <button class="btn btn-outline btn-sm" on:click={saveProfile} disabled={saving}>
                {saving ? "Salvataggio…" : "Salva profilo"}
            </button>
            {#if saveMsg}<p class="ok-msg">{saveMsg}</p>{/if}
            {#if saveError}<p class="error-msg">{saveError}</p>{/if}

            {#if savedProfiles.length}
                <ul class="profile-list">
                    {#each savedProfiles as p}
                        <li>
                            <span title={p.description || ""}>{p.name}</span>
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
        grid-template-columns: 360px 1fr;
        gap: 1.5rem;
        align-items: start;
    }
    .config-card {
        position: sticky;
        top: 1rem;
    }
    .results-col {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .chart-wrap {
        height: 320px;
    }
    .mix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }
    .mix-table th {
        text-align: left;
        font-weight: 600;
        color: var(--color-text-secondary, #6b7280);
        padding: 2px 4px;
        font-size: 0.7rem;
    }
    .mix-table td {
        padding: 2px 4px;
        white-space: nowrap;
    }
    .input.mini {
        width: 64px;
        padding: 3px 5px;
    }
    .dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 5px;
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
