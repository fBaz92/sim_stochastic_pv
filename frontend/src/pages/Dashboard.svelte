<script>
    import { onMount } from "svelte";
    import { get } from "svelte/store";
    import { api } from "../api";
    import { pendingRunId } from "../lib/stores";
    import ResultsChart from "../components/ResultsChart.svelte";

    let runs = [];
    let loading = true;
    let selectedRun = null;
    let activeTab = "overview"; // overview, energy, soc

    async function loadRuns() {
        loading = true;
        try {
            runs = await api.listRuns();
        } catch (e) {
            console.error(e);
            alert("Failed to load runs");
        } finally {
            loading = false;
        }
    }

    function selectRun(run) {
        selectedRun = run;
        activeTab = "overview";
    }

    onMount(async () => {
        await loadRuns();

        // Phase 6 — Wizard redirect: if the Scenario Wizard stored a run ID
        // before redirecting here, auto-select that run and clear the store so
        // subsequent visits start fresh.
        const targetId = get(pendingRunId);
        if (targetId != null) {
            pendingRunId.set(null);
            const target = runs.find((r) => r.id === targetId);
            if (target) selectRun(target);
        }
    });

    // Chart Helpers

    /**
     * Build the Chart.js dataset config for the cumulative profit fan chart.
     *
     * Returns both ``chartData`` (datasets + labels) and ``chartPlugins``
     * (the break-even annotation plugin) so the caller can pass them
     * separately to <ResultsChart>.
     *
     * @param {object} data  plots_data from the run summary.
     * @returns {{ chartData: object, chartPlugins: object[] } | null}
     */
    function getProfitChart(data) {
        if (!data || !data.profit) return null;
        const p = data.profit;
        const chartData = {
            labels: p.months,
            datasets: [
                {
                    label: "Guadagno medio (€)",
                    data: p.mean_gain_eur,
                    borderColor: "#198754",
                    backgroundColor: "#198754",
                    type: "line",
                    pointRadius: 0,
                    borderWidth: 2.5,
                },
                {
                    label: "P05",
                    data: p.p05_gain_eur,
                    borderColor: "transparent",
                    backgroundColor: "rgba(25, 135, 84, 0.18)",
                    fill: "+1",
                    type: "line",
                    pointRadius: 0,
                },
                {
                    label: "P95",
                    data: p.p95_gain_eur,
                    borderColor: "transparent",
                    backgroundColor: "transparent",
                    fill: false,
                    type: "line",
                    pointRadius: 0,
                },
            ],
        };
        const chartPlugins = [
            makeBreakEvenPlugin(
                p.break_even_month_median ?? null,
                p.break_even_month_p05 ?? null,
                p.break_even_month_p95 ?? null,
            ),
        ];
        return { chartData, chartPlugins };
    }

    function getEnergyChart(data) {
        if (!data || !data.energy_monthly) return null;
        return {
            labels: data.energy_monthly.months,
            datasets: [
                {
                    label: "PV Prod (kWh)",
                    data: data.energy_monthly.pv_prod_mean_kwh,
                    backgroundColor: "#ffc107",
                },
                {
                    label: "Grid Import (kWh)",
                    data: data.energy_monthly.grid_import_mean_kwh,
                    backgroundColor: "#dc3545",
                },
                {
                    label: "Self Consumed (kWh)",
                    data: data.energy_monthly.solar_used_mean_kwh,
                    backgroundColor: "#0d6efd",
                },
            ],
        };
    }

    function getSocChart(data) {
        if (!data || !data.soc_profile) return null;
        // Just show Jan and Jul for brevity
        const jan = data.soc_profile.months_data.find((m) => m.month === 0);
        const jul = data.soc_profile.months_data.find((m) => m.month === 6);
        return {
            labels: data.soc_profile.hours,
            datasets: [
                {
                    label: "Jan Avg SoC",
                    data: jan ? jan.soc_mean : [],
                    borderColor: "#0d6efd",
                    tension: 0.4,
                },
                {
                    label: "Jul Avg SoC",
                    data: jul ? jul.soc_mean : [],
                    borderColor: "#ffc107",
                    tension: 0.4,
                },
            ],
        };
    }

    // ─── Phase 4 helpers ────────────────────────────────────────────────────

    /**
     * Format a break-even month count as a human-readable Italian string.
     *
     * @param {number|null} months  0-based month index, or null if unknown.
     * @returns {string}  E.g. "4 anni e 2 mesi" or "—" if null/undefined.
     */
    function formatBreakEven(months) {
        if (months == null) return "—";
        // month_index is 0-based; add 1 to get the month number.
        const m = Math.round(months) + 1;
        const years = Math.floor(m / 12);
        const rem = m % 12;
        if (years === 0) return `${rem} mes${rem === 1 ? "e" : "i"}`;
        if (rem === 0) return `${years} ann${years === 1 ? "o" : "i"}`;
        return `${years} ann${years === 1 ? "o" : "i"} e ${rem} mes${rem === 1 ? "e" : "i"}`;
    }

    /**
     * Format an IRR as a percentage string.
     *
     * @param {number|null} irr  Annual IRR as a decimal (e.g. 0.08 = 8 %).
     * @returns {string}
     */
    function formatIrr(irr) {
        if (irr == null) return "—";
        return `${(irr * 100).toFixed(1)} %`;
    }

    /**
     * Determine the CSS colour class for a probability (0–1) value.
     * Green ≥ 0.70, orange 0.40–0.69, red below 0.40.
     *
     * @param {number|null} p
     * @returns {string}  One of "text-success", "text-warning", "text-danger".
     */
    function probColorClass(p) {
        if (p == null) return "";
        if (p >= 0.70) return "text-success";
        if (p >= 0.40) return "text-warning";
        return "text-danger";
    }

    /**
     * Build a Chart.js inline plugin object that draws:
     *   1. A shaded vertical band between break_even_month_p05 and
     *      break_even_month_p95 (light red, semi-transparent).
     *   2. A dashed vertical line at break_even_month_median (red).
     *
     * The plugin is created fresh on each call so that the closure captures
     * the current break-even values — Chart.js plugins are attached at
     * creation time and are not updated reactively.
     *
     * @param {number|null} median  Median break-even month (0-based index).
     * @param {number|null} p05     5th-percentile break-even month.
     * @param {number|null} p95     95th-percentile break-even month.
     * @returns {object}  Chart.js plugin object.
     */
    function makeBreakEvenPlugin(median, p05, p95) {
        return {
            id: "breakEvenAnnotation",
            afterDraw(chart) {
                if (median == null && p05 == null && p95 == null) return;
                const { ctx, chartArea, scales } = chart;
                if (!chartArea || !scales.x) return;
                const { top, bottom } = chartArea;

                ctx.save();

                // Shaded band (p05 → p95) — drawn first so the median line
                // sits on top of it.
                if (p05 != null && p95 != null) {
                    const xLeft = scales.x.getPixelForValue(p05);
                    const xRight = scales.x.getPixelForValue(p95);
                    ctx.fillStyle = "rgba(220, 53, 69, 0.10)";
                    ctx.fillRect(xLeft, top, xRight - xLeft, bottom - top);
                }

                // Dashed vertical line at median break-even month.
                if (median != null) {
                    const xMed = scales.x.getPixelForValue(median);
                    ctx.beginPath();
                    ctx.moveTo(xMed, top);
                    ctx.lineTo(xMed, bottom);
                    ctx.strokeStyle = "rgba(220, 53, 69, 0.85)";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([6, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                ctx.restore();
            },
        };
    }

    /**
     * Build the fan chart datasets for the simulated electricity prices.
     *
     * Layered rendering, from back to front:
     *   1. p05 line (transparent border, filled toward p95 above it).
     *   2. p95 line (transparent border, anchor for the fill area).
     *   3. A handful of sample Monte Carlo trajectories (thin, semi-transparent).
     *   4. The cross-path mean as a thick solid line on top.
     *
     * For deterministic price models all sample paths and the band collapse to
     * the mean line — by design, since there is no uncertainty to depict.
     */
    function getPriceChart(data) {
        if (!data || !data.price) return null;
        const p = data.price;

        const datasets = [
            // p05 / p95 band (drawn as two transparent lines with a fill between them)
            {
                label: "P05",
                data: p.p05_eur_per_kwh,
                borderColor: "transparent",
                backgroundColor: "rgba(13, 110, 253, 0.18)",
                fill: "+1",
                pointRadius: 0,
                type: "line",
                order: 3,
            },
            {
                label: "P95",
                data: p.p95_eur_per_kwh,
                borderColor: "transparent",
                backgroundColor: "transparent",
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 3,
            },
        ];

        // Sample trajectories: very thin, semi-transparent grey strokes so the
        // user perceives the breadth of the simulation without visual clutter.
        const samplePaths = Array.isArray(p.sample_paths) ? p.sample_paths : [];
        for (let i = 0; i < samplePaths.length; i++) {
            datasets.push({
                label: i === 0 ? "Path simulati" : `_path_${i}`, // legend shown only once
                data: samplePaths[i],
                borderColor: "rgba(108, 117, 125, 0.35)",
                backgroundColor: "transparent",
                borderWidth: 1,
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 2,
            });
        }

        // Cross-path mean as the dominant series
        datasets.push({
            label: "Prezzo medio (€/kWh)",
            data: p.mean_eur_per_kwh,
            borderColor: "#0d6efd",
            backgroundColor: "#0d6efd",
            borderWidth: 2.5,
            fill: false,
            pointRadius: 0,
            type: "line",
            order: 1,
        });

        return { labels: p.months, datasets };
    }
</script>

<div class="container dashboard">
    <div class="sidebar">
        <div class="header-actions">
            <h2>Recent Runs</h2>
            <button class="btn btn-outline btn-sm" on:click={loadRuns}>↻</button>
        </div>

        {#if loading}
            <p>Loading...</p>
        {:else}
            <div class="run-list">
                {#each runs as run}
                    <button
                        type="button"
                        class="run-item"
                        class:active={selectedRun?.id === run.id}
                        on:click={() => selectRun(run)}
                    >
                        <div class="run-header">
                            <span class={`type ${run.result_type ?? ""}`}
                                >{run.result_type === "analysis"
                                    ? "Scenario"
                                    : run.result_type === "optimization"
                                      ? "Campagna"
                                      : run.result_type}</span
                            >
                            <span class="date"
                                >{new Date(
                                    run.created_at,
                                ).toLocaleDateString()}</span
                            >
                        </div>
                        <div class="run-id">
                            #{run.id} • {run.scenario
                                ? run.scenario.name
                                : "Custom"}
                        </div>
                    </button>
                {/each}
            </div>
        {/if}
    </div>

    <div class="main-content">
        {#if selectedRun}
            <div class="details-header">
                <div>
                    <h1>Run #{selectedRun.id}</h1>
                    <span class={`badge ${selectedRun.result_type ?? ""}`}
                        >{selectedRun.result_type === "analysis"
                            ? "Scenario"
                            : selectedRun.result_type === "optimization"
                              ? "Campagna"
                              : selectedRun.result_type}</span
                    >
                </div>
            </div>

            <div class="tabs">
                <button
                    class="tab-btn"
                    class:active={activeTab === "overview"}
                    on:click={() => (activeTab = "overview")}>Overview</button
                >
                <button
                    class="tab-btn"
                    class:active={activeTab === "energy"}
                    on:click={() => (activeTab = "energy")}>Energy</button
                >
                <button
                    class="tab-btn"
                    class:active={activeTab === "soc"}
                    on:click={() => (activeTab = "soc")}>Battery SoC</button
                >
                <button
                    class="tab-btn"
                    class:active={activeTab === "price"}
                    on:click={() => (activeTab = "price")}>Prezzo energia</button
                >
                <button
                    class="tab-btn"
                    class:active={activeTab === "raw"}
                    on:click={() => (activeTab = "raw")}>Raw Data</button
                >
            </div>

            <!-- Phase 4 — "Decisione" section: shown only for scenario runs -->
            {#if selectedRun.result_type === "analysis"}
                {@const s = selectedRun.summary}
                <section class="decisione-section">
                    <h2 class="decisione-title">Decisione</h2>
                    <div class="decisione-cards">
                        <!-- Card 1: Probabilità di guadagno -->
                        <div class="card decisione-card">
                            <span class="decisione-label">Probabilità di guadagno</span>
                            <span class={`decisione-value ${probColorClass(s.prob_gain)}`}>
                                {s.prob_gain != null
                                    ? `${(s.prob_gain * 100).toFixed(1)} %`
                                    : "—"}
                            </span>
                            <span class="decisione-hint">a fine orizzonte</span>
                        </div>
                        <!-- Card 2: Break-even atteso -->
                        <div class="card decisione-card">
                            <span class="decisione-label">Break-even atteso</span>
                            <span class="decisione-value">
                                {formatBreakEven(s.break_even_month_median)}
                            </span>
                            {#if s.break_even_month_p05 != null && s.break_even_month_p95 != null}
                                <span class="decisione-hint">
                                    banda 5°–95°:
                                    {formatBreakEven(s.break_even_month_p05)} –
                                    {formatBreakEven(s.break_even_month_p95)}
                                </span>
                            {:else}
                                <span class="decisione-hint">nessun percorso in pareggio</span>
                            {/if}
                        </div>
                        <!-- Card 3: IRR atteso -->
                        <div class="card decisione-card">
                            <span class="decisione-label">IRR atteso</span>
                            <span class="decisione-value">
                                {formatIrr(s.irr_mean)}
                            </span>
                            <span class="decisione-hint">media annuale</span>
                        </div>
                        <!-- Card 4: NPV mediano -->
                        <div class="card decisione-card">
                            <span class="decisione-label">NPV mediano</span>
                            <span class={`decisione-value ${s.npv_median_eur != null && s.npv_median_eur >= 0 ? "text-success" : "text-danger"}`}>
                                {s.npv_median_eur != null
                                    ? `€ ${s.npv_median_eur.toFixed(0)}`
                                    : "—"}
                            </span>
                            <span class="decisione-hint">nominale, fine orizzonte</span>
                        </div>
                    </div>
                </section>
            {/if}

            <div class="tab-content">
                {#if activeTab === "overview"}
                    <!-- Secondary summary cards (mean/real gain detail) -->
                    <div class="stat-cards">
                        <div class="card stat">
                            <h3>Guadagno medio</h3>
                            <p class="value text-specs">
                                €{selectedRun.summary.final_gain_mean_eur?.toFixed(0)}
                            </p>
                        </div>
                        <div class="card stat">
                            <h3>Guadagno reale</h3>
                            <p class="value text-specs">
                                €{selectedRun.summary.final_gain_real_mean_eur?.toFixed(0)}
                            </p>
                        </div>
                        {#if selectedRun.summary.prob_break_even_within_horizon != null}
                            <div class="card stat">
                                <h3>Prob. break-even</h3>
                                <p class="value text-specs {probColorClass(selectedRun.summary.prob_break_even_within_horizon)}">
                                    {(selectedRun.summary.prob_break_even_within_horizon * 100).toFixed(1)} %
                                </p>
                            </div>
                        {/if}
                    </div>

                    {#if selectedRun.summary.plots_data}
                        {@const profitChart = getProfitChart(selectedRun.summary.plots_data)}
                        <div class="card chart-section">
                            <h3>Proiezione finanziaria</h3>
                            {#if selectedRun.summary.break_even_month_median != null}
                                <p class="muted">
                                    Linea tratteggiata rossa = break-even mediano
                                    ({formatBreakEven(selectedRun.summary.break_even_month_median)}).
                                    Area rossa = banda p05–p95 del break-even.
                                </p>
                            {/if}
                            {#if profitChart}
                                <ResultsChart
                                    type="line"
                                    data={profitChart.chartData}
                                    plugins={profitChart.chartPlugins}
                                    options={{
                                        scales: {
                                            x: { title: { display: true, text: "Mese dall'inizio" } },
                                            y: { title: { display: true, text: "Guadagno cumulato (€)" } },
                                        },
                                    }}
                                />
                            {/if}
                        </div>
                    {/if}
                {:else if activeTab === "energy"}
                    {#if selectedRun.summary.plots_data}
                        <div class="card chart-section">
                            <h3>Monthly Energy Balance</h3>
                            <ResultsChart
                                type="bar"
                                data={getEnergyChart(
                                    selectedRun.summary.plots_data,
                                )}
                                options={{
                                    scales: {
                                        x: { stacked: false },
                                        y: { stacked: false },
                                    },
                                }}
                            />
                        </div>
                    {:else}
                        <p>No energy data available.</p>
                    {/if}
                {:else if activeTab === "soc"}
                    {#if selectedRun.summary.plots_data}
                        <div class="card chart-section">
                            <h3>Daily SoC Profiles (Jan vs Jul)</h3>
                            <ResultsChart
                                type="line"
                                data={getSocChart(
                                    selectedRun.summary.plots_data,
                                )}
                            />
                        </div>
                    {:else}
                        <p>No SoC data available.</p>
                    {/if}
                {:else if activeTab === "price"}
                    {#if selectedRun.summary.plots_data && selectedRun.summary.plots_data.price}
                        <div class="card chart-section">
                            <h3>Traiettorie simulate del prezzo dell'energia</h3>
                            <p class="muted">
                                Banda 5°–95° percentile (area chiara), media
                                Monte Carlo (linea blu) e
                                {selectedRun.summary.plots_data.price.sample_paths?.length ?? 0}
                                traiettorie campione. Con modelli deterministici
                                la banda collassa sulla media.
                            </p>
                            <ResultsChart
                                type="line"
                                data={getPriceChart(
                                    selectedRun.summary.plots_data,
                                )}
                                options={{
                                    plugins: {
                                        legend: {
                                            labels: {
                                                // hide every "_path_N" entry
                                                filter: (item) =>
                                                    !item.text.startsWith(
                                                        "_path_",
                                                    ),
                                            },
                                        },
                                    },
                                    scales: {
                                        x: { title: { display: true, text: "Mese dall'inizio" } },
                                        y: { title: { display: true, text: "EUR/kWh" } },
                                    },
                                }}
                            />
                        </div>
                    {:else}
                        <p>
                            Nessuna traiettoria di prezzo disponibile per
                            questo run (il backend potrebbe essere una versione
                            precedente alla Fase 3).
                        </p>
                    {/if}
                {:else if activeTab === "raw"}
                    <div class="raw-data">
                        <pre>{JSON.stringify(
                                selectedRun.summary,
                                null,
                                2,
                            )}</pre>
                    </div>
                {/if}
            </div>
        {:else}
            <div class="empty-state">
                <p>Select a run from the sidebar to view details.</p>
            </div>
        {/if}
    </div>
</div>

<style>
    /* ── Phase 4 — Decisione section ─────────────────────────────────────── */
    .decisione-section {
        margin-bottom: 2rem;
        padding: 1.25rem 1.5rem 1rem;
        border: 1.5px solid var(--color-accent, #0d6efd);
        border-radius: var(--radius-sm, 6px);
        background: linear-gradient(
            135deg,
            rgba(13, 110, 253, 0.04) 0%,
            transparent 60%
        );
    }
    .decisione-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--color-accent, #0d6efd);
        margin: 0 0 1rem;
    }
    .decisione-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
    }
    .decisione-card {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .decisione-label {
        font-size: 0.78rem;
        color: var(--color-text-muted, #6c757d);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .decisione-value {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.1;
        color: var(--color-text, inherit);
    }
    .decisione-hint {
        font-size: 0.72rem;
        color: var(--color-text-muted, #6c757d);
    }

    /* ── Colour helpers ─────────────────────────────────────────────────── */
    .text-success { color: var(--color-success, #198754) !important; }
    .text-warning { color: var(--color-warning, #ffc107) !important; }
    .text-danger  { color: var(--color-danger,  #dc3545) !important; }

    /* ── Layout ─────────────────────────────────────────────────────────── */
    .dashboard {
        display: grid;
        grid-template-columns: 300px 1fr;
        gap: 2rem;
        height: calc(100vh - 100px);
    }
    .sidebar {
        border-right: 1px solid var(--color-border);
        padding-right: 1rem;
        overflow-y: auto;
    }
    .main-content {
        overflow-y: auto;
        padding-bottom: 2rem;
    }
    .run-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .run-item {
        padding: 0.75rem;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-sm);
        cursor: pointer;
        transition: all 0.2s;
        background: transparent;
        text-align: left;
        width: 100%;
    }
    .run-item:hover {
        background-color: var(--color-bg-tertiary);
    }
    .run-item.active {
        border-color: var(--color-accent);
        background-color: var(--color-bg-tertiary);
    }
    .run-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
        font-size: 0.8rem;
    }
    .type {
        text-transform: uppercase;
        font-weight: bold;
    }
    .type.analysis {
        color: var(--color-success);
    }
    .type.optimization {
        color: var(--color-warning);
    }

    .stat-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .raw-data {
        margin-top: 2rem;
        background: #f1f3f5;
        padding: 1rem;
        border-radius: var(--radius-sm);
        overflow-x: auto;
    }
    .details-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    .chart-section {
        margin-bottom: 2rem;
        min-height: 400px;
    }
    .muted {
        font-size: 0.82rem;
        color: var(--color-text-muted, #6c757d);
        margin: 0 0 0.75rem;
    }
</style>
