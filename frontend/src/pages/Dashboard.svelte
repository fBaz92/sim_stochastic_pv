<script>
    import { onMount } from "svelte";
    import { get } from "svelte/store";
    import { api } from "../api";
    import { pendingRunId } from "../lib/stores";
    import ResultsChart from "../components/ResultsChart.svelte";

    let runs = [];
    let loading = true;
    let selectedRun = null;
    let activeTab = "overview"; // overview, energy, soc, price, inflation, cashflow, raw

    // Phase 12+ — toggle for the profit projection chart: "nominal"
    // shows the legacy nominal cumulative gain (default); "real" shows
    // the inflation-adjusted curve from plots_data.profit.mean_gain_real_eur.
    let profitView = "nominal";

    // Phase 12+ — month range filter applied to the time-series charts.
    // ``null`` = no filter (full horizon).
    let monthFrom = null;
    let monthTo = null;

    // Phase 12 — Dashboard filters + pagination.
    const PAGE_SIZE = 5;
    let visibleLimit = PAGE_SIZE;
    let filterScenarioName = "";
    let filterLocation = "";
    let filterDateFrom = "";
    let filterDateTo = "";
    let showArchived = false;
    let availableLocations = [];

    async function loadRuns() {
        loading = true;
        try {
            runs = await api.listRuns({
                limit: visibleLimit,
                scenarioName: filterScenarioName || undefined,
                location: filterLocation || undefined,
                dateFrom: filterDateFrom || undefined,
                dateTo: filterDateTo || undefined,
                includeArchived: showArchived,
            });
        } catch (e) {
            console.error(e);
            alert("Failed to load runs");
        } finally {
            loading = false;
        }
    }

    async function loadLocations() {
        try {
            availableLocations = await api.listRunLocations();
        } catch (e) {
            console.error("Failed to load locations:", e);
        }
    }

    /** Restore visibleLimit to the first page and refetch. */
    function applyFilters() {
        visibleLimit = PAGE_SIZE;
        loadRuns();
    }

    function showMore() {
        visibleLimit += PAGE_SIZE;
        loadRuns();
    }

    function clearFilters() {
        filterScenarioName = "";
        filterLocation = "";
        filterDateFrom = "";
        filterDateTo = "";
        showArchived = false;
        applyFilters();
    }

    /**
     * Archive or unarchive a run. The list is refetched so the row
     * disappears (in default mode) or stays with the archived badge
     * (when showArchived is on).
     */
    async function toggleArchive(run, ev) {
        ev.stopPropagation();
        try {
            if (run.archived_at) {
                await api.unarchiveRun(run.id);
            } else {
                await api.archiveRun(run.id);
            }
            if (selectedRun?.id === run.id) selectedRun = null;
            await loadRuns();
        } catch (e) {
            alert("Errore archiviazione: " + e.message);
        }
    }

    async function deleteRun(run, ev) {
        ev.stopPropagation();
        if (!confirm(`Eliminare definitivamente il run #${run.id}? L'operazione non è reversibile.`)) {
            return;
        }
        try {
            await api.deleteRun(run.id);
            if (selectedRun?.id === run.id) selectedRun = null;
            await loadRuns();
        } catch (e) {
            alert("Errore eliminazione: " + e.message);
        }
    }

    function selectRun(run) {
        selectedRun = run;
        activeTab = "overview";
    }

    onMount(async () => {
        await Promise.all([loadRuns(), loadLocations()]);

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
    /**
     * Slice an array of values according to the active month range filter.
     * When both ``from`` and ``to`` are null the array is returned as is.
     */
    function sliceByRange(arr) {
        if (arr == null) return arr;
        if (monthFrom == null && monthTo == null) return arr;
        const fromIdx = monthFrom != null ? Math.max(0, monthFrom) : 0;
        const toIdx = monthTo != null ? Math.min(arr.length, monthTo + 1) : arr.length;
        return arr.slice(fromIdx, toIdx);
    }

    function getProfitChart(data) {
        if (!data || !data.profit) return null;
        const p = data.profit;
        const isReal = profitView === "real";
        const meanSeries = isReal
            ? (p.mean_gain_real_eur ?? p.mean_gain_eur)
            : p.mean_gain_eur;
        const p05Series = isReal
            ? (p.p05_gain_real_eur ?? p.p05_gain_eur)
            : p.p05_gain_eur;
        const p95Series = isReal
            ? (p.p95_gain_real_eur ?? p.p95_gain_eur)
            : p.p95_gain_eur;
        const seriesLabel = isReal
            ? "Guadagno reale (€, al netto dell'inflazione)"
            : "Guadagno medio nominale (€)";
        const chartData = {
            labels: sliceByRange(p.months),
            datasets: [
                {
                    label: seriesLabel,
                    data: sliceByRange(meanSeries),
                    borderColor: isReal ? "#0d6efd" : "#198754",
                    backgroundColor: isReal ? "#0d6efd" : "#198754",
                    type: "line",
                    pointRadius: 0,
                    borderWidth: 2.5,
                },
                {
                    label: "P05",
                    data: sliceByRange(p05Series),
                    borderColor: "transparent",
                    backgroundColor: isReal
                        ? "rgba(13, 110, 253, 0.18)"
                        : "rgba(25, 135, 84, 0.18)",
                    fill: "+1",
                    type: "line",
                    pointRadius: 0,
                },
                {
                    label: "P95",
                    data: sliceByRange(p95Series),
                    borderColor: "transparent",
                    backgroundColor: "transparent",
                    fill: false,
                    type: "line",
                    pointRadius: 0,
                },
            ],
        };
        // The break-even annotation is drawn in canvas-space coordinates
        // matching the original month indices. Re-anchor it when a range
        // filter is active by subtracting the offset.
        const offset = monthFrom ?? 0;
        const shift = (v) => (v == null ? null : v - offset);
        const chartPlugins = [
            makeBreakEvenPlugin(
                shift(p.break_even_month_median),
                shift(p.break_even_month_p05),
                shift(p.break_even_month_p95),
            ),
        ];
        return { chartData, chartPlugins };
    }

    function getEnergyChart(data) {
        if (!data || !data.energy_monthly) return null;
        return {
            labels: sliceByRange(data.energy_monthly.months),
            datasets: [
                {
                    label: "PV Prod (kWh)",
                    data: sliceByRange(data.energy_monthly.pv_prod_mean_kwh),
                    backgroundColor: "#ffc107",
                },
                {
                    label: "Grid Import (kWh)",
                    data: sliceByRange(data.energy_monthly.grid_import_mean_kwh),
                    backgroundColor: "#dc3545",
                },
                {
                    label: "Self Consumed (kWh)",
                    data: sliceByRange(data.energy_monthly.solar_used_mean_kwh),
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
    /**
     * Build the fan chart datasets for the expected cumulative inflation
     * factor (Phase 11). Same visual language as the price fan chart:
     * a p05–p95 band, a handful of sample trajectories, and the mean
     * line on top.
     *
     * @param {object} data  plots_data from the run summary.
     * @returns {object|null}  Chart.js dataset config, or null if the
     *     simulator ran in deterministic mode (no fan chart to draw).
     */
    function getInflationChart(data) {
        if (!data || !data.inflation) return null;
        const inf = data.inflation;

        const datasets = [
            {
                label: "P05",
                data: inf.p05_factor,
                borderColor: "transparent",
                backgroundColor: "rgba(108, 117, 125, 0.18)",
                fill: "+1",
                pointRadius: 0,
                type: "line",
                order: 3,
            },
            {
                label: "P95",
                data: inf.p95_factor,
                borderColor: "transparent",
                backgroundColor: "transparent",
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 3,
            },
        ];

        const samplePaths = Array.isArray(inf.sample_paths) ? inf.sample_paths : [];
        for (let i = 0; i < samplePaths.length; i++) {
            datasets.push({
                label: i === 0 ? "Path simulati" : `_path_${i}`,
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

        datasets.push({
            label: "Fattore di inflazione medio",
            data: inf.mean_factor,
            borderColor: "#6f42c1",
            backgroundColor: "#6f42c1",
            borderWidth: 2.5,
            fill: false,
            pointRadius: 0,
            type: "line",
            order: 1,
        });

        return { labels: inf.years, datasets };
    }

    function getPriceChart(data) {
        if (!data || !data.price) return null;
        const p = data.price;

        const datasets = [
            // p05 / p95 band (drawn as two transparent lines with a fill between them)
            {
                label: "P05",
                data: sliceByRange(p.p05_eur_per_kwh),
                borderColor: "transparent",
                backgroundColor: "rgba(13, 110, 253, 0.18)",
                fill: "+1",
                pointRadius: 0,
                type: "line",
                order: 3,
            },
            {
                label: "P95",
                data: sliceByRange(p.p95_eur_per_kwh),
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
                data: sliceByRange(samplePaths[i]),
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
            data: sliceByRange(p.mean_eur_per_kwh),
            borderColor: "#0d6efd",
            backgroundColor: "#0d6efd",
            borderWidth: 2.5,
            fill: false,
            pointRadius: 0,
            type: "line",
            order: 1,
        });

        return { labels: sliceByRange(p.months), datasets };
    }
</script>

<div class="container dashboard">
    <div class="sidebar">
        <div class="header-actions">
            <h2>Risultati</h2>
            <button class="btn btn-outline btn-sm" on:click={loadRuns} title="Ricarica">↻</button>
        </div>

        <!-- Phase 12 — filters -->
        <div class="filters">
            <input
                class="input input-sm"
                type="search"
                placeholder="Cerca per nome scenario..."
                bind:value={filterScenarioName}
                on:input={applyFilters}
            />
            <select class="input input-sm" bind:value={filterLocation} on:change={applyFilters}>
                <option value="">Tutti i luoghi</option>
                {#each availableLocations as loc}
                    <option value={loc}>{loc}</option>
                {/each}
            </select>
            <div class="filter-row">
                <input
                    class="input input-sm"
                    type="date"
                    bind:value={filterDateFrom}
                    on:change={applyFilters}
                    title="Da"
                />
                <input
                    class="input input-sm"
                    type="date"
                    bind:value={filterDateTo}
                    on:change={applyFilters}
                    title="A"
                />
            </div>
            <label class="checkbox-label">
                <input
                    type="checkbox"
                    bind:checked={showArchived}
                    on:change={applyFilters}
                />
                Mostra archiviati
            </label>
            <button
                class="btn btn-text btn-sm clear-filters"
                type="button"
                on:click={clearFilters}
            >Reset filtri</button>
        </div>

        {#if loading}
            <p>Caricamento...</p>
        {:else if runs.length === 0}
            <p class="muted">Nessun risultato con questi filtri.</p>
        {:else}
            <div class="run-list">
                {#each runs as run (run.id)}
                    <div
                        class="run-item"
                        class:active={selectedRun?.id === run.id}
                        class:archived={run.archived_at != null}
                        role="button"
                        tabindex="0"
                        on:click={() => selectRun(run)}
                        on:keydown={(e) => (e.key === "Enter" || e.key === " ") && selectRun(run)}
                    >
                        <div class="run-header">
                            <span class={`type ${run.result_type ?? ""}`}
                                >{run.result_type === "analysis"
                                    ? "Scenario"
                                    : run.result_type === "optimization"
                                      ? "Design"
                                      : run.result_type}</span
                            >
                            <span class="date"
                                >{new Date(
                                    run.created_at,
                                ).toLocaleDateString()}</span
                            >
                        </div>
                        <div class="run-id">
                            #{run.id} • {run.summary?.scenario ?? "Custom"}
                            {#if run.summary?.location_name}
                                <span class="loc">📍 {run.summary.location_name}</span>
                            {/if}
                            {#if run.archived_at}
                                <span class="archived-badge">archiviato</span>
                            {/if}
                        </div>
                        <div class="run-actions">
                            <button
                                type="button"
                                class="icon-btn"
                                title={run.archived_at ? "Ripristina" : "Archivia"}
                                on:click={(ev) => toggleArchive(run, ev)}
                            >{run.archived_at ? "↩" : "🗄"}</button>
                            <button
                                type="button"
                                class="icon-btn danger"
                                title="Elimina definitivamente"
                                on:click={(ev) => deleteRun(run, ev)}
                            >🗑</button>
                        </div>
                    </div>
                {/each}
            </div>
            {#if runs.length >= visibleLimit}
                <button
                    type="button"
                    class="btn btn-outline btn-sm show-more"
                    on:click={showMore}
                >Mostra altri 5 ↓</button>
            {/if}
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
                              ? "Design"
                              : selectedRun.result_type}</span
                    >
                </div>
                <!-- Phase 11 — export buttons. Plain <a download> links so
                     the browser handles the save-as dialog natively. -->
                {#if selectedRun.result_type === "analysis"}
                    <div class="export-actions">
                        <a
                            class="btn btn-outline btn-sm"
                            href={api.runCashflowXlsxUrl(selectedRun.id)}
                            download
                            title="Scarica i flussi di cassa medi mensili come Excel"
                        >📊 Scarica Excel</a>
                        <a
                            class="btn btn-outline btn-sm"
                            href={api.runReportPdfUrl(selectedRun.id)}
                            download
                            title="Scarica il report PDF con tutti i grafici e i KPI"
                        >📄 Scarica PDF</a>
                    </div>
                {/if}
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
                    class:active={activeTab === "inflation"}
                    on:click={() => (activeTab = "inflation")}>Inflazione</button
                >
                <button
                    class="tab-btn"
                    class:active={activeTab === "cashflow"}
                    on:click={() => (activeTab = "cashflow")}>Cash flow</button
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
                        <!-- Card 5: Bonus fiscale totale (Phase 11). Shown only
                             when a positive total is present so legacy runs
                             without the field don't display "€ 0". -->
                        {#if s.tax_bonus_total_eur != null && s.tax_bonus_total_eur > 0}
                            <div class="card decisione-card">
                                <span class="decisione-label">Bonus fiscale totale</span>
                                <span class="decisione-value text-success">
                                    € {s.tax_bonus_total_eur.toFixed(0)}
                                </span>
                                <span class="decisione-hint">nominale, sull'orizzonte</span>
                            </div>
                        {/if}
                        <!-- Card: dedicated-withdrawal export revenue. Shown
                             only when a market profile valued the PV export. -->
                        {#if s.market?.export_revenue_total_mean_eur != null && s.market.export_revenue_total_mean_eur > 0}
                            <div class="card decisione-card">
                                <span class="decisione-label">Ricavo da immissione</span>
                                <span class="decisione-value text-success">
                                    € {s.market.export_revenue_total_mean_eur.toFixed(0)}
                                </span>
                                <span class="decisione-hint">
                                    ritiro dedicato
                                    {#if s.market.export_kwh_total_mean}
                                        · {Math.round(s.market.export_kwh_total_mean).toLocaleString("it-IT")} kWh
                                    {/if}
                                    {#if s.market.market_drives_purchase}
                                        · mercato guida l'acquisto
                                    {/if}
                                </span>
                            </div>
                        {/if}
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
                        {@const totalMonths = (selectedRun.summary.plots_data.profit?.months?.length ?? 240)}
                        {@const profitChart = getProfitChart(selectedRun.summary.plots_data)}
                        <div class="card chart-section">
                            <div class="chart-toolbar">
                                <h3>Proiezione finanziaria</h3>
                                <div class="chart-controls">
                                    <div class="seg-toggle">
                                        <button
                                            class:active={profitView === "nominal"}
                                            on:click={() => (profitView = "nominal")}
                                            title="Guadagno cumulato senza correzione per inflazione"
                                        >Nominale</button>
                                        <button
                                            class:active={profitView === "real"}
                                            on:click={() => (profitView = "real")}
                                            title="Guadagno cumulato scontato per l'inflazione (potere d'acquisto)"
                                        >Reale</button>
                                    </div>
                                    <div class="range-controls">
                                        <label>Da mese
                                            <input
                                                type="number"
                                                min="0"
                                                max={totalMonths - 1}
                                                bind:value={monthFrom}
                                                placeholder="0"
                                            />
                                        </label>
                                        <label>A mese
                                            <input
                                                type="number"
                                                min="0"
                                                max={totalMonths - 1}
                                                bind:value={monthTo}
                                                placeholder={String(totalMonths - 1)}
                                            />
                                        </label>
                                        {#if monthFrom != null || monthTo != null}
                                            <button
                                                class="btn btn-text btn-sm"
                                                on:click={() => { monthFrom = null; monthTo = null; }}
                                            >Reset</button>
                                        {/if}
                                    </div>
                                </div>
                            </div>
                            <p class="muted">
                                {#if profitView === "nominal"}
                                    Guadagno cumulato <strong>nominale</strong>: somma dei
                                    risparmi mensili (incl. bonus fiscale) meno l'investimento.
                                    Non sconta l'inflazione.
                                {:else}
                                    Guadagno cumulato <strong>reale</strong>: ogni risparmio
                                    mensile è diviso per il fattore di inflazione cumulativo
                                    del mese — rappresenta il potere d'acquisto in € di oggi.
                                {/if}
                                {#if selectedRun.summary.break_even_month_median != null}
                                    Linea tratteggiata rossa = break-even mediano
                                    ({formatBreakEven(selectedRun.summary.break_even_month_median)}).
                                    Suggerimento: trascina con shift sul grafico per zoomare un intervallo.
                                {/if}
                            </p>
                            {#if profitChart}
                                <ResultsChart
                                    type="line"
                                    data={profitChart.chartData}
                                    plugins={profitChart.chartPlugins}
                                    downloadFilename={`run_${selectedRun.id}_profit_${profitView}`}
                                    options={{
                                        scales: {
                                            x: { title: { display: true, text: "Mese dall'inizio" } },
                                            y: { title: { display: true, text: profitView === "real" ? "Guadagno reale cumulato (€)" : "Guadagno cumulato (€)" } },
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
                                downloadFilename={`run_${selectedRun.id}_energy`}
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
                                downloadFilename={`run_${selectedRun.id}_soc`}
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
                                downloadFilename={`run_${selectedRun.id}_price`}
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
                            questo run. Esegui di nuovo l'analisi per generarla.
                        </p>
                    {/if}
                {:else if activeTab === "inflation"}
                    {#if selectedRun.summary.plots_data && selectedRun.summary.plots_data.inflation}
                        <div class="card chart-section">
                            <h3>Inflazione attesa</h3>
                            <p class="muted">
                                Fattore cumulativo di inflazione per anno: 1.0 a
                                inizio orizzonte, poi composto in base ai tassi
                                annuali estratti dalla Normale troncata. La banda
                                p05–p95 misura l'incertezza Monte Carlo.
                            </p>
                            <ResultsChart
                                type="line"
                                data={getInflationChart(
                                    selectedRun.summary.plots_data,
                                )}
                                downloadFilename={`run_${selectedRun.id}_inflation`}
                                options={{
                                    plugins: {
                                        legend: {
                                            labels: {
                                                filter: (item) =>
                                                    !item.text.startsWith("_path_"),
                                            },
                                        },
                                    },
                                    scales: {
                                        x: { title: { display: true, text: "Anno" } },
                                        y: { title: { display: true, text: "Fattore cumulativo" } },
                                    },
                                }}
                            />
                        </div>
                    {:else}
                        <p>
                            Nessuna traiettoria di inflazione disponibile per
                            questo run. Attiva la modalità stocastica nella
                            sezione "Inflazione" del wizard scenario per generarle.
                        </p>
                    {/if}
                {:else if activeTab === "cashflow"}
                    {#if selectedRun.summary.plots_data?.cashflow_table}
                        {@const cf = selectedRun.summary.plots_data.cashflow_table}
                        <div class="card chart-section">
                            <div class="chart-toolbar">
                                <h3>Cash flow medio mensile</h3>
                                <div class="chart-controls">
                                    <a
                                        class="btn btn-outline btn-sm"
                                        href={api.runCashflowXlsxUrl(selectedRun.id)}
                                        download
                                    >📊 Scarica Excel</a>
                                </div>
                            </div>
                            <p class="muted">
                                Valori medi su tutti i path Monte Carlo. Il bonus fiscale è
                                una colonna separata anche se è già incluso nei risparmi nominali.
                                Profit cum. reale = nominal / fattore di inflazione cumulativo.
                            </p>
                            <div class="cashflow-table-wrap">
                                <table class="cashflow-table">
                                    <thead>
                                        <tr>
                                            <th>Mese</th>
                                            <th>Anno</th>
                                            <th>Mese/Anno</th>
                                            <th>Risp. nom. (€)</th>
                                            <th>Risp. reale (€)</th>
                                            <th>Bonus (€)</th>
                                            <th>Profit cum. nom. (€)</th>
                                            <th>Profit cum. reale (€)</th>
                                            <th>Prezzo (€/kWh)</th>
                                            <th>Inflaz. fattore</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {#each cf.months as m, i}
                                            <tr class:bonus-row={cf.bonus_per_month_eur?.[i] > 0}>
                                                <td>{m}</td>
                                                <td>{Math.floor(m / 12)}</td>
                                                <td>{(m % 12) + 1}</td>
                                                <td>{cf.mean_savings_eur?.[i]?.toFixed(2) ?? "—"}</td>
                                                <td>{cf.mean_savings_real_eur?.[i]?.toFixed(2) ?? "—"}</td>
                                                <td>{cf.bonus_per_month_eur?.[i] > 0 ? cf.bonus_per_month_eur[i].toFixed(2) : ""}</td>
                                                <td>{cf.mean_profit_cum_eur?.[i]?.toFixed(2) ?? "—"}</td>
                                                <td>{cf.mean_profit_cum_real_eur?.[i]?.toFixed(2) ?? "—"}</td>
                                                <td>{cf.mean_price_eur_per_kwh?.[i]?.toFixed(4) ?? "—"}</td>
                                                <td>{cf.mean_inflation_factor?.[i]?.toFixed(4) ?? "—"}</td>
                                            </tr>
                                        {/each}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    {:else}
                        <p>
                            Cash flow non disponibile per questo run. Esegui di
                            nuovo l'analisi per generare la tabella.
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
    /* Phase 12 — filter panel */
    .filters {
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border, #e2e8f0);
    }
    .filter-row {
        display: flex;
        gap: 0.4rem;
    }
    .filter-row .input { flex: 1; }
    .input-sm {
        padding: 0.35rem 0.5rem;
        font-size: 0.82rem;
    }
    .clear-filters {
        align-self: flex-end;
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
    }
    .show-more {
        margin-top: 0.75rem;
        width: 100%;
    }
    .archived-badge {
        font-size: 0.7rem;
        background: var(--color-bg-tertiary, #e9ecef);
        color: var(--color-text-muted, #6c757d);
        padding: 0.05rem 0.4rem;
        border-radius: 4px;
        margin-left: 0.4rem;
    }
    .loc {
        font-size: 0.75rem;
        color: var(--color-text-muted, #6c757d);
        margin-left: 0.3rem;
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
        position: relative;
    }
    .run-item:hover {
        background-color: var(--color-bg-tertiary);
    }
    .run-item.active {
        border-color: var(--color-accent);
        background-color: var(--color-bg-tertiary);
    }
    .run-item.archived {
        opacity: 0.65;
    }
    .run-actions {
        position: absolute;
        top: 0.4rem;
        right: 0.4rem;
        display: none;
        gap: 0.2rem;
    }
    .run-item:hover .run-actions,
    .run-item.active .run-actions {
        display: flex;
    }
    .icon-btn {
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 3px;
        cursor: pointer;
        font-size: 0.85rem;
        padding: 0.1rem 0.3rem;
        line-height: 1;
    }
    .icon-btn:hover { background: var(--color-bg-tertiary, #e9ecef); }
    .icon-btn.danger:hover {
        background: var(--color-danger, #dc3545);
        color: #fff;
        border-color: var(--color-danger, #dc3545);
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
    /* Phase 11 — export action buttons inside the details header */
    .export-actions {
        display: flex;
        gap: 0.5rem;
    }
    .chart-section {
        margin-bottom: 2rem;
        min-height: 400px;
    }
    /* Phase 12+ — chart toolbar (title + controls inline). */
    .chart-toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.4rem;
    }
    .chart-toolbar h3 { margin: 0; }
    .chart-controls {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .seg-toggle {
        display: inline-flex;
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: var(--radius-sm, 4px);
        overflow: hidden;
    }
    .seg-toggle button {
        background: var(--color-bg-primary, #fff);
        border: none;
        padding: 0.3rem 0.7rem;
        font-size: 0.82rem;
        cursor: pointer;
        color: var(--color-text-secondary, #6c757d);
    }
    .seg-toggle button + button {
        border-left: 1px solid var(--color-border, #e2e8f0);
    }
    .seg-toggle button.active {
        background: var(--color-accent, #0d6efd);
        color: #fff;
    }
    .range-controls {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
    }
    .range-controls label {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        color: var(--color-text-muted, #6c757d);
    }
    .range-controls input[type="number"] {
        width: 4.5rem;
        padding: 0.2rem 0.35rem;
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 3px;
        font-size: 0.78rem;
    }

    /* Phase 12+ — cash flow table. */
    .cashflow-table-wrap {
        max-height: 600px;
        overflow: auto;
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 4px;
    }
    .cashflow-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        font-variant-numeric: tabular-nums;
    }
    .cashflow-table th,
    .cashflow-table td {
        padding: 0.25rem 0.5rem;
        text-align: right;
        border-bottom: 1px solid var(--color-border, #e9ecef);
        white-space: nowrap;
    }
    .cashflow-table th {
        background: var(--color-bg-secondary, #f8f9fa);
        position: sticky;
        top: 0;
        z-index: 1;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .cashflow-table tbody tr.bonus-row {
        background: rgba(25, 135, 84, 0.05);
    }
    .cashflow-table tbody tr:hover {
        background: var(--color-bg-tertiary, #e9ecef);
    }
    .muted {
        font-size: 0.82rem;
        color: var(--color-text-muted, #6c757d);
        margin: 0 0 0.75rem;
    }
</style>
