<script>
    import { onMount } from "svelte";
    import { api } from "../api";
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

    onMount(loadRuns);

    // Chart Helpers
    function getProfitChart(data) {
        if (!data || !data.profit) return null;
        return {
            labels: data.profit.months,
            datasets: [
                {
                    label: "Mean Gain (€)",
                    data: data.profit.mean_gain_eur,
                    borderColor: "#198754",
                    backgroundColor: "#198754",
                    type: "line",
                },
                {
                    label: "P05",
                    data: data.profit.p05_gain_eur,
                    borderColor: "transparent",
                    backgroundColor: "rgba(25, 135, 84, 0.2)",
                    fill: "+1",
                    type: "line",
                    pointRadius: 0,
                },
                {
                    label: "P95",
                    data: data.profit.p95_gain_eur,
                    borderColor: "transparent",
                    backgroundColor: "transparent",
                    fill: false,
                    type: "line",
                    pointRadius: 0,
                },
            ],
        };
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

            <div class="tab-content">
                {#if activeTab === "overview"}
                    <div class="stat-cards">
                        <div class="card stat">
                            <h3>Prob. Gain</h3>
                            <p class="value text-specs">
                                {(selectedRun.summary.prob_gain * 100).toFixed(
                                    1,
                                )}%
                            </p>
                        </div>
                        <div class="card stat">
                            <h3>Mean Gain</h3>
                            <p class="value text-specs">
                                €{selectedRun.summary.final_gain_mean_eur?.toFixed(
                                    0,
                                )}
                            </p>
                        </div>
                        <div class="card stat">
                            <h3>Real Gain</h3>
                            <p class="value text-specs">
                                €{selectedRun.summary.final_gain_real_mean_eur?.toFixed(
                                    0,
                                )}
                            </p>
                        </div>
                    </div>

                    {#if selectedRun.summary.plots_data}
                        <div class="card chart-section">
                            <h3>Financial Projection</h3>
                            <ResultsChart
                                type="line"
                                data={getProfitChart(
                                    selectedRun.summary.plots_data,
                                )}
                            />
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
</style>
