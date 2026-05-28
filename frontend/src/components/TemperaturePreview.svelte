<!--
    TemperaturePreview — Phase 15.

    Two complementary views of the calibrated ThermalModel:

    1. **Daily-mean fan chart** — mean line + p05/p95 band + sample paths
       in light grey, mirroring the Phase-10 price preview. Shows the
       *long-term seasonal evolution* of the daily-mean temperature.
    2. **Per-month hourly box plot** — for each calendar month, the
       p05/p25/p50/p75/p95 of the *hourly* temperatures across all paths.
       Shows both the diurnal swing (afternoon peak vs night trough) and
       the inter-path / extreme-event variability. Answers "in July, how
       hot does it actually get during the day?" — which the daily fan
       chart loses by averaging away.

    Both charts use the shared ``ResultsChart.svelte`` Chart.js wrapper.

    Props:
        data        — payload from ``GET /api/profiles/climate/{id}/preview``
                      with the daily fan keys and the ``monthly_*`` arrays.
                      ``null`` while loading.
        loading     — true while the request is in flight.
        error       — error string or null.
        height      — CSS height of each chart container. Default 280px.
        title       — Chart title, default "Profilo termico simulato".
-->
<script>
    import ResultsChart from "./ResultsChart.svelte";

    export let data = null;
    export let loading = false;
    export let error = null;
    export let height = "280px";
    export let title = "Profilo termico simulato (50 path Monte Carlo)";

    const MONTHS_SHORT = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"];
    // First day-of-year of each month (non-leap). Used to place x-axis labels.
    const MONTH_START_DOY = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

    /**
     * Build the Chart.js config for the monthly hourly box plot.
     *
     * Rendered as stacked floating bars:
     * - Outer bar: p05–p95 (light red, semi-transparent).
     * - Inner bar: p25–p75 (solid red).
     * - Scatter marker: p50 (median, white dot with red outline).
     *
     * Chart.js native bar dataset supports the ``[low, high]`` "floating
     * bar" format we use here, so no extra plugin is required.
     */
    function buildMonthlyBoxConfig(preview) {
        if (!preview || !preview.monthly_p05_c) return { data: null, options: null };

        const p05 = preview.monthly_p05_c;
        const p25 = preview.monthly_p25_c;
        const p50 = preview.monthly_p50_c;
        const p75 = preview.monthly_p75_c;
        const p95 = preview.monthly_p95_c;

        const datasets = [
            {
                label: "p05–p95",
                data: p05.map((lo, i) => [lo, p95[i]]),
                backgroundColor: "rgba(220, 53, 69, 0.18)",
                borderColor: "rgba(220, 53, 69, 0.35)",
                borderWidth: 1,
                borderSkipped: false,
                barPercentage: 0.65,
                categoryPercentage: 0.85,
                order: 3,
            },
            {
                label: "p25–p75 (50% centrale)",
                data: p25.map((lo, i) => [lo, p75[i]]),
                backgroundColor: "rgba(220, 53, 69, 0.55)",
                borderColor: "rgba(220, 53, 69, 0.9)",
                borderWidth: 1,
                borderSkipped: false,
                barPercentage: 0.45,
                categoryPercentage: 0.85,
                order: 2,
            },
            {
                label: "Mediana (p50)",
                type: "line",
                data: p50,
                showLine: false,
                pointRadius: 5,
                pointStyle: "circle",
                pointBackgroundColor: "#ffffff",
                pointBorderColor: "#dc3545",
                pointBorderWidth: 2,
                order: 1,
            },
        ];

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.parsed.y;
                            if (Array.isArray(ctx.raw)) {
                                return `${ctx.dataset.label}: ${ctx.raw[0].toFixed(1)} – ${ctx.raw[1].toFixed(1)} °C`;
                            }
                            return `${ctx.dataset.label}: ${v.toFixed(1)} °C`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Mese" },
                    grid: { display: false },
                },
                y: {
                    title: { display: true, text: "Temperatura oraria (°C)" },
                },
            },
        };

        return { data: { labels: MONTHS_SHORT, datasets }, options };
    }

    /** Compose the Chart.js data + options from the preview payload. */
    function buildChartConfig(preview) {
        if (!preview) return { data: null, options: null };
        const datasets = [
            {
                label: "P05",
                data: preview.p05_c,
                borderColor: "transparent",
                backgroundColor: "rgba(220, 53, 69, 0.18)",
                fill: "+1",
                pointRadius: 0,
                type: "line",
                order: 3,
            },
            {
                label: "P95",
                data: preview.p95_c,
                borderColor: "transparent",
                backgroundColor: "transparent",
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 3,
            },
        ];

        const paths = Array.isArray(preview.sample_paths_c) ? preview.sample_paths_c : [];
        // Cap the number of strokes for readability; the band already conveys
        // the spread.
        const maxStrokes = Math.min(paths.length, 20);
        for (let i = 0; i < maxStrokes; i++) {
            datasets.push({
                label: i === 0 ? "Path simulati" : `_path_${i}`,
                data: paths[i],
                borderColor: "rgba(108, 117, 125, 0.25)",
                backgroundColor: "transparent",
                borderWidth: 1,
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 2,
            });
        }

        datasets.push({
            label: "Temperatura media (°C)",
            data: preview.mean_c,
            borderColor: "#dc3545",
            backgroundColor: "#dc3545",
            borderWidth: 2.2,
            fill: false,
            pointRadius: 0,
            type: "line",
            order: 1,
        });

        const labels = preview.days;
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        // Hide the per-path entries from the legend.
                        filter: (item) =>
                            !(item.text && item.text.startsWith("_path_"))
                            && item.text !== "P05" && item.text !== "P95",
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (ctx) => `Giorno ${ctx[0].label}`,
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} °C`,
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Giorno dell'anno" },
                    ticks: {
                        autoSkip: false,
                        callback: function(value, index) {
                            // Place a month label at the first day-of-year
                            // of each month.
                            const i = MONTH_START_DOY.indexOf(Number(this.getLabelForValue(value)));
                            return i >= 0 ? MONTHS_SHORT[i] : "";
                        },
                    },
                },
                y: {
                    title: { display: true, text: "Temperatura (°C)" },
                },
            },
        };

        return { data: { labels, datasets }, options };
    }

    $: chartConfig = buildChartConfig(data);
    $: monthlyBoxConfig = buildMonthlyBoxConfig(data);
</script>

<div class="preview-card card subtle">
    <div class="preview-header">
        <strong>{title}</strong>
        {#if data}
            <span class="text-meta">
                Banda p05–p95 + media su {data.sample_paths_c?.length ?? 0} traiettorie ·
                Modello: stagionalità armonica + AR(1) + GPD per eventi estremi
            </span>
        {/if}
    </div>

    {#if loading}
        <div class="info-box">
            <p>Simulo le traiettorie di temperatura…</p>
        </div>
    {:else if error}
        <div class="info-box error">
            <p><strong>Preview non disponibile:</strong> {error}</p>
        </div>
    {:else if data && chartConfig.data}
        <div class="chart-wrap" style:height>
            <ResultsChart data={chartConfig.data} options={chartConfig.options} />
        </div>
        <p class="chart-hint">
            La media giornaliera è circa il punto medio tra il valore alle 02:00
            (più freddo) e quello alle 14:00 (più caldo). Il grafico qui sotto
            mostra la distribuzione delle temperature <em>orarie</em> per ogni
            mese — banda chiara = p05–p95 (90% delle ore), banda piena = p25–p75
            (50% centrale), pallino = mediana.
        </p>
        {#if monthlyBoxConfig.data}
            <div class="chart-wrap" style:height>
                <ResultsChart
                    data={monthlyBoxConfig.data}
                    options={monthlyBoxConfig.options}
                />
            </div>
        {/if}
    {:else}
        <div class="info-box">
            <p>Nessun profilo termico ancora calibrato per questa posizione.</p>
        </div>
    {/if}
</div>

<style>
    .preview-card {
        margin-top: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid var(--border, #d1d5db);
    }
    .preview-card.subtle {
        background: var(--bg-soft, #f9fafb);
    }
    .preview-header {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        margin-bottom: 0.6rem;
    }
    .text-meta {
        font-size: 0.78rem;
        color: var(--text-muted, #6b7280);
    }
    .chart-wrap {
        position: relative;
        width: 100%;
    }
    .chart-wrap + .chart-hint {
        margin: 0.8rem 0 0.4rem;
        font-size: 0.85rem;
        color: var(--text-muted, #6b7280);
        line-height: 1.4;
    }
    .info-box {
        padding: 0.6rem 0.8rem;
        background: var(--bg-soft, #f3f4f6);
        border-radius: 6px;
        border: 1px solid var(--border, #d1d5db);
    }
    .info-box.error {
        border-color: var(--danger, #dc2626);
        background: var(--danger-bg, #fef2f2);
    }
</style>
