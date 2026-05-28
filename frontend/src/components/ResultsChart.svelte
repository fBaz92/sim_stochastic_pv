<script>
    import { onMount, onDestroy } from 'svelte';
    import Chart from 'chart.js/auto';
    import zoomPlugin from 'chartjs-plugin-zoom';
    import { Download } from 'lucide-svelte';

    // Phase 12+ — register the official Chart.js zoom plugin globally
    // exactly once. Subsequent registrations are no-ops, so the module
    // is safe to import from anywhere.
    Chart.register(zoomPlugin);

    export let type = 'bar';
    export let data = {};
    export let options = {};
    /** Enable wheel/drag zoom & pan on the chart. Defaults to true for
     *  time-series charts; the bar chart on the Energy tab keeps it too
     *  since it is also indexed by month. */
    export let zoomable = true;
    /**
     * Optional array of Chart.js inline plugin objects (chart-level, not
     * global).  Useful for custom drawing hooks such as the break-even
     * annotation line added in Phase 4 of the roadmap.  Each element must
     * be a plain object with an `id` string and at least one Chart.js plugin
     * lifecycle method (e.g. `afterDraw`).
     *
     * NOTE: plugins are attached at chart-creation time and are not
     * updated reactively — if you need the plugin to read changing data,
     * capture those values inside a closure that is rebuilt each time you
     * reconstruct the chart (i.e. each time a new run is selected in
     * Dashboard.svelte).
     */
    export let plugins = [];

    /**
     * Optional file name (no extension) used when the user clicks the
     * download icon. Defaults to ``chart`` so legacy call sites keep
     * working without changes.
     */
    export let downloadFilename = 'chart';

    let canvas;
    let chart;
    let mounted = false;

    /** Build the Chart.js options object, merging user options with the
     *  zoom/pan defaults. Plugins[].zoom takes precedence over wheel
     *  conflicts (e.g. native page scroll) so we only enable it when
     *  the modifier key is held (shift for drag-to-zoom, no modifier
     *  for wheel/pinch). */
    function buildOptions() {
        const merged = {
            responsive: true,
            maintainAspectRatio: false,
            ...options,
        };
        if (zoomable) {
            merged.plugins = {
                ...(options.plugins || {}),
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'x',
                        modifierKey: null,
                    },
                    zoom: {
                        wheel: { enabled: true, modifierKey: 'ctrl' },
                        pinch: { enabled: true },
                        drag: {
                            enabled: true,
                            modifierKey: 'shift',
                            backgroundColor: 'rgba(13, 110, 253, 0.15)',
                            borderColor: 'rgba(13, 110, 253, 0.8)',
                            borderWidth: 1,
                        },
                        mode: 'x',
                    },
                    limits: {
                        x: { minRange: 1 },
                    },
                },
            };
        }
        return merged;
    }

    onMount(() => {
        const ctx = canvas.getContext('2d');
        chart = new Chart(ctx, {
            type: type,
            data: data,
            plugins: plugins,
            options: buildOptions(),
        });
        mounted = true;
    });

    /** Reset the zoom/pan state to the original viewport. */
    function handleResetZoom() {
        if (chart && typeof chart.resetZoom === 'function') {
            chart.resetZoom();
        }
    }

    onDestroy(() => {
        if (chart) chart.destroy();
    });

    // Reactivity: when ``data`` (or options) change after mount,
    // mutate the chart in place rather than recreating it. This lets
    // consumers drive live previews (Phase 10) cheaply.
    $: if (mounted && chart && data) {
        chart.data = data;
        // Some option subtrees (e.g. legend filters) are read on render,
        // so reassigning the whole config is the safest path.
        chart.options = buildOptions();
        chart.update('none');
    }

    /**
     * Phase 11 — export the current chart as a PNG image. Uses Chart.js'
     * native ``toBase64Image`` (which applies the device pixel ratio for
     * crisp output) and triggers the download via a transient <a> tag.
     */
    function handleDownload() {
        if (!chart) return;
        const dataUrl = chart.toBase64Image('image/png', 1);
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `${downloadFilename}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
</script>

<div class="chart-container">
    <div class="chart-tools">
        {#if zoomable}
            <button
                class="chart-tool"
                type="button"
                on:click={handleResetZoom}
                title="Reset zoom — Trascina con shift per selezionare un'area, ctrl+rotella per zoom"
                aria-label="Reset zoom"
            >⟲</button>
        {/if}
        <button
            class="chart-tool"
            type="button"
            on:click={handleDownload}
            title="Scarica grafico come PNG"
            aria-label="Scarica grafico come PNG"
        >
            <Download size="14" />
        </button>
    </div>
    <canvas bind:this={canvas}></canvas>
</div>

<style>
    .chart-container {
        position: relative;
        height: 300px;
        width: 100%;
    }
    .chart-tools {
        position: absolute;
        top: 0.25rem;
        right: 0.25rem;
        z-index: 10;
        display: flex;
        gap: 0.25rem;
    }
    .chart-tool {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 4px;
        cursor: pointer;
        padding: 0.2rem 0.4rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--color-text-secondary, #6c757d);
        font-size: 0.9rem;
        line-height: 1;
        transition: background 0.15s, color 0.15s;
    }
    .chart-tool:hover {
        background: var(--color-accent, #0d6efd);
        color: #fff;
        border-color: var(--color-accent, #0d6efd);
    }
</style>
