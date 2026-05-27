<script>
    import { onMount, onDestroy } from 'svelte';
    import Chart from 'chart.js/auto';

    export let type = 'bar';
    export let data = {};
    export let options = {};
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

    let canvas;
    let chart;
    let mounted = false;

    onMount(() => {
        const ctx = canvas.getContext('2d');
        chart = new Chart(ctx, {
            type: type,
            data: data,
            plugins: plugins,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...options
            }
        });
        mounted = true;
    });

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
        chart.options = {
            responsive: true,
            maintainAspectRatio: false,
            ...options,
        };
        chart.update('none');
    }
</script>

<div class="chart-container">
    <canvas bind:this={canvas}></canvas>
</div>

<style>
    .chart-container {
        position: relative;
        height: 300px;
        width: 100%;
    }
</style>
