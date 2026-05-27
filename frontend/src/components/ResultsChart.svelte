<script>
    import { onMount, onDestroy } from 'svelte';
    import Chart from 'chart.js/auto';

    export let type = 'bar';
    export let data = {};
    export let options = {};

    let canvas;
    let chart;
    let mounted = false;

    onMount(() => {
        const ctx = canvas.getContext('2d');
        chart = new Chart(ctx, {
            type: type,
            data: data,
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
