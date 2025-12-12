<script>
    import { onMount } from 'svelte';
    import Chart from 'chart.js/auto';

    export let type = 'bar';
    export let data = {};
    export let options = {};

    let canvas;
    let chart;

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

        return () => {
            if (chart) chart.destroy();
        };
    });
    
    // Watch for data changes if needed, but for now assumption is static mount
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
