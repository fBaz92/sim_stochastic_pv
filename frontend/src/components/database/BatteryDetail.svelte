<script>
    /**
     * BatteryDetail — product sheet of a catalogue battery.
     *
     * Plots the State-of-Health trajectory over the years for a chosen
     * cycling intensity. The curve uses the same linear-fade model as
     * the simulator (`BatteryBank._update_soh`): SoH decreases linearly
     * with the equivalent full cycles and reaches 0 at `cycles_life`,
     * so the chart is exactly what the Monte Carlo will apply.
     */
    import ResultsChart from "../ResultsChart.svelte";

    export let battery; // catalogue record

    let cyclesPerDay = 1.0;

    $: specs = battery.specs ?? {};
    $: cyclesLife = specs.cycles_life ?? 6000;

    $: sohConfig = (() => {
        const years = Array.from({ length: 26 }, (_, y) => y);
        // Same linear fade as the simulator: SoH = 1 − cycles/cycles_life.
        const soh = years.map((y) =>
            Math.max(0, 1 - (Number(cyclesPerDay) * 365 * y) / cyclesLife) * 100,
        );
        const eolYear = cyclesLife / (Number(cyclesPerDay) * 365) * 0.2; // 80% line hit
        return {
            type: "line",
            data: {
                labels: years,
                datasets: [
                    {
                        label: "SoH (%)",
                        data: soh,
                        borderColor: "#1d4ed8",
                        backgroundColor: "rgba(29,78,216,0.1)",
                        pointRadius: 0,
                        fill: true,
                    },
                    {
                        label: "Fine vita tipica (80%)",
                        data: years.map(() => 80),
                        borderColor: "#ef4444",
                        borderDash: [6, 4],
                        pointRadius: 0,
                        fill: false,
                    },
                ],
            },
            options: {
                plugins: { legend: { display: true } },
                scales: {
                    x: { title: { display: true, text: "Anni" } },
                    y: { min: 0, max: 100, title: { display: true, text: "Capacità residua (%)" } },
                },
            },
            eolYear,
        };
    })();
</script>

<div class="detail">
    <h3>📋 Scheda — {battery.name}</h3>
    <div class="nameplate">
        <span>Capacità <strong>{battery.capacity_kwh} kWh</strong></span>
        <span>Vita ciclica <strong>{cyclesLife} cicli</strong></span>
        {#if specs.price_eur}<span>Prezzo <strong>{specs.price_eur} €</strong></span>{/if}
        {#if specs.chemistry}<span>Chimica <strong>{specs.chemistry}</strong></span>{/if}
    </div>

    <div class="controls">
        <label>
            Cicli equivalenti al giorno: <strong>{Number(cyclesPerDay).toFixed(1)}</strong>
            <input type="range" min="0.2" max="2" step="0.1" bind:value={cyclesPerDay} />
        </label>
        <span class="hint">
            Con questo utilizzo, l'80% di capacità si raggiunge dopo
            ≈ {(cyclesLife * 0.2 / (Number(cyclesPerDay) * 365)).toFixed(1)} anni.
        </span>
    </div>

    {#key cyclesPerDay}
        <div class="chart">
            <ResultsChart
                type={sohConfig.type}
                data={sohConfig.data}
                options={sohConfig.options}
                downloadFilename="degrado_batteria"
            />
        </div>
    {/key}
    <p class="hint">
        Modello di degrado lineare con i cicli, identico a quello applicato
        dal Monte Carlo (solo la scarica usura; SoH = 1 − cicli/vita ciclica).
    </p>
</div>

<style>
    .detail {
        margin: 1rem 0 1.5rem;
        padding: 1.25rem;
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 8px;
        background: var(--color-bg-tertiary, #fafafa);
    }
    .detail h3 { margin: 0 0 0.6rem; }
    .nameplate {
        display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
        font-size: 0.85rem; color: var(--color-text-secondary);
        margin-bottom: 0.8rem;
    }
    .controls { display: flex; align-items: center; gap: 1.25rem; margin-bottom: 0.6rem; flex-wrap: wrap; }
    .controls label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.85rem; }
    .controls input[type="range"] { width: 220px; }
    .chart { height: 260px; position: relative; }
    .hint { font-size: 0.82rem; color: var(--color-text-secondary); }
</style>
