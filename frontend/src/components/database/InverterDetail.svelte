<script>
    /**
     * InverterDetail — product sheet of a catalogue inverter.
     *
     * Draws the DC operating window as horizontal voltage bands
     * (absolute DC window, wide MPPT tracking window, full-load MPPT
     * window) so the user sees at a glance where a string voltage must
     * live, plus the per-MPPT current limits and the nameplate table.
     * Everything is client-side from the specs blob — no endpoint.
     */
    export let inverter; // catalogue record (with specs blob)

    $: specs = inverter.specs ?? {};

    // Voltage bands to draw, ordered from widest (hardware) to narrowest
    // (full-load tracking). Bands without data are skipped.
    $: bands = [
        {
            label: "Finestra DC assoluta",
            min: specs.v_dc_min_v,
            max: specs.v_dc_max_v,
            color: "rgba(239, 68, 68, 0.18)",
            border: "#ef4444",
        },
        {
            label: "Inseguimento MPPT",
            min: specs.v_mppt_min_v,
            max: specs.v_mppt_max_v,
            color: "rgba(245, 158, 11, 0.20)",
            border: "#f59e0b",
        },
        {
            label: "MPPT a pieno carico",
            min: specs.v_mppt_full_load_min_v,
            max: specs.v_mppt_full_load_max_v,
            color: "rgba(52, 211, 153, 0.25)",
            border: "#10b981",
        },
    ].filter((b) => b.min != null && b.max != null);

    $: vScale = Math.max(...bands.map((b) => b.max), 100) * 1.05;
    const pct = (v) => (v / vScale) * 100;
</script>

<div class="detail">
    <h3>📋 Scheda — {inverter.name}</h3>
    <div class="nameplate">
        <span>P AC <strong>{specs.p_ac_max_kw ?? inverter.nominal_power_kw} kW</strong></span>
        {#if specs.p_dc_max_kw}<span>P DC max <strong>{specs.p_dc_max_kw} kW</strong></span>{/if}
        {#if specs.efficiency_max}<span>Rendimento <strong>{(specs.efficiency_max * 100).toFixed(1)}%</strong></span>{/if}
        {#if specs.n_mppt_trackers}<span>MPPT <strong>{specs.n_mppt_trackers}</strong></span>{/if}
        {#if specs.i_dc_max_per_mppt_a}<span>I max op/MPPT <strong>{specs.i_dc_max_per_mppt_a} A</strong></span>{/if}
        {#if specs.i_sc_max_per_mppt_a}<span>Isc max/MPPT <strong>{specs.i_sc_max_per_mppt_a} A</strong></span>{/if}
        {#if specs.max_strings_per_mppt}<span>Stringhe/MPPT <strong>{specs.max_strings_per_mppt}</strong></span>{/if}
    </div>

    {#if bands.length > 0}
        <h4>Finestre di tensione DC</h4>
        <div class="bands">
            {#each bands as band}
                <div class="band-row">
                    <span class="band-label">{band.label}</span>
                    <div class="band-track">
                        <div
                            class="band-fill"
                            style="left: {pct(band.min)}%; width: {pct(band.max - band.min)}%; background: {band.color}; border-color: {band.border};"
                        >
                            <span class="band-min">{band.min} V</span>
                            <span class="band-max">{band.max} V</span>
                        </div>
                    </div>
                </div>
            {/each}
            <div class="band-row">
                <span class="band-label"></span>
                <div class="band-axis">
                    <span>0 V</span><span>{Math.round(vScale)} V</span>
                </div>
            </div>
        </div>
        <p class="hint">
            Una stringa ben dimensionata tiene la Voc invernale dentro la
            finestra rossa e la Vmp operativa dentro la finestra verde —
            il <a href="#/progettazione">designer</a> verifica entrambe coi
            margini di temperatura del sito.
        </p>
    {:else}
        <p class="hint">
            Nessuna finestra di tensione nel datasheet di questo inverter:
            aggiungile modificando il componente.
        </p>
    {/if}
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
    .detail h4 { margin: 0.8rem 0 0.6rem; font-size: 0.92rem; }
    .nameplate {
        display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
        font-size: 0.85rem; color: var(--color-text-secondary);
    }
    .bands { display: flex; flex-direction: column; gap: 0.55rem; }
    .band-row { display: grid; grid-template-columns: 180px 1fr; align-items: center; gap: 0.75rem; }
    .band-label { font-size: 0.82rem; color: var(--color-text-secondary); text-align: right; }
    .band-track {
        position: relative; height: 30px;
        background: var(--color-bg-primary, #fff);
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 4px;
    }
    .band-fill {
        position: absolute; top: 2px; bottom: 2px;
        border: 1px solid; border-radius: 3px;
        display: flex; justify-content: space-between; align-items: center;
        padding: 0 0.4rem; font-size: 0.72rem; white-space: nowrap;
    }
    .band-axis {
        display: flex; justify-content: space-between;
        font-size: 0.72rem; color: var(--color-text-muted, #94a3b8);
    }
    .hint { font-size: 0.82rem; color: var(--color-text-secondary); margin-top: 0.8rem; }
</style>
