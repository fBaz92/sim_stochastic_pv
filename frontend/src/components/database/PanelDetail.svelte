<script>
    /**
     * PanelDetail — product sheet of a catalogue panel.
     *
     * Renders the two standard curve families generated server-side by
     * the single-diode model fitted on the panel datasheet
     * (GET /api/panels/{id}/curves): I-V at several irradiances and
     * P-V at several cell temperatures, with the nameplate table.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";
    import ResultsChart from "../ResultsChart.svelte";

    export let panel; // catalogue record (with specs blob)

    let curves = null;
    let loading = false;
    let error = "";

    const PALETTE = ["#94a3b8", "#60a5fa", "#34d399", "#f59e0b", "#ef4444"];

    onMount(async () => {
        loading = true;
        try {
            curves = await api.getPanelCurves(panel.id);
        } catch (e) {
            error = e.message || "Curve non disponibili.";
        } finally {
            loading = false;
        }
    });

    function ivConfig(family) {
        return {
            type: "line",
            data: {
                datasets: family.map((c, k) => ({
                    label: `${c.irradiance_w_m2} W/m²`,
                    data: c.v.map((v, j) => ({ x: v, y: c.i[j] })),
                    borderColor: PALETTE[k % PALETTE.length],
                    pointRadius: 0,
                    fill: false,
                })).concat([{
                    label: "MPP",
                    data: family.map((c) => ({ x: c.mpp_v, y: c.mpp_i })),
                    borderColor: "#111",
                    backgroundColor: "#111",
                    pointRadius: 3,
                    showLine: false,
                }]),
            },
            options: {
                parsing: false,
                plugins: { legend: { display: true } },
                scales: {
                    x: { type: "linear", title: { display: true, text: "Tensione (V)" } },
                    y: { title: { display: true, text: "Corrente (A)" } },
                },
            },
        };
    }

    function pvConfig(family) {
        return {
            type: "line",
            data: {
                datasets: family.map((c, k) => ({
                    label: `${c.t_cell_c} °C`,
                    data: c.v.map((v, j) => ({ x: v, y: c.p[j] })),
                    borderColor: PALETTE[(k + 1) % PALETTE.length],
                    pointRadius: 0,
                    fill: false,
                })).concat([{
                    label: "MPP",
                    data: family.map((c) => ({ x: c.mpp_v, y: c.mpp_p })),
                    borderColor: "#111",
                    backgroundColor: "#111",
                    pointRadius: 3,
                    showLine: false,
                }]),
            },
            options: {
                parsing: false,
                plugins: { legend: { display: true } },
                scales: {
                    x: { type: "linear", title: { display: true, text: "Tensione (V)" } },
                    y: { title: { display: true, text: "Potenza (W)" } },
                },
            },
        };
    }

    $: specs = panel.specs ?? {};
</script>

<div class="detail">
    <h3>📋 Scheda — {panel.name}</h3>
    <div class="nameplate">
        <span>Pmax <strong>{panel.power_w} W</strong></span>
        {#if specs.v_mpp_stc_v}<span>Vmp <strong>{specs.v_mpp_stc_v} V</strong></span>{/if}
        {#if specs.i_mpp_stc_a}<span>Imp <strong>{specs.i_mpp_stc_a} A</strong></span>{/if}
        {#if specs.v_oc_stc_v}<span>Voc <strong>{specs.v_oc_stc_v} V</strong></span>{/if}
        {#if specs.i_sc_stc_a}<span>Isc <strong>{specs.i_sc_stc_a} A</strong></span>{/if}
        {#if specs.alpha_isc_pct_per_c}<span>α <strong>{specs.alpha_isc_pct_per_c} %/°C</strong></span>{/if}
        {#if specs.beta_voc_pct_per_c}<span>β <strong>{specs.beta_voc_pct_per_c} %/°C</strong></span>{/if}
        {#if specs.gamma_pmax_pct_per_c}<span>γ <strong>{specs.gamma_pmax_pct_per_c} %/°C</strong></span>{/if}
        {#if specs.v_system_max_v}<span>V sistema <strong>{specs.v_system_max_v} V</strong></span>{/if}
        {#if specs.max_series_fuse_a}<span>Max fusibile <strong>{specs.max_series_fuse_a} A</strong></span>{/if}
        {#if specs.noct_c}<span>NOCT <strong>{specs.noct_c} °C</strong></span>{/if}
    </div>

    {#if loading}
        <p>Calcolo delle curve dal modello a singolo diodo…</p>
    {:else if error}
        <p class="error">{error}</p>
    {:else if curves}
        <div class="charts">
            <div class="chart-box">
                <h4>Curve I-V per irraggiamento (cella a 25 °C)</h4>
                {#key curves}
                    {@const cfg = ivConfig(curves.irradiance_family)}
                    <div class="chart"><ResultsChart type={cfg.type} data={cfg.data} options={cfg.options} downloadFilename="curve_iv" /></div>
                {/key}
            </div>
            <div class="chart-box">
                <h4>Curve P-V per temperatura di cella (1000 W/m²)</h4>
                {#key curves}
                    {@const cfg2 = pvConfig(curves.temperature_family)}
                    <div class="chart"><ResultsChart type={cfg2.type} data={cfg2.data} options={cfg2.options} downloadFilename="curve_pv" /></div>
                {/key}
            </div>
        </div>
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
    .nameplate {
        display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem;
        font-size: 0.85rem; color: var(--color-text-secondary);
        margin-bottom: 1rem;
    }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
    @media (max-width: 980px) { .charts { grid-template-columns: 1fr; } }
    .chart-box h4 { margin: 0 0 0.4rem; font-size: 0.92rem; }
    .chart { height: 280px; position: relative; }
    .error { color: var(--color-danger, #dc3545); }
</style>
