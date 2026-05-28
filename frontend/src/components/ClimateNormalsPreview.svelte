<!--
    ClimateNormalsPreview — Phase 14.

    Read-only 12-month preview of the location climate normals returned by
    ``GET /api/external/climate-normals``. Renders a compact table with one
    column per month: Tmax (°C), Tmin (°C), Tavg (°C), and the
    cloud-cover-derived ``p_sunny`` probability.

    Phase 15 will replace this with an interactive fan-chart preview of the
    simulated temperature paths from the ThermalModel; this minimal table is
    the placeholder until then.

    Props:
        data        — the ClimateNormalsResponse JSON from the backend, or
                      null while loading.
        loading     — true while the request is in flight (parent-driven).
        error       — error message string or null.
-->
<script>
    const MONTHS_SHORT = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                          "Lug", "Ago", "Set", "Ott", "Nov", "Dic"];

    export let data = null;
    export let loading = false;
    export let error = null;
</script>

{#if loading}
    <div class="info-box">
        <p>Scarico le normali climatiche per questa località…</p>
    </div>
{:else if error}
    <div class="info-box error">
        <p><strong>Open-Meteo non disponibile:</strong> {error}</p>
        <p class="hint">
            Puoi comunque procedere con l'import PVGIS: i valori di p_sunny
            riceveranno un default conservativo.
        </p>
    </div>
{:else if data}
    <div class="preview-card card subtle">
        <div class="preview-header">
            <strong>Clima locale (Open-Meteo Archive, {data.start_year}–{data.end_year})</strong>
            <span class="text-meta">
                {data.latitude.toFixed(2)}°, {data.longitude.toFixed(2)}°
                {#if data.elevation_m != null} · {data.elevation_m.toFixed(0)} m{/if}
            </span>
        </div>
        <div class="month-grid-header">
            <span class="row-label"></span>
            {#each MONTHS_SHORT as m}
                <span class="month-col">{m}</span>
            {/each}
        </div>
        <div class="month-grid-row">
            <span class="row-label" title="Temperatura media giornaliera (°C)">🌡️ T media</span>
            {#each data.avg_tmean_c as v}
                <span class="month-col data-cell">{v.toFixed(1)}°</span>
            {/each}
        </div>
        <div class="month-grid-row">
            <span class="row-label" title="Media delle temperature massime giornaliere">▲ T max</span>
            {#each data.avg_tmax_c as v}
                <span class="month-col data-cell">{v.toFixed(1)}°</span>
            {/each}
        </div>
        <div class="month-grid-row">
            <span class="row-label" title="Media delle temperature minime giornaliere">▼ T min</span>
            {#each data.avg_tmin_c as v}
                <span class="month-col data-cell">{v.toFixed(1)}°</span>
            {/each}
        </div>
        <div class="month-grid-row">
            <span class="row-label" title="Probabilità di giornata serena (1 − copertura nuvolosa)">☀️ p_sunny</span>
            {#each data.p_sunny as v}
                <span class="month-col data-cell">{(v * 100).toFixed(0)}%</span>
            {/each}
        </div>
        <p class="footnote">
            I valori sono normali su {data.end_year - data.start_year + 1} anni di archivio
            ERA5. Verranno usati come seed di ``p_sunny`` quando importi il
            profilo da PVGIS, e come base di calibrazione del modello termico
            stocastico.
        </p>
    </div>
{/if}

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
    .month-grid-header,
    .month-grid-row {
        display: grid;
        grid-template-columns: 110px repeat(12, minmax(36px, 1fr));
        gap: 0.25rem;
        align-items: center;
        margin: 0.15rem 0;
    }
    .month-col {
        text-align: center;
        font-size: 0.78rem;
        color: var(--text-muted, #6b7280);
    }
    .month-grid-row .data-cell {
        font-variant-numeric: tabular-nums;
        font-size: 0.82rem;
        color: var(--text-strong, #111827);
    }
    .row-label {
        font-size: 0.82rem;
        color: var(--text-strong, #111827);
    }
    .text-meta {
        font-size: 0.78rem;
        color: var(--text-muted, #6b7280);
    }
    .footnote {
        margin: 0.6rem 0 0;
        font-size: 0.78rem;
        color: var(--text-muted, #6b7280);
    }
    .info-box {
        padding: 0.6rem 0.8rem;
        background: var(--bg-soft, #f3f4f6);
        border-radius: 6px;
        border: 1px solid var(--border, #d1d5db);
        margin-top: 0.8rem;
    }
    .info-box.error {
        border-color: var(--danger, #dc2626);
        background: var(--danger-bg, #fef2f2);
    }
    .hint {
        font-size: 0.82rem;
        color: var(--text-muted, #6b7280);
    }
</style>
