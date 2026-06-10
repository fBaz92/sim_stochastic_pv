<script>
    /**
     * Compare — "Confronto": paired Monte Carlo over 2–4 plant designs.
     *
     * The user picks the saved designs (offers or detailed projects) and
     * a shared economic context; the backend runs the same stochastic
     * paths over every design (common random numbers), so the ΔNPV per
     * path isolates the design choice from the shared weather/price
     * randomness. Two offers differing only in price show a
     * deterministic delta; real trade-offs (more storage vs lower cost)
     * show a tight, honest delta band instead of two overlapping clouds.
     */
    import { onMount, onDestroy } from "svelte";
    import { api } from "../api.js";

    let designs = [];
    let loadProfiles = [];
    let priceProfiles = [];
    let loadError = "";

    let selectedIds = [];
    let loadProfileId = "";
    let priceProfileId = "";
    let nYears = 20;
    let nMc = 50;

    let running = false;
    let progressMsg = "";
    let error = "";
    let result = null;
    let pollTimer = null;

    function toggle(id) {
        if (selectedIds.includes(id)) {
            selectedIds = selectedIds.filter((x) => x !== id);
        } else if (selectedIds.length < 4) {
            selectedIds = [...selectedIds, id];
        }
    }

    async function runComparison() {
        error = "";
        result = null;
        if (selectedIds.length < 2) {
            error = "Seleziona da 2 a 4 impianti da confrontare.";
            return;
        }
        running = true;
        progressMsg = "Avvio…";
        try {
            const payload = {
                design_ids: selectedIds,
                n_years: Number(nYears),
                n_mc: Number(nMc),
            };
            if (loadProfileId) payload.load_profile_id = Number(loadProfileId);
            if (priceProfileId) payload.price_profile_id = Number(priceProfileId);
            const { job_id } = await api.submitCompareJob(payload);
            poll(job_id);
        } catch (e) {
            error = e.message || "Errore nell'avvio del confronto.";
            running = false;
        }
    }

    function poll(jobId) {
        clearTimeout(pollTimer);
        pollTimer = setTimeout(async () => {
            try {
                const job = await api.getJob(jobId);
                if (job.status === "done") {
                    result = job.result;
                    running = false;
                    progressMsg = "";
                    return;
                }
                if (job.status === "failed") {
                    error = "Confronto fallito: " + (job.error || "errore sconosciuto");
                    running = false;
                    return;
                }
                progressMsg = job.message || `${job.progress_done}/${job.progress_total}`;
                poll(jobId);
            } catch (e) {
                error = e.message;
                running = false;
            }
        }, 800);
    }

    const eur = (v, digits = 0) =>
        v == null ? "—" : Number(v).toLocaleString("it-IT", {
            maximumFractionDigits: digits,
        }) + " €";
    const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
    const years = (months) =>
        months == null ? "mai nell'orizzonte" : (months / 12).toFixed(1) + " anni";

    function designById(id) {
        return result?.designs.find((d) => d.design_id === id);
    }

    function verdictFor(delta) {
        const variant = designById(delta.design_id)?.name ?? "variante";
        const base = designById(delta.vs_design_id)?.name ?? "baseline";
        if (delta.prob_better >= 0.95) {
            return { cls: "ok", text: `${variant} batte ${base} praticamente sempre.` };
        }
        if (delta.prob_better >= 0.6) {
            return { cls: "ok", text: `${variant} è in vantaggio nella maggior parte degli scenari.` };
        }
        if (delta.prob_better > 0.4) {
            return { cls: "warn", text: `Sostanziale pareggio: la scelta dipende da fattori non economici.` };
        }
        if (delta.prob_better > 0.05) {
            return { cls: "ko", text: `${base} resta in vantaggio nella maggior parte degli scenari.` };
        }
        return { cls: "ko", text: `${base} batte ${variant} praticamente sempre.` };
    }

    onMount(async () => {
        try {
            const [des, loads, prices] = await Promise.all([
                api.listDesigns(),
                api.listLoadProfiles(),
                api.listPriceProfiles(),
            ]);
            designs = des;
            loadProfiles = loads;
            priceProfiles = prices;
        } catch (e) {
            loadError = "Errore nel caricamento: " + e.message;
        }
    });
    onDestroy(() => clearTimeout(pollTimer));
</script>

<div class="page">
    <h1 class="page-title">Confronto impianti</h1>
    <p class="hint">
        Confronta 2–4 impianti (offerte ricevute o progetti) sullo stesso
        contesto economico. Le simulazioni usano <strong>gli stessi
        scenari stocastici</strong> per tutti gli impianti, quindi il ΔNPV
        è un confronto appaiato: isola la scelta progettuale dal caso.
    </p>

    {#if loadError}<p class="error">{loadError}</p>{/if}

    <div class="setup-grid">
        <div class="card">
            <h3>Impianti da confrontare <span class="counter">({selectedIds.length}/4)</span></h3>
            {#if designs.length === 0}
                <p class="empty">
                    Nessun impianto salvato: creane uno da
                    <a href="#/offerta">Analizza un'offerta</a> o da
                    <a href="#/progettazione">Progettazione</a>.
                </p>
            {:else}
                <ul class="design-list">
                    {#each designs as d (d.id)}
                        <li>
                            <label class="design-row" class:baseline={selectedIds[0] === d.id}>
                                <input
                                    type="checkbox"
                                    checked={selectedIds.includes(d.id)}
                                    on:change={() => toggle(d.id)}
                                />
                                <span class="design-name">{d.name}</span>
                                <span class="design-meta">
                                    {d.data.p_ac_kw} kW
                                    · {d.data.p_dc_kwp ?? d.data.p_ac_kw} kWp
                                    {#if d.data.storage_kwh}· {d.data.storage_kwh} kWh{/if}
                                    · {eur(d.data.total_cost_eur)}
                                    {#if selectedIds[0] === d.id}<strong> · baseline</strong>{/if}
                                </span>
                            </label>
                        </li>
                    {/each}
                </ul>
                <p class="hint small">Il primo selezionato è la baseline dei delta.</p>
            {/if}
        </div>

        <div class="card">
            <h3>Contesto comune</h3>
            <div class="form-group">
                <label class="label" for="cmp-load">Profilo di consumo</label>
                <select id="cmp-load" class="input" bind:value={loadProfileId}>
                    <option value="">Standard ARERA (residenziale tipico)</option>
                    {#each loadProfiles as p}
                        <option value={String(p.id)}>{p.name}</option>
                    {/each}
                </select>
            </div>
            <div class="form-group">
                <label class="label" for="cmp-price">Prezzo dell'energia</label>
                <select id="cmp-price" class="input" bind:value={priceProfileId}>
                    <option value="">Predefinito (0,25 €/kWh, +3%/anno stocastico)</option>
                    {#each priceProfiles as p}
                        <option value={String(p.id)}>{p.name}</option>
                    {/each}
                </select>
            </div>
            <div class="two-col">
                <div class="form-group">
                    <label class="label" for="cmp-years">Orizzonte (anni)</label>
                    <input id="cmp-years" class="input" type="number" min="1" max="40" bind:value={nYears} />
                </div>
                <div class="form-group">
                    <label class="label" for="cmp-mc">Path Monte Carlo</label>
                    <input id="cmp-mc" class="input" type="number" min="10" max="200" step="10" bind:value={nMc} />
                </div>
            </div>
            <button class="btn btn-primary" on:click={runComparison} disabled={running || selectedIds.length < 2}>
                {running ? "Confronto in corso…" : "Confronta"}
            </button>
            {#if running && progressMsg}<p class="hint small">{progressMsg}</p>{/if}
            {#if error}<p class="error">{error}</p>{/if}
        </div>
    </div>

    {#if result}
        <div class="card">
            <h3>Risultati ({result.n_mc} path appaiati, seed {result.seed})</h3>
            <div class="table-wrap">
                <table class="cmp-table">
                    <thead>
                        <tr>
                            <th></th>
                            {#each result.designs as d, i}
                                <th>{d.name}{i === 0 ? " (baseline)" : ""}</th>
                            {/each}
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Costo chiavi in mano</td>
                            {#each result.designs as d}<td>{eur(d.capex_eur)}</td>{/each}</tr>
                        <tr><td>Potenza DC</td>
                            {#each result.designs as d}<td>{d.p_dc_kwp.toFixed(2)} kWp</td>{/each}</tr>
                        <tr><td>Accumulo</td>
                            {#each result.designs as d}<td>{d.storage_kwh > 0 ? d.storage_kwh + " kWh" : "—"}</td>{/each}</tr>
                        <tr><td>Produzione PV annua</td>
                            {#each result.designs as d}<td>{Math.round(d.annual_pv_kwh_mean).toLocaleString("it-IT")} kWh</td>{/each}</tr>
                        <tr><td>Guadagno finale medio</td>
                            {#each result.designs as d}
                                <td class:ok-text={d.final_gain_mean_eur > 0} class:ko-text={d.final_gain_mean_eur < 0}>
                                    <strong>{eur(d.final_gain_mean_eur)}</strong></td>
                            {/each}</tr>
                        <tr><td>Banda p05–p95</td>
                            {#each result.designs as d}
                                <td>{eur(d.final_gain_p05_eur)} … {eur(d.final_gain_p95_eur)}</td>
                            {/each}</tr>
                        <tr><td>Probabilità di guadagno</td>
                            {#each result.designs as d}<td>{pct(d.prob_gain)}</td>{/each}</tr>
                        <tr><td>IRR medio</td>
                            {#each result.designs as d}
                                <td>{d.irr_mean != null ? (d.irr_mean * 100).toFixed(1) + "%" : "—"}</td>
                            {/each}</tr>
                        <tr><td>Break-even mediano</td>
                            {#each result.designs as d}<td>{years(d.break_even_month_median)}</td>{/each}</tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="deltas">
            {#each result.deltas as delta}
                {@const v = verdictFor(delta)}
                <div class="card delta-card">
                    <h4>
                        {designById(delta.design_id)?.name}
                        <span class="vs">vs</span>
                        {designById(delta.vs_design_id)?.name}
                    </h4>
                    <p class="delta-num" class:ok-text={delta.delta_final_gain_p50_eur > 0} class:ko-text={delta.delta_final_gain_p50_eur < 0}>
                        ΔNPV mediano <strong>{eur(delta.delta_final_gain_p50_eur)}</strong>
                        <span class="band">[{eur(delta.delta_final_gain_p05_eur)} … {eur(delta.delta_final_gain_p95_eur)}]</span>
                    </p>
                    <p class="meta">
                        Probabilità che la variante batta la baseline:
                        <strong>{pct(delta.prob_better)}</strong>
                    </p>
                    <p class="verdict {v.cls}">{v.text}</p>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .page { max-width: 1050px; margin: 0 auto; }
    .hint { color: var(--color-text-secondary); font-size: 0.9rem; max-width: 75ch; }
    .hint.small { font-size: 0.82rem; margin-top: 0.4rem; }
    .setup-grid { display: grid; grid-template-columns: 3fr 2fr; gap: 1.25rem; margin-top: 1rem; }
    @media (max-width: 900px) { .setup-grid { grid-template-columns: 1fr; } }
    .card { padding: 1.1rem 1.25rem; }
    .card h3 { margin: 0 0 0.75rem; }
    .counter { font-weight: 400; font-size: 0.85rem; color: var(--color-text-secondary); }
    .design-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }
    .design-row {
        display: flex; align-items: baseline; gap: 0.6rem;
        padding: 0.45rem 0.6rem; border: 1px solid var(--color-border, #e5e5e5);
        border-radius: 6px; cursor: pointer; font-size: 0.9rem;
    }
    .design-row.baseline { border-color: var(--color-primary, #1d4ed8); }
    .design-name { font-weight: 500; }
    .design-meta { color: var(--color-text-secondary); font-size: 0.82rem; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .form-group { margin-bottom: 0.7rem; }
    .table-wrap { overflow-x: auto; }
    .cmp-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem; }
    .cmp-table th, .cmp-table td {
        padding: 0.4rem 0.6rem; text-align: right;
        border-bottom: 1px solid var(--color-border, #eee);
    }
    .cmp-table th:first-child, .cmp-table td:first-child {
        text-align: left; color: var(--color-text-secondary);
    }
    .deltas { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; margin-top: 1.25rem; }
    .delta-card h4 { margin: 0 0 0.5rem; }
    .vs { font-weight: 400; color: var(--color-text-secondary); font-size: 0.85rem; }
    .delta-num { font-size: 1.05rem; margin: 0.25rem 0; }
    .band { font-size: 0.82rem; color: var(--color-text-secondary); }
    .meta { color: var(--color-text-secondary); font-size: 0.88rem; margin: 0.25rem 0; }
    .verdict { font-weight: 600; margin: 0.5rem 0 0; }
    .verdict.ok { color: var(--color-success, #28a745); }
    .verdict.warn { color: var(--color-warning-text, #b8860b); }
    .verdict.ko { color: var(--color-danger, #dc3545); }
    .ok-text { color: var(--color-success, #28a745); }
    .ko-text { color: var(--color-danger, #dc3545); }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .error { color: var(--color-danger, #dc3545); margin-top: 0.6rem; }
</style>
