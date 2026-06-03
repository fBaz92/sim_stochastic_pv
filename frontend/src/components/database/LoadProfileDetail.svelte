<script>
    /**
     * LoadProfileDetail — the load-profile "consumption personality" page.
     *
     * Opens when the user clicks a saved profile. It lets the user *see* the
     * profile (a representative week per month, with uncertainty bands and a
     * baseline / appliances / HVAC breakdown) and *define* the layers that make
     * it a full personality:
     *
     *   - daily variability (log-normal AR(1) multiplier),
     *   - discrete appliances (kW × duration × times/week),
     *   - heat-pump / HVAC against a chosen climate (with weekly temperature).
     *
     * The "mixed scenario" is built in: the **home** regime carries all layers,
     * the **away** regime is the semi-constant pattern with optional variability
     * only. The preview runs server-side (``/api/profiles/load/preview``) on the
     * edited-but-unsaved shape, so changes are visible before saving. "Salva"
     * persists the layers into the profile's ``data`` (the home/away patterns
     * themselves stay editable in the quick-edit form).
     */
    import { onMount, createEventDispatcher } from "svelte";
    import { api } from "../../api.js";
    import ResultsChart from "../ResultsChart.svelte";

    export let profile; // { id, name, profile_type, data }

    const dispatch = createEventDispatcher();

    const MONTHS = ["Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"];
    const DAYS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"];

    const rng = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
    // Appliance quick-add presets (sent to the backend as custom events).
    const PRESETS = [
        { label: "Lavatrice", name: "Lavatrice", pKw: 1.5, durationH: 1.5, perWeek: 3, allowedHours: rng(9, 19) },
        { label: "Lavastoviglie", name: "Lavastoviglie", pKw: 1.2, durationH: 1.0, perWeek: 4, allowedHours: rng(13, 23) },
        { label: "Forno", name: "Forno", pKw: 2.5, durationH: 0.75, perWeek: 2, allowedHours: [11, 12, 13, 18, 19, 20] },
        { label: "Asciugatrice", name: "Asciugatrice", pKw: 2.2, durationH: 1.0, perWeek: 1.5, allowedHours: rng(10, 15) },
        { label: "Auto EV (lenta)", name: "Auto EV", pKw: 2.3, durationH: 8, perWeek: 5, allowedHours: [...rng(22, 24), ...rng(0, 7)] },
        { label: "Piano a induzione", name: "Piano induzione", pKw: 1.8, durationH: 0.5, perWeek: 7, allowedHours: [11, 12, 13, 19, 20, 21] },
    ];

    // ── Editable state ──────────────────────────────────────────────────────
    let month = 0;
    let regime = "home";
    let showComposition = false;
    let climateId = "";
    let climateProfiles = [];

    let stoch = { enabled: false, sigma_log: 0.2, phi: 0.5 };
    let app = { enabled: false, smartPv: false, items: [] };
    let hvac = {
        enabled: false, preset: "standard", area: 100,
        copH: 3.5, copC: 3.0, pMax: 3.0, spH: 20, spC: 26, dynamic: false,
    };

    // ── Preview state ─────────────────────────────────────────────────────────
    let result = null;
    let loading = false;
    let error = null;
    let saving = false;
    let saveMsg = "";
    let previewTimer = null;

    function initFromProfile() {
        const d = profile.data || {};
        const s = d.stochastic || {};
        stoch = {
            enabled: !!s.enabled,
            sigma_log: s.sigma_log ?? 0.2,
            phi: s.phi_intra_day ?? 0.5,
        };
        const a = d.appliances || {};
        app = {
            enabled: !!a.enabled,
            smartPv: !!a.smart_pv,
            items: (a.items || []).map(fromItemPayload),
        };
        const t = d.thermal || {};
        const h = t.house || {};
        const hp = t.heat_pump || {};
        const sp = t.setpoint || {};
        hvac = {
            enabled: !!t.enabled,
            preset: h.insulation_preset ?? "standard",
            area: h.floor_area_m2 ?? 100,
            copH: hp.cop_heating ?? 3.5,
            copC: hp.cop_cooling ?? 3.0,
            pMax: hp.p_elec_max_kw ?? 3.0,
            spH: sp.t_setpoint_heating_c ?? 20,
            spC: sp.t_setpoint_cooling_c ?? 26,
            dynamic: !!t.dynamic,
        };
    }

    function fromItemPayload(it) {
        const freq = Array.isArray(it.monthly_frequency)
            ? it.monthly_frequency[0]
            : (Array.isArray(it.monthly_frequency_override) ? it.monthly_frequency_override[0] : 0);
        return {
            name: it.name || it.type || "Elettrodomestico",
            pKw: it.p_kw ?? 1.0,
            durationH: it.duration_hours ?? 1.0,
            perWeek: freq ? Number((freq * 12 / 52).toFixed(2)) : 1,
            allowedHours: it.allowed_hours || rng(6, 23),
        };
    }

    function toItemPayload(it) {
        const freqPerMonth = Number(it.perWeek) * (52 / 12);
        const allowed = it.allowedHours && it.allowedHours.length ? it.allowedHours : rng(0, 24);
        return {
            type: "custom",
            name: it.name || "Elettrodomestico",
            p_kw: Number(it.pKw),
            duration_hours: Number(it.durationH),
            monthly_frequency: Array(12).fill(freqPerMonth),
            allowed_hours: allowed,
            schedule_mode: app.smartPv ? "smart_pv" : "naive_timer",
        };
    }

    // Merge the edited layers back into the profile's data (patterns untouched).
    function buildData() {
        const d = { ...(profile.data || {}) };
        d.stochastic = {
            enabled: stoch.enabled,
            sigma_log: Number(stoch.sigma_log),
            phi_intra_day: Number(stoch.phi),
        };
        d.appliances = {
            enabled: app.enabled,
            smart_pv: app.smartPv,
            items: app.items.map(toItemPayload),
        };
        if (hvac.enabled) {
            d.thermal = {
                enabled: true,
                house: { floor_area_m2: Number(hvac.area), insulation_preset: hvac.preset },
                heat_pump: { cop_heating: Number(hvac.copH), cop_cooling: Number(hvac.copC), p_elec_max_kw: Number(hvac.pMax) },
                setpoint: { t_setpoint_heating_c: Number(hvac.spH), t_setpoint_cooling_c: Number(hvac.spC) },
                dynamic: hvac.dynamic,
            };
        } else {
            d.thermal = { enabled: false };
        }
        return d;
    }

    async function runPreview() {
        loading = true;
        error = null;
        try {
            result = await api.previewLoadProfileInline({
                profile_type: profile.profile_type,
                data: buildData(),
                month,
                regime,
                climate_profile_id: climateId ? Number(climateId) : null,
                n_paths: 80,
                seed: 42,
            });
        } catch (e) {
            error = e.message;
            result = null;
        } finally {
            loading = false;
        }
    }

    // Debounced re-preview used by every control so quick edits don't spam.
    function schedulePreview() {
        if (previewTimer) clearTimeout(previewTimer);
        previewTimer = setTimeout(runPreview, 450);
    }

    function addPreset(p) {
        app.items = [...app.items, { name: p.name, pKw: p.pKw, durationH: p.durationH, perWeek: p.perWeek, allowedHours: p.allowedHours }];
        app.enabled = true;
        schedulePreview();
    }

    function addBlank() {
        app.items = [...app.items, { name: "Nuovo", pKw: 1.0, durationH: 1.0, perWeek: 2, allowedHours: rng(6, 23) }];
        app.enabled = true;
    }

    function removeItem(i) {
        app.items = app.items.filter((_, idx) => idx !== i);
        schedulePreview();
    }

    async function save() {
        saving = true;
        error = null;
        saveMsg = "";
        try {
            await api.updateLoadProfile(profile.id, {
                name: profile.name,
                profile_type: profile.profile_type,
                data: buildData(),
            });
            saveMsg = "Personalità del profilo salvata.";
            dispatch("saved");
        } catch (e) {
            error = e.message;
        } finally {
            saving = false;
        }
    }

    onMount(async () => {
        initFromProfile();
        try {
            climateProfiles = await api.listClimateProfiles();
        } catch (e) {
            climateProfiles = [];
        }
        await runPreview();
    });

    // ── Chart configs ──────────────────────────────────────────────────────
    $: weekLabels = result
        ? result.week_hours.map((i) => (i % 24 === 0 ? DAYS[Math.floor(i / 24)] : ""))
        : [];

    function bandDatasets(p05, p95, mean, color, label) {
        return [
            { label: "p05", data: p05, borderColor: "transparent", pointRadius: 0, fill: false },
            { label: `${label} (banda p05–p95)`, data: p95, borderColor: "transparent", backgroundColor: color + "26", pointRadius: 0, fill: "-1" },
            { label, data: mean, borderColor: color, backgroundColor: color, pointRadius: 0, borderWidth: 2, fill: false },
        ];
    }

    function buildLoadChart(res) {
        const common = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true, labels: { filter: (it) => it.text !== "p05" } } },
            scales: {
                x: { title: { display: true, text: "Settimana tipo (Lun → Dom)" }, ticks: { maxRotation: 0, autoSkip: false } },
                y: { title: { display: true, text: "kW" }, beginAtZero: true, stacked: showComposition },
            },
        };
        if (showComposition) {
            return {
                type: "line",
                data: {
                    labels: weekLabels,
                    datasets: [
                        { label: "Baseline", data: res.baseline_kw_mean, borderColor: "#3b82f6", backgroundColor: "#3b82f6aa", pointRadius: 0, fill: true },
                        { label: "Elettrodomestici", data: res.appliance_kw_mean, borderColor: "#f59e0b", backgroundColor: "#f59e0baa", pointRadius: 0, fill: true },
                        { label: "HVAC", data: res.hvac_kw_mean, borderColor: "#ef4444", backgroundColor: "#ef4444aa", pointRadius: 0, fill: true },
                    ],
                },
                options: common,
            };
        }
        return {
            type: "line",
            data: { labels: weekLabels, datasets: bandDatasets(res.total_kw_p05, res.total_kw_p95, res.total_kw_mean, "#1d4ed8", "Carico totale") },
            options: common,
        };
    }

    function buildTempChart(res) {
        const datasets = bandDatasets(res.temp_out_c_p05, res.temp_out_c_p95, res.temp_out_c_mean, "#0d9488", "T esterna");
        if (res.temp_in_c_mean) {
            datasets.push({ label: "T interna", data: res.temp_in_c_mean, borderColor: "#dc2626", backgroundColor: "#dc2626", pointRadius: 0, borderWidth: 2, borderDash: [5, 3], fill: false });
        }
        return {
            type: "line",
            data: { labels: weekLabels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: true, labels: { filter: (it) => it.text !== "p05" } } },
                scales: {
                    x: { title: { display: true, text: "Settimana tipo (Lun → Dom)" }, ticks: { maxRotation: 0, autoSkip: false } },
                    y: { title: { display: true, text: "°C" } },
                },
            },
        };
    }

    $: loadChart = result ? buildLoadChart(result) : null;
    $: tempChart = result && result.has_thermal ? buildTempChart(result) : null;
    $: applianceRows = result
        ? Object.entries(result.appliance_kwh_annual_by_name).sort((a, b) => b[1] - a[1])
        : [];
</script>

<div class="detail">
    <div class="top-bar">
        <button class="btn btn-ghost btn-sm" on:click={() => dispatch("close")}>← Torna ai profili</button>
        <h2>{profile.name}</h2>
        <div class="spacer"></div>
        <button class="btn btn-primary" on:click={save} disabled={saving}>
            {saving ? "Salvataggio…" : "💾 Salva personalità"}
        </button>
    </div>
    {#if saveMsg}<p class="ok-msg">{saveMsg}</p>{/if}
    {#if error}<p class="error-msg">{error}</p>{/if}

    <div class="grid">
        <!-- ── Controls ─────────────────────────────────────────────── -->
        <div class="card controls">
            <div class="row">
                <div class="form-group">
                    <label class="label" for="lp-month">Mese</label>
                    <select id="lp-month" class="select" bind:value={month} on:change={runPreview}>
                        {#each MONTHS as m, i}<option value={i}>{m}</option>{/each}
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="lp-regime">Regime</label>
                    <select id="lp-regime" class="select" bind:value={regime} on:change={runPreview}>
                        <option value="home">A casa (personalità completa)</option>
                        <option value="away">Via (semi-costante)</option>
                    </select>
                </div>
            </div>
            <label class="check"><input type="checkbox" bind:checked={showComposition} /> Mostra composizione (baseline / elettrodomestici / HVAC)</label>

            <div class="divider"></div>
            <div class="section-title">Variabilità giornaliera</div>
            <label class="check"><input type="checkbox" bind:checked={stoch.enabled} on:change={schedulePreview} /> Abilita variabilità stocastica</label>
            {#if stoch.enabled}
                <div class="form-group">
                    <label class="label" for="lp-sigma">Ampiezza σ (≈ {(stoch.sigma_log * 100).toFixed(0)}% 1σ): {stoch.sigma_log}</label>
                    <input id="lp-sigma" type="range" min="0" max="0.6" step="0.02" bind:value={stoch.sigma_log} on:input={schedulePreview} />
                </div>
                <div class="form-group">
                    <label class="label" for="lp-phi">Persistenza intra-day φ: {stoch.phi}</label>
                    <input id="lp-phi" type="range" min="0" max="0.95" step="0.05" bind:value={stoch.phi} on:input={schedulePreview} />
                </div>
            {/if}

            {#if regime === "home"}
                <div class="divider"></div>
                <div class="section-title">Elettrodomestici discreti</div>
                <label class="check"><input type="checkbox" bind:checked={app.enabled} on:change={schedulePreview} /> Abilita elettrodomestici</label>
                {#if app.enabled}
                    <label class="check"><input type="checkbox" bind:checked={app.smartPv} on:change={schedulePreview} /> Avvia preferibilmente con il sole (smart PV)</label>
                    <div class="preset-row">
                        {#each PRESETS as p}
                            <button type="button" class="chip" on:click={() => addPreset(p)}>+ {p.label}</button>
                        {/each}
                        <button type="button" class="chip chip-blank" on:click={addBlank}>+ Personalizzato</button>
                    </div>
                    {#if app.items.length}
                        <table class="app-table">
                            <thead><tr><th>Nome</th><th>kW</th><th>Ore</th><th>×/sett.</th><th></th></tr></thead>
                            <tbody>
                                {#each app.items as it, i}
                                    <tr>
                                        <td><input class="input mini-w" bind:value={it.name} on:change={schedulePreview} /></td>
                                        <td><input class="input mini" type="number" min="0" step="0.1" bind:value={it.pKw} on:change={schedulePreview} /></td>
                                        <td><input class="input mini" type="number" min="0" step="0.25" bind:value={it.durationH} on:change={schedulePreview} /></td>
                                        <td><input class="input mini" type="number" min="0" step="0.5" bind:value={it.perWeek} on:change={schedulePreview} /></td>
                                        <td><button type="button" class="link-btn" on:click={() => removeItem(i)} title="Rimuovi">×</button></td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    {:else}
                        <p class="hint">Nessun elettrodomestico. Aggiungine dai preset o personalizzato.</p>
                    {/if}
                {/if}

                <div class="divider"></div>
                <div class="section-title">Pompa di calore / HVAC</div>
                <label class="check"><input type="checkbox" bind:checked={hvac.enabled} on:change={schedulePreview} /> Abilita HVAC (richiede un clima)</label>
                {#if hvac.enabled}
                    <div class="form-group">
                        <label class="label" for="lp-climate">Profilo climatico</label>
                        <select id="lp-climate" class="select" bind:value={climateId} on:change={runPreview}>
                            <option value="">— seleziona un clima —</option>
                            {#each climateProfiles as c}<option value={c.id}>{c.name}</option>{/each}
                        </select>
                        {#if !climateId}<p class="hint warn">Seleziona un profilo climatico per simulare l'HVAC e vedere la temperatura.</p>{/if}
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label class="label" for="lp-iso">Isolamento</label>
                            <select id="lp-iso" class="select" bind:value={hvac.preset} on:change={schedulePreview}>
                                <option value="poor">Scarso (anni '60-'70)</option>
                                <option value="standard">Medio (anni '90)</option>
                                <option value="good">Buono (NZEB / classe A)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="label" for="lp-area">Superficie m²</label>
                            <input id="lp-area" class="input" type="number" min="10" step="5" bind:value={hvac.area} on:change={schedulePreview} />
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group"><label class="label" for="lp-ch">COP risc.</label><input id="lp-ch" class="input" type="number" min="1" step="0.1" bind:value={hvac.copH} on:change={schedulePreview} /></div>
                        <div class="form-group"><label class="label" for="lp-cc">COP raffr.</label><input id="lp-cc" class="input" type="number" min="1" step="0.1" bind:value={hvac.copC} on:change={schedulePreview} /></div>
                        <div class="form-group"><label class="label" for="lp-pm">P max kW</label><input id="lp-pm" class="input" type="number" min="0.5" step="0.5" bind:value={hvac.pMax} on:change={schedulePreview} /></div>
                    </div>
                    <div class="row">
                        <div class="form-group"><label class="label" for="lp-sh">Setpoint risc. °C</label><input id="lp-sh" class="input" type="number" step="0.5" bind:value={hvac.spH} on:change={schedulePreview} /></div>
                        <div class="form-group"><label class="label" for="lp-sc">Setpoint raffr. °C</label><input id="lp-sc" class="input" type="number" step="0.5" bind:value={hvac.spC} on:change={schedulePreview} /></div>
                    </div>
                    <label class="check"><input type="checkbox" bind:checked={hvac.dynamic} on:change={schedulePreview} /> Modello dinamico RC (temperatura interna)</label>
                {/if}
            {/if}
        </div>

        <!-- ── Results ──────────────────────────────────────────────── -->
        <div class="results">
            {#if loading && !result}
                <div class="card"><p class="text-meta">Calcolo dell'anteprima in corso…</p></div>
            {:else if result}
                <div class="card kpi-card">
                    <div class="kpis">
                        <div class="kpi"><div class="kpi-v">{result.annual_kwh_mean.toFixed(0)}</div><div class="kpi-l">kWh / anno (totale)</div></div>
                        <div class="kpi"><div class="kpi-v">{result.baseline_kwh_annual.toFixed(0)}</div><div class="kpi-l">baseline</div></div>
                        {#if result.has_appliances}<div class="kpi"><div class="kpi-v">{result.appliance_kwh_annual.toFixed(0)}</div><div class="kpi-l">elettrodomestici</div></div>{/if}
                        {#if result.has_hvac}<div class="kpi"><div class="kpi-v">{result.hvac_kwh_annual_mean.toFixed(0)}</div><div class="kpi-l">HVAC</div></div>{/if}
                    </div>
                    {#if loading}<p class="hint">aggiornamento…</p>{/if}
                </div>

                <div class="card">
                    <div class="section-title">Settimana tipo — {MONTHS[result.month]} ({regime === "home" ? "a casa" : "via"})</div>
                    <div class="chart-wrap"><ResultsChart type={loadChart.type} data={loadChart.data} options={loadChart.options} downloadFilename="profilo_carico_settimana" /></div>
                </div>

                {#if result.has_thermal && tempChart}
                    <div class="card">
                        <div class="section-title">Temperatura settimanale — {MONTHS[result.month]}</div>
                        <div class="chart-wrap"><ResultsChart type={tempChart.type} data={tempChart.data} options={tempChart.options} downloadFilename="temperatura_settimana" /></div>
                    </div>
                {/if}

                {#if applianceRows.length}
                    <div class="card">
                        <div class="section-title">Consumo annuo per elettrodomestico</div>
                        <table class="kwh-table">
                            <thead><tr><th>Elettrodomestico</th><th>kWh / anno</th></tr></thead>
                            <tbody>
                                {#each applianceRows as [name, kwh]}<tr><td>{name}</td><td>{kwh.toFixed(0)}</td></tr>{/each}
                            </tbody>
                        </table>
                    </div>
                {/if}
            {:else}
                <div class="card"><p class="text-meta">Nessuna anteprima disponibile.</p></div>
            {/if}
        </div>
    </div>
</div>

<style>
    .top-bar { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
    .top-bar h2 { margin: 0; }
    .spacer { flex: 1; }
    .grid { display: grid; grid-template-columns: 380px 1fr; gap: 1.5rem; align-items: start; }
    .controls { padding: 1.25rem; position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow-y: auto; }
    .results { display: flex; flex-direction: column; gap: 1.5rem; min-width: 0; }
    .chart-wrap { height: 320px; }
    .row { display: flex; gap: 0.75rem; }
    .row .form-group { flex: 1; }
    .divider { height: 1px; background: var(--color-border, #e5e7eb); margin: 1rem 0; }
    .section-title { font-weight: 600; margin-bottom: 0.5rem; }
    .check { display: flex; align-items: center; gap: 0.4rem; font-size: 0.88rem; margin: 0.4rem 0; cursor: pointer; }
    .preset-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
    .chip { border: 1px solid var(--color-border, #d1d5db); background: var(--color-bg-secondary, #f9fafb); border-radius: 999px; padding: 0.2rem 0.6rem; font-size: 0.8rem; cursor: pointer; }
    .chip:hover { border-color: var(--color-accent, #3b82f6); }
    .chip-blank { border-style: dashed; }
    .app-table, .kwh-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .app-table th, .kwh-table th { text-align: left; color: var(--color-text-secondary, #6b7280); font-size: 0.72rem; padding: 2px 3px; }
    .app-table td, .kwh-table td { padding: 2px 3px; border-bottom: 1px solid var(--color-border, #f1f5f9); }
    .input.mini { width: 56px; padding: 2px 4px; }
    .input.mini-w { width: 100%; padding: 2px 4px; }
    .kpis { display: flex; flex-wrap: wrap; gap: 1rem; }
    .kpi { min-width: 90px; }
    .kpi-v { font-size: 1.4rem; font-weight: 700; color: var(--color-accent, #1d4ed8); }
    .kpi-l { font-size: 0.75rem; color: var(--color-text-secondary, #6b7280); }
    .hint { font-size: 0.78rem; color: var(--color-text-secondary, #6b7280); margin: 0.3rem 0; }
    .hint.warn { color: #b45309; }
    .ok-msg { color: #16a34a; font-size: 0.85rem; }
    .error-msg { color: #dc2626; font-size: 0.85rem; }
    .link-btn { background: none; border: none; color: #dc2626; cursor: pointer; font-size: 1rem; line-height: 1; }
    @media (max-width: 900px) {
        .grid { grid-template-columns: 1fr; }
        .controls { position: static; max-height: none; }
    }
</style>
