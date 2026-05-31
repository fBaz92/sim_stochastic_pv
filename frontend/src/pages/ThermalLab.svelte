<!--
    ThermalLab — Phase 19 "Laboratorio termico".

    A dedicated section to reason about the building envelope and the heat
    pump *before* the full economic scenario. The user fixes a climate
    profile (Phase 15), a heat pump, comfort setpoints and an occupancy
    pattern, then compares several house configurations (insulation presets
    or a custom UA) over a small Monte Carlo and reads:

    - a KPI comparison table (annual HVAC energy + cost + comfort breaches +
      peak power + worst-case indoor temperature);
    - a "typical year" daily chart of HVAC energy per configuration overlaid
      with the outdoor temperature, with the worst heating/cooling days
      highlighted;
    - a cost-per-configuration bar chart;
    - an hourly preview (dynamic mode) of one configuration showing the
      setpoint vs the achieved indoor temperature and the electric draw.

    Backend: POST /api/thermal-lab/compare and /timeseries.
-->
<script>
    import { onMount } from "svelte";
    import { api } from "../api.js";
    import ResultsChart from "../components/ResultsChart.svelte";

    // ── Static reference data ──────────────────────────────────────────────
    const MONTHS_SHORT = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"];
    const MONTH_START_DOY = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    // Distinct colours per house variant (red→amber→green reads as
    // worse→better insulation when the presets are picked in order).
    const VARIANT_COLORS = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"];

    const PRESET_OPTIONS = [
        { key: "poor", label: "Scarso (anni '60–'70)", w: 2.5 },
        { key: "standard", label: "Standard (anni '90)", w: 1.5 },
        { key: "good", label: "Ottimo (NZEB / classe A)", w: 0.8 },
    ];

    // ── Climate profiles ────────────────────────────────────────────────────
    let climateProfiles = [];
    let selectedClimateId = null;
    let climateError = null;

    // ── Configuration state ─────────────────────────────────────────────────
    let selectedPresets = { poor: true, standard: true, good: true };
    let floorArea = 100;
    let customUaEnabled = false;
    let customUa = 1.0; // W/°C/m²

    let copHeating = 3.5;
    let copCooling = 3.0;
    let pElecMax = 3.0;

    let tHeating = 20.0;
    let tCooling = 26.0;

    let awayEnabled = false;
    let awayStart = 8; // hour user leaves
    let awayEnd = 17; // hour user returns
    let awaySetbackEnabled = false;
    let tAway = 16.0;

    let nightSetbackEnabled = false;
    let tNight = 17.0;

    // Energy price (Phase 19-bis): a flat scalar, or a PriceModel
    // (escalation / GBM / mean-reverting) that spreads the cost band.
    let priceMode = "fixed"; // fixed | escalating | gbm | mean_reverting
    let priceBase = 0.25; // €/kWh, also the model's base price
    let priceEscalationPct = 2.0; // %/yr (escalating)
    let priceDriftPct = 2.5; // %/yr (gbm)
    let priceVolPct = 10.0; // %/yr (gbm / mean_reverting)
    let priceMrLongTerm = 0.30; // €/kWh (mean_reverting equilibrium)
    let priceMrSpeed = 0.3; // /yr (mean_reverting)

    const PRICE_MODES = [
        { key: "fixed", label: "Fisso" },
        { key: "escalating", label: "Escalation" },
        { key: "gbm", label: "GBM (random walk)" },
        { key: "mean_reverting", label: "Mean-reverting" },
    ];

    let nPaths = 30;
    let nYears = 1;
    let seed = 42;
    let dynamic = false;

    // ── Results state ────────────────────────────────────────────────────────
    let result = null;
    let running = false;
    let runError = null;

    // ── Timeseries preview state ──────────────────────────────────────────────
    let tsVariantIndex = 0;
    let tsDays = 14;
    let tsStartDay = 0; // day-of-year window start (season selector)
    const SEASON_OPTIONS = [
        { day: 0, label: "Inverno (gen)" },
        { day: 90, label: "Primavera (apr)" },
        { day: 181, label: "Estate (lug)" },
        { day: 273, label: "Autunno (ott)" },
    ];
    let timeseries = null;
    let tsRunning = false;
    let tsError = null;

    onMount(async () => {
        try {
            climateProfiles = await api.listClimateProfiles();
            if (climateProfiles.length > 0) selectedClimateId = climateProfiles[0].id;
        } catch (e) {
            climateError = e.message;
        }
    });

    /**
     * Build the list of house-variant payloads from the selected presets and
     * the optional custom-UA variant. Shared across the compare request and
     * the timeseries preview (so the preview matches a compared variant).
     */
    function buildVariants() {
        const variants = [];
        for (const opt of PRESET_OPTIONS) {
            if (selectedPresets[opt.key]) {
                variants.push({
                    label: opt.label,
                    insulation_preset: opt.key,
                    floor_area_m2: floorArea,
                });
            }
        }
        if (customUaEnabled) {
            variants.push({
                label: `Custom (${customUa} W/°C/m²)`,
                ua_w_per_c_per_m2: customUa,
                floor_area_m2: floorArea,
            });
        }
        return variants;
    }

    /** Hours-of-day the user is at home, or null for "always home". */
    function buildHomeHours() {
        if (!awayEnabled) return null;
        const home = [];
        for (let h = 0; h < 24; h++) {
            if (h < awayStart || h >= awayEnd) home.push(h);
        }
        return home;
    }

    /** Optional 24-entry heating schedule for the night-setback feature. */
    function buildHeatingSchedule() {
        if (!nightSetbackEnabled) return null;
        const sched = [];
        for (let h = 0; h < 24; h++) {
            sched.push(h < 6 || h >= 23 ? tNight : tHeating);
        }
        return sched;
    }

    function buildSetpoint() {
        const sp = {
            t_setpoint_heating_c: tHeating,
            t_setpoint_cooling_c: tCooling,
        };
        if (awayEnabled && awaySetbackEnabled) sp.t_setpoint_away_c = tAway;
        const sched = buildHeatingSchedule();
        if (sched) sp.heating_schedule_c = sched;
        return sp;
    }

    /** Optional PriceModel block, or null when the price is a flat scalar. */
    function buildPriceBlock() {
        if (priceMode === "fixed") return null;
        const block = { model_type: priceMode, base_price_eur_per_kwh: priceBase };
        if (priceMode === "escalating") {
            block.annual_escalation = priceEscalationPct / 100;
            block.use_stochastic_escalation = true;
        } else if (priceMode === "gbm") {
            block.drift_annual = priceDriftPct / 100;
            block.volatility_annual = priceVolPct / 100;
        } else if (priceMode === "mean_reverting") {
            block.long_term_price_eur_per_kwh = priceMrLongTerm;
            block.mean_reversion_speed_annual = priceMrSpeed;
            block.volatility_annual = priceVolPct / 100;
        }
        return block;
    }

    /** Full request body shared by the compare run and the file exports. */
    function buildComparePayload() {
        return {
            climate_profile_id: selectedClimateId,
            n_paths: nPaths,
            n_years: nYears,
            seed,
            dynamic,
            home_hours_of_day: buildHomeHours(),
            electricity_price_eur_per_kwh: priceBase,
            price: buildPriceBlock(),
            heat_pump: {
                cop_heating: copHeating,
                cop_cooling: copCooling,
                p_elec_max_kw: pElecMax,
            },
            setpoint: buildSetpoint(),
            house_variants: buildVariants(),
        };
    }

    async function runCompare() {
        runError = null;
        if (!selectedClimateId) {
            runError = "Seleziona un profilo climatico.";
            return;
        }
        if (buildVariants().length === 0) {
            runError = "Seleziona almeno una configurazione di casa.";
            return;
        }
        running = true;
        result = null;
        timeseries = null;
        try {
            result = await api.compareThermalLab(buildComparePayload());
            tsVariantIndex = 0;
            await runTimeseries();
        } catch (e) {
            runError = e.message;
        } finally {
            running = false;
        }
    }

    let exporting = null; // 'xlsx' | 'pdf' | null
    let exportError = null;

    async function doExport(kind) {
        if (!result) return;
        exportError = null;
        exporting = kind;
        try {
            const payload = buildComparePayload();
            if (kind === "xlsx") await api.exportThermalLabXlsx(payload);
            else await api.exportThermalLabPdf(payload);
        } catch (e) {
            exportError = e.message;
        } finally {
            exporting = null;
        }
    }

    async function runTimeseries() {
        if (!result || !selectedClimateId) return;
        const variants = buildVariants();
        const house = variants[tsVariantIndex] ?? variants[0];
        if (!house) return;
        tsError = null;
        tsRunning = true;
        try {
            timeseries = await api.previewThermalTimeseries({
                climate_profile_id: selectedClimateId,
                n_days: tsDays,
                start_day: tsStartDay,
                seed,
                dynamic,
                home_hours_of_day: buildHomeHours(),
                heat_pump: {
                    cop_heating: copHeating,
                    cop_cooling: copCooling,
                    p_elec_max_kw: pElecMax,
                },
                setpoint: buildSetpoint(),
                house,
            });
        } catch (e) {
            tsError = e.message;
        } finally {
            tsRunning = false;
        }
    }

    // ── CSV export of the comparison table ────────────────────────────────────
    function exportCsv() {
        if (!result) return;
        const header = [
            "configurazione", "UA_kW_per_C", "kWh_anno_medio", "kWh_anno_p05",
            "kWh_anno_p95", "risc_kWh_anno", "raffr_kWh_anno",
            "costo_eur_anno_medio", "comfort_breach_h_anno",
            "picco_kW", "T_interna_min_C", "T_interna_max_C",
            "giorno_peggiore_riscaldamento", "giorno_peggiore_raffrescamento",
        ];
        const rows = result.variants.map((v) => [
            v.label, v.ua_kw_per_c.toFixed(3), v.hvac_kwh_annual_mean.toFixed(1),
            v.hvac_kwh_annual_p05.toFixed(1), v.hvac_kwh_annual_p95.toFixed(1),
            v.heating_kwh_annual_mean.toFixed(1), v.cooling_kwh_annual_mean.toFixed(1),
            v.annual_cost_eur_mean.toFixed(1), v.comfort_breach_hours_per_year_mean.toFixed(1),
            v.p_elec_hvac_peak_kw_mean.toFixed(2),
            dynamic ? v.t_in_min_c.toFixed(1) : "—",
            dynamic ? v.t_in_max_c.toFixed(1) : "—",
            v.worst_heating_day_index ?? "—", v.worst_cooling_day_index ?? "—",
        ]);
        const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "laboratorio_termico_confronto.csv";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
    }

    // ── Chart builders ─────────────────────────────────────────────────────────

    /** X-axis month-label callback shared by the day-indexed charts. */
    function monthTickCallback(value) {
        const i = MONTH_START_DOY.indexOf(Number(this.getLabelForValue(value)));
        return i >= 0 ? MONTHS_SHORT[i] : "";
    }

    /** Daily HVAC energy per variant + outdoor temperature (secondary axis). */
    function buildDailyConfig(res) {
        if (!res) return { data: null, options: null };
        const datasets = res.variants.map((v, idx) => ({
            label: v.label,
            data: v.daily_hvac_kwh,
            borderColor: VARIANT_COLORS[idx % VARIANT_COLORS.length],
            backgroundColor: "transparent",
            borderWidth: 1.8,
            pointRadius: 0,
            fill: false,
            yAxisID: "y",
            order: 2,
        }));
        // Worst-day markers per variant (heating = filled circle, cooling = triangle).
        res.variants.forEach((v, idx) => {
            const color = VARIANT_COLORS[idx % VARIANT_COLORS.length];
            const pts = [];
            if (v.worst_heating_day_index != null) {
                pts.push({ x: v.worst_heating_day_index, y: v.daily_hvac_kwh[v.worst_heating_day_index], style: "circle" });
            }
            if (v.worst_cooling_day_index != null) {
                pts.push({ x: v.worst_cooling_day_index, y: v.daily_hvac_kwh[v.worst_cooling_day_index], style: "triangle" });
            }
            if (pts.length) {
                datasets.push({
                    label: `_worst_${idx}`,
                    data: pts.map((p) => ({ x: p.x, y: p.y })),
                    pointStyle: pts.map((p) => p.style),
                    showLine: false,
                    pointRadius: 6,
                    pointBackgroundColor: color,
                    pointBorderColor: "#fff",
                    pointBorderWidth: 1.5,
                    yAxisID: "y",
                    order: 1,
                });
            }
        });
        datasets.push({
            label: "Temperatura esterna (°C)",
            data: res.daily_outdoor_mean_c,
            borderColor: "rgba(108, 117, 125, 0.8)",
            borderDash: [5, 4],
            backgroundColor: "transparent",
            borderWidth: 1.4,
            pointRadius: 0,
            fill: false,
            yAxisID: "y1",
            order: 3,
        });
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: {
                    labels: { filter: (item) => !(item.text && item.text.startsWith("_worst_")) },
                },
                tooltip: {
                    callbacks: { title: (ctx) => `Giorno ${ctx[0].label}` },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: "Mese" },
                    ticks: { autoSkip: false, callback: monthTickCallback },
                    grid: { display: false },
                },
                y: {
                    position: "left",
                    title: { display: true, text: "Energia HVAC (kWh/giorno)" },
                    beginAtZero: true,
                },
                y1: {
                    position: "right",
                    title: { display: true, text: "T esterna (°C)" },
                    grid: { drawOnChartArea: false },
                },
            },
        };
        return { data: { labels: res.days, datasets }, options };
    }

    /** Annual cost per variant (bar), tooltip shows the p05–p95 band. */
    function buildCostConfig(res) {
        if (!res) return { data: null, options: null };
        const labels = res.variants.map((v) => v.label);
        const data = res.variants.map((v) => v.annual_cost_eur_mean);
        const colors = res.variants.map((_v, idx) => VARIANT_COLORS[idx % VARIANT_COLORS.length]);
        const datasets = [{
            label: "Costo HVAC (€/anno)",
            data,
            backgroundColor: colors.map((c) => c + "cc"),
            borderColor: colors,
            borderWidth: 1,
        }];
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = res.variants[ctx.dataIndex];
                            return [
                                `Costo medio: ${v.annual_cost_eur_mean.toFixed(0)} €/anno`,
                                `Banda p05–p95: ${v.annual_cost_eur_p05.toFixed(0)} – ${v.annual_cost_eur_p95.toFixed(0)} €`,
                            ];
                        },
                    },
                },
            },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "€/anno" } },
            },
        };
        return { data: { labels, datasets }, options };
    }

    /** Hourly preview: outdoor + indoor temperature + setpoints + electric draw. */
    function buildTimeseriesConfig(ts) {
        if (!ts) return { data: null, options: null };
        const datasets = [
            {
                label: "T esterna (°C)",
                data: ts.t_outdoor_c,
                borderColor: "rgba(108, 117, 125, 0.7)",
                backgroundColor: "transparent",
                borderWidth: 1.4,
                pointRadius: 0,
                yAxisID: "y",
                order: 3,
            },
        ];
        if (ts.t_indoor_c) {
            datasets.push({
                label: "T interna (°C)",
                data: ts.t_indoor_c,
                borderColor: "#3b82f6",
                backgroundColor: "transparent",
                borderWidth: 2,
                pointRadius: 0,
                yAxisID: "y",
                order: 1,
            });
        }
        datasets.push(
            {
                label: "Setpoint riscaldamento",
                data: ts.t_set_heating_c,
                borderColor: "rgba(16, 185, 129, 0.9)",
                borderDash: [4, 3],
                backgroundColor: "transparent",
                borderWidth: 1.2,
                pointRadius: 0,
                spanGaps: false,
                yAxisID: "y",
                order: 2,
            },
            {
                label: "Setpoint raffrescamento",
                data: ts.t_set_cooling_c,
                borderColor: "rgba(245, 158, 11, 0.9)",
                borderDash: [4, 3],
                backgroundColor: "transparent",
                borderWidth: 1.2,
                pointRadius: 0,
                spanGaps: false,
                yAxisID: "y",
                order: 2,
            },
            {
                label: "Potenza elettrica HVAC (kW)",
                data: ts.p_elec_hvac_kw,
                borderColor: "rgba(139, 92, 246, 0.55)",
                backgroundColor: "rgba(139, 92, 246, 0.15)",
                borderWidth: 1,
                pointRadius: 0,
                fill: true,
                yAxisID: "y1",
                order: 4,
            },
        );
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { display: true },
                tooltip: { callbacks: { title: (ctx) => `Ora ${ctx[0].label}` } },
            },
            scales: {
                x: {
                    title: { display: true, text: "Ora (dall'inizio della finestra)" },
                    ticks: {
                        autoSkip: false,
                        callback: function (value) {
                            const h = Number(this.getLabelForValue(value));
                            return h % 24 === 0 ? `g${h / 24}` : "";
                        },
                    },
                    grid: { display: false },
                },
                y: { position: "left", title: { display: true, text: "Temperatura (°C)" } },
                y1: {
                    position: "right",
                    title: { display: true, text: "P elettrica (kW)" },
                    beginAtZero: true,
                    grid: { drawOnChartArea: false },
                },
            },
        };
        return { data: { labels: ts.hours, datasets }, options };
    }

    $: dailyConfig = buildDailyConfig(result);
    $: costConfig = buildCostConfig(result);
    $: timeseriesConfig = buildTimeseriesConfig(timeseries);
    $: selectedClimate = climateProfiles.find((p) => p.id === selectedClimateId) ?? null;
</script>

<div class="container">
    <h1 class="page-title">Laboratorio termico</h1>
    <p class="page-subtitle">
        Confronta diversi livelli di isolamento dell'involucro e dimensiona la
        pompa di calore <em>prima</em> dell'analisi economica completa. Fissato
        un profilo climatico (creato nella sezione Database → Posizioni), il
        laboratorio lancia una piccola simulazione Monte Carlo e mostra consumi,
        costo, comfort e temperatura interna per ogni configurazione di casa.
    </p>

    <div class="lab-grid">
        <!-- ── Configuration form ───────────────────────────────────────── -->
        <div class="card config-card">
            <h2 class="section-title">Configurazione</h2>

            {#if climateError}
                <div class="info-box error">Errore caricamento profili: {climateError}</div>
            {:else if climateProfiles.length === 0}
                <div class="info-box">
                    Nessun profilo climatico disponibile. Creane uno dalla
                    sezione <a href="#/database">Database → Posizioni</a>
                    (import da Open-Meteo).
                </div>
            {:else}
                <div class="form-group">
                    <label class="label" for="climate">Profilo climatico</label>
                    <select id="climate" class="select" bind:value={selectedClimateId}>
                        {#each climateProfiles as p}
                            <option value={p.id}>{p.name} — {p.location_name}</option>
                        {/each}
                    </select>
                </div>
            {/if}

            <div class="form-group">
                <span class="label">Configurazioni di casa da confrontare</span>
                {#each PRESET_OPTIONS as opt}
                    <label class="checkbox-label">
                        <input type="checkbox" bind:checked={selectedPresets[opt.key]} />
                        {opt.label} <span class="text-meta">· {opt.w} W/°C/m²</span>
                    </label>
                {/each}
                <label class="checkbox-label">
                    <input type="checkbox" bind:checked={customUaEnabled} />
                    Custom UA
                </label>
                {#if customUaEnabled}
                    <input class="input" type="number" step="0.1" min="0"
                           bind:value={customUa} aria-label="UA custom (W/°C/m²)" />
                {/if}
            </div>

            <div class="form-group">
                <label class="label" for="floor">Superficie (m²)</label>
                <input id="floor" class="input" type="number" min="1" bind:value={floorArea} />
            </div>

            <div class="divider"></div>
            <h3 class="sub">Pompa di calore</h3>
            <div class="grid-mini">
                <div class="form-group">
                    <label class="label" for="coph">COP riscaldamento</label>
                    <input id="coph" class="input" type="number" step="0.1" min="0.1" bind:value={copHeating} />
                </div>
                <div class="form-group">
                    <label class="label" for="copc">COP raffrescamento</label>
                    <input id="copc" class="input" type="number" step="0.1" min="0.1" bind:value={copCooling} />
                </div>
                <div class="form-group">
                    <label class="label" for="pmax">P elettrica max (kW)</label>
                    <input id="pmax" class="input" type="number" step="0.5" min="0.1" bind:value={pElecMax} />
                </div>
            </div>

            <div class="divider"></div>
            <h3 class="sub">Setpoint comfort</h3>
            <div class="grid-mini">
                <div class="form-group">
                    <label class="label" for="th">Riscaldamento (°C)</label>
                    <input id="th" class="input" type="number" step="0.5" bind:value={tHeating} />
                </div>
                <div class="form-group">
                    <label class="label" for="tc">Raffrescamento (°C)</label>
                    <input id="tc" class="input" type="number" step="0.5" bind:value={tCooling} />
                </div>
            </div>
            <label class="checkbox-label">
                <input type="checkbox" bind:checked={nightSetbackEnabled} />
                Setback notturno riscaldamento (23:00–06:00)
            </label>
            {#if nightSetbackEnabled}
                <div class="form-group">
                    <label class="label" for="tn">Setpoint notturno (°C)</label>
                    <input id="tn" class="input" type="number" step="0.5" bind:value={tNight} />
                </div>
            {/if}

            <div class="divider"></div>
            <h3 class="sub">Presenza</h3>
            <label class="checkbox-label">
                <input type="checkbox" bind:checked={awayEnabled} />
                Fuori casa in alcune ore
            </label>
            {#if awayEnabled}
                <div class="grid-mini">
                    <div class="form-group">
                        <label class="label" for="as">Esco alle</label>
                        <input id="as" class="input" type="number" min="0" max="23" bind:value={awayStart} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="ae">Rientro alle</label>
                        <input id="ae" class="input" type="number" min="0" max="23" bind:value={awayEnd} />
                    </div>
                </div>
                <label class="checkbox-label">
                    <input type="checkbox" bind:checked={awaySetbackEnabled} />
                    Mantieni un setback quando sono fuori
                </label>
                {#if awaySetbackEnabled}
                    <div class="form-group">
                        <label class="label" for="ta">Setpoint assenza (°C)</label>
                        <input id="ta" class="input" type="number" step="0.5" bind:value={tAway} />
                    </div>
                {/if}
            {/if}

            <div class="divider"></div>
            <h3 class="sub">Prezzo energia</h3>
            <div class="grid-mini">
                <div class="form-group">
                    <label class="label" for="pmode">Modello</label>
                    <select id="pmode" class="select" bind:value={priceMode}>
                        {#each PRICE_MODES as m}
                            <option value={m.key}>{m.label}</option>
                        {/each}
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="pbase">{priceMode === "fixed" ? "Prezzo (€/kWh)" : "Prezzo base (€/kWh)"}</label>
                    <input id="pbase" class="input" type="number" step="0.01" min="0" bind:value={priceBase} />
                </div>
                {#if priceMode === "escalating"}
                    <div class="form-group">
                        <label class="label" for="pesc">Escalation (%/anno)</label>
                        <input id="pesc" class="input" type="number" step="0.5" bind:value={priceEscalationPct} />
                    </div>
                {:else if priceMode === "gbm"}
                    <div class="form-group">
                        <label class="label" for="pdrift">Drift (%/anno)</label>
                        <input id="pdrift" class="input" type="number" step="0.5" bind:value={priceDriftPct} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="pvol">Volatilità (%/anno)</label>
                        <input id="pvol" class="input" type="number" step="1" min="0" bind:value={priceVolPct} />
                    </div>
                {:else if priceMode === "mean_reverting"}
                    <div class="form-group">
                        <label class="label" for="pmrl">Prezzo equilibrio (€/kWh)</label>
                        <input id="pmrl" class="input" type="number" step="0.01" min="0" bind:value={priceMrLongTerm} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="pmrs">Velocità rientro (/anno)</label>
                        <input id="pmrs" class="input" type="number" step="0.05" min="0" bind:value={priceMrSpeed} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="pvol2">Volatilità (%/anno)</label>
                        <input id="pvol2" class="input" type="number" step="1" min="0" bind:value={priceVolPct} />
                    </div>
                {/if}
            </div>
            {#if priceMode !== "fixed"}
                <p class="text-meta">
                    Con un modello stocastico la banda di costo p05–p95 riflette
                    l'incertezza del prezzo su {nYears} anno/i, non solo l'energia.
                </p>
            {/if}

            <div class="divider"></div>
            <h3 class="sub">Simulazione</h3>
            <div class="grid-mini">
                <div class="form-group">
                    <label class="label" for="np">Path MC</label>
                    <input id="np" class="input" type="number" min="1" max="200" bind:value={nPaths} />
                </div>
                <div class="form-group">
                    <label class="label" for="ny">Anni</label>
                    <input id="ny" class="input" type="number" min="1" max="20" bind:value={nYears} />
                </div>
                <div class="form-group">
                    <label class="label" for="seed">Seed</label>
                    <input id="seed" class="input" type="number" bind:value={seed} />
                </div>
            </div>
            <label class="checkbox-label" title="Integra l'ODE RC per la temperatura interna reale (mostra il drift quando la pompa è sottodimensionata)">
                <input type="checkbox" bind:checked={dynamic} />
                Modello dinamico RC (temperatura interna)
            </label>

            <button class="btn btn-primary run-btn" on:click={runCompare} disabled={running}>
                {running ? "Simulazione in corso…" : "Confronta configurazioni"}
            </button>
            {#if runError}
                <div class="info-box error">{runError}</div>
            {/if}
        </div>

        <!-- ── Results ───────────────────────────────────────────────────── -->
        <div class="results-col">
            {#if !result}
                <div class="card placeholder">
                    <p>
                        {#if selectedClimate}
                            Pronto a confrontare per <strong>{selectedClimate.location_name}</strong>.
                            Imposta i parametri e premi <em>Confronta configurazioni</em>.
                        {:else}
                            Seleziona un profilo climatico per iniziare.
                        {/if}
                    </p>
                </div>
            {:else}
                <!-- KPI table -->
                <div class="card">
                    <div class="header-actions">
                        <h2 class="section-title" style="border:none;margin:0;">Confronto KPI</h2>
                        <div class="export-actions">
                            <button class="btn btn-outline btn-sm" on:click={exportCsv}>CSV</button>
                            <button class="btn btn-outline btn-sm" on:click={() => doExport("xlsx")} disabled={exporting}>
                                {exporting === "xlsx" ? "…" : "Excel"}
                            </button>
                            <button class="btn btn-outline btn-sm" on:click={() => doExport("pdf")} disabled={exporting}>
                                {exporting === "pdf" ? "…" : "PDF"}
                            </button>
                        </div>
                    </div>
                    {#if exportError}
                        <div class="info-box error">Export non riuscito: {exportError}</div>
                    {/if}
                    <div class="table-wrap">
                        <table class="kpi-table">
                            <thead>
                                <tr>
                                    <th>Configurazione</th>
                                    <th>UA (kW/°C)</th>
                                    <th>kWh/anno</th>
                                    <th>Risc./Raffr. (kWh)</th>
                                    <th>Costo (€/anno)</th>
                                    <th>Breach (h/anno)</th>
                                    <th>Picco (kW)</th>
                                    {#if dynamic}<th>T int. min/max (°C)</th>{/if}
                                </tr>
                            </thead>
                            <tbody>
                                {#each result.variants as v, idx}
                                    <tr>
                                        <td>
                                            <span class="swatch" style="background:{VARIANT_COLORS[idx % VARIANT_COLORS.length]}"></span>
                                            {v.label}
                                        </td>
                                        <td>{v.ua_kw_per_c.toFixed(3)}</td>
                                        <td>
                                            {v.hvac_kwh_annual_mean.toFixed(0)}
                                            <span class="text-meta">({v.hvac_kwh_annual_p05.toFixed(0)}–{v.hvac_kwh_annual_p95.toFixed(0)})</span>
                                        </td>
                                        <td>{v.heating_kwh_annual_mean.toFixed(0)} / {v.cooling_kwh_annual_mean.toFixed(0)}</td>
                                        <td>
                                            {v.annual_cost_eur_mean.toFixed(0)}
                                            <span class="text-meta">({v.annual_cost_eur_p05.toFixed(0)}–{v.annual_cost_eur_p95.toFixed(0)})</span>
                                        </td>
                                        <td class:warn={v.comfort_breach_hours_per_year_mean > 0}>
                                            {v.comfort_breach_hours_per_year_mean.toFixed(0)}
                                        </td>
                                        <td>{v.p_elec_hvac_peak_kw_mean.toFixed(2)}</td>
                                        {#if dynamic}
                                            <td>{v.t_in_min_c.toFixed(1)} / {v.t_in_max_c.toFixed(1)}</td>
                                        {/if}
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>
                    <p class="text-meta">
                        {result.n_paths} path × {result.n_years} anno/i. "Breach" = ore/anno
                        in cui la pompa è satura e non tiene il setpoint (indice di
                        sottodimensionamento).
                    </p>
                </div>

                <!-- Daily energy + outdoor temp -->
                <div class="card">
                    <h2 class="section-title">Consumi giornalieri (anno tipico)</h2>
                    <div class="chart-wrap">
                        <ResultsChart
                            type="line"
                            data={dailyConfig.data}
                            options={dailyConfig.options}
                            downloadFilename="lab_termico_consumi_giornalieri"
                        />
                    </div>
                    <p class="text-meta">
                        ● = giorno più gravoso in riscaldamento · ▲ = giorno più gravoso
                        in raffrescamento, per ciascuna configurazione.
                    </p>
                </div>

                <!-- Cost bar -->
                <div class="card">
                    <h2 class="section-title">Costo annuo per configurazione</h2>
                    <div class="chart-wrap small">
                        <ResultsChart
                            type="bar"
                            data={costConfig.data}
                            options={costConfig.options}
                            zoomable={false}
                            downloadFilename="lab_termico_costo"
                        />
                    </div>
                </div>

                <!-- Hourly timeseries preview -->
                <div class="card">
                    <div class="header-actions">
                        <h2 class="section-title" style="border:none;margin:0;">
                            Anteprima oraria ({tsDays} giorni)
                        </h2>
                        <div class="ts-controls">
                            <select class="select select-sm" bind:value={tsVariantIndex} on:change={runTimeseries}>
                                {#each result.variants as v, idx}
                                    <option value={idx}>{v.label}</option>
                                {/each}
                            </select>
                            <select class="select select-sm" bind:value={tsStartDay} on:change={runTimeseries}>
                                {#each SEASON_OPTIONS as s}
                                    <option value={s.day}>{s.label}</option>
                                {/each}
                            </select>
                        </div>
                    </div>
                    {#if !dynamic}
                        <div class="info-box">
                            La temperatura interna è disponibile solo con il
                            <strong>modello dinamico RC</strong> attivo. In steady-state
                            la casa è per ipotesi sempre al setpoint.
                        </div>
                    {/if}
                    {#if tsError}
                        <div class="info-box error">{tsError}</div>
                    {:else if tsRunning}
                        <div class="info-box">Calcolo della serie oraria…</div>
                    {:else if timeseriesConfig.data}
                        <div class="chart-wrap">
                            <ResultsChart
                                type="line"
                                data={timeseriesConfig.data}
                                options={timeseriesConfig.options}
                                downloadFilename="lab_termico_serie_oraria"
                            />
                        </div>
                    {/if}
                </div>
            {/if}
        </div>
    </div>
</div>

<style>
    .page-subtitle {
        color: var(--color-text-secondary);
        max-width: 75ch;
        margin-bottom: 1.5rem;
    }
    .lab-grid {
        display: grid;
        grid-template-columns: 340px 1fr;
        gap: 1.5rem;
        align-items: start;
    }
    .config-card {
        position: sticky;
        top: 1rem;
        height: auto;
    }
    @media (max-width: 900px) {
        .lab-grid { grid-template-columns: 1fr; }
        /* In the single-column layout the config card must not stay pinned,
           otherwise it overlaps the results as they scroll under it. */
        .config-card { position: static; }
    }
    .results-col {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        min-width: 0;
    }
    .sub {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: var(--color-text-secondary);
    }
    .grid-mini {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem 0.75rem;
    }
    .run-btn {
        width: 100%;
        margin-top: 1rem;
    }
    .chart-wrap {
        position: relative;
        width: 100%;
        height: 340px;
    }
    .chart-wrap.small { height: 260px; }
    .placeholder {
        color: var(--color-text-secondary);
        text-align: center;
        padding: 3rem 1rem;
    }
    .table-wrap { overflow-x: auto; }
    .kpi-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .kpi-table th, .kpi-table td {
        text-align: left;
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid var(--color-border);
        white-space: nowrap;
    }
    .kpi-table th {
        color: var(--color-text-secondary);
        font-weight: 600;
    }
    .kpi-table td.warn { color: var(--color-danger); font-weight: 600; }
    .swatch {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 2px;
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    .export-actions { display: flex; gap: 0.4rem; }
    .ts-controls { display: flex; gap: 0.5rem; }
    .select-sm { width: auto; padding: 0.3rem 0.5rem; font-size: 0.85rem; }
    .info-box {
        padding: 0.6rem 0.8rem;
        background: var(--color-bg-secondary);
        border-radius: var(--radius-sm);
        border: 1px solid var(--color-border);
        margin-top: 0.75rem;
    }
    .info-box.error {
        border-color: var(--color-danger);
        background: #fef2f2;
        color: #842029;
    }
    .info-box a { color: var(--color-accent); text-decoration: underline; }
</style>
