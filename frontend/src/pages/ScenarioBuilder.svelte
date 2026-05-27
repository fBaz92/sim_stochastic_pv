<script>
    /**
     * ScenarioBuilder — Fase 6: wizard a 6 passi
     *
     * Guida l'utente attraverso la configurazione completa di uno scenario
     * PV+batteria in un flusso sequenziale:
     *
     *   1. Luogo       — selezione profilo solare, preview meteo read-only
     *   2. Impianto    — kWp, tilt/azimuth, inverter, batteria, degradazione
     *   3. Carico      — tipologia profilo (ARERA / mensile / home-away / weekly)
     *   4. Mercato     — modello prezzo (deterministico / GBM / mean-reverting)
     *   5. Investimento — capitale investito, orizzonte, campioni MC
     *   6. Riepilogo   — sommario + salva configurazione + esegui analisi
     *
     * Al termine dell'analisi viene scritto ``pendingRunId`` nello store Svelte
     * e l'utente viene reindirizzato alla Dashboard che auto-seleziona il run
     * appena creato.
     */
    import { onMount } from "svelte";
    import { api } from "../api";
    import { pendingRunId } from "../lib/stores";
    import MonthInput from "../components/forms/MonthInput.svelte";
    import MonthlyProfileEditor from "../components/forms/MonthlyProfileEditor.svelte";
    import WeeklyPatternEditor from "../components/forms/WeeklyPatternEditor.svelte";

    // ── Step navigation ────────────────────────────────────────────────────
    const STEPS = [
        "Luogo",
        "Impianto",
        "Carico",
        "Mercato",
        "Investimento",
        "Riepilogo",
    ];
    let currentStep = 0;

    function goTo(i) {
        if (i >= 0 && i < STEPS.length) currentStep = i;
    }
    function next() { if (currentStep < STEPS.length - 1) currentStep++; }
    function prev() { if (currentStep > 0) currentStep--; }

    // ── Remote data ────────────────────────────────────────────────────────
    let solarProfiles = [];
    let inverters = [];
    let batteries = [];
    let loadProfiles = [];
    let savedScenarios = [];

    // ── Step 1 — Luogo ─────────────────────────────────────────────────────
    let selectedSolarProfileId = "";

    /** The full SolarProfileResponse object for the selected location. */
    $: selectedSolarProfile = solarProfiles.find(
        (p) => String(p.id) === selectedSolarProfileId,
    ) ?? null;

    const MONTHS_SHORT = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"];

    // ── Step 2 — Impianto ──────────────────────────────────────────────────
    // When a solar profile is selected its optimal orientation pre-fills
    // the tilt/azimuth fields.  The user can override them.
    let overrideTilt = false;
    let overrideAzimuth = false;

    let pvKwp = 3.0;
    let degradationPerYear = 0.007;
    let manualTilt = 35.0;
    let manualAzimuth = 180.0;

    // Inverter selection
    let selectedInverterIndex = -1;
    let selectedInverterId = null;
    let inverterPAcMaxKw = 3.0;

    // Battery selection
    let selectedBatteryIndex = -1;
    let selectedBatteryId = null;
    let batteryCapacityKwh = 5.0;
    let batteryCyclesLife = 5000;
    let nBatteries = 0;

    function onInverterChange() {
        if (selectedInverterIndex >= 0) {
            const inv = inverters[selectedInverterIndex];
            selectedInverterId = inv.id;
            inverterPAcMaxKw = inv.p_ac_max_kw ?? inv.nominal_power_kw ?? inverterPAcMaxKw;
        } else {
            selectedInverterId = null;
        }
    }

    function onBatteryChange() {
        if (selectedBatteryIndex >= 0) {
            const bat = batteries[selectedBatteryIndex];
            selectedBatteryId = bat.id;
            batteryCapacityKwh = bat.capacity_kwh ?? batteryCapacityKwh;
            batteryCyclesLife = bat.specs?.cycles_life ?? batteryCyclesLife;
            if (nBatteries === 0) nBatteries = 1;
        } else {
            selectedBatteryId = null;
        }
    }

    // Effective tilt/azimuth sent to backend (null = use profile optimal)
    $: effectiveTilt = overrideTilt
        ? manualTilt
        : selectedSolarProfile?.optimal_tilt_degrees ?? null;
    $: effectiveAzimuth = overrideAzimuth
        ? manualAzimuth
        : selectedSolarProfile?.optimal_azimuth_degrees ?? null;

    // ── Step 3 — Carico ────────────────────────────────────────────────────
    // Source: "db" (saved profile) or "inline" (custom in-wizard)
    let loadSource = "db"; // "db" | "inline"
    let selectedLoadProfileId = "";

    // Days at home (always editable, even when profile is from DB)
    let minDaysHome = Array(12).fill(25);
    let maxDaysHome = Array(12).fill(28);

    // Inline custom profile state
    let homeProfileType = "arera"; // "arera" | "custom" | "custom_24h" | "weekly"
    let awayProfileType = "arera";

    let homeProfilesW = Array(12).fill(300);
    let homeProfiles24h = Array.from({ length: 12 }, () => Array(24).fill(200));
    let awayProfilesW = Array(12).fill(100);
    let awayProfiles24h = Array.from({ length: 12 }, () => Array(24).fill(100));

    // Weekly pattern (standalone, for when homeProfileType === "weekly")
    let weeklyMonthlyW = Array(12).fill(300);
    let weeklyPattern = Array.from({ length: 7 }, () => Array(24).fill(100));

    /** Human-readable summary of the selected DB load profile. */
    function describeLoadProfile(item) {
        if (!item) return "";
        if (item.profile_type === "home_away") {
            const hk = item.data?.home?.type === "arera" ? "ARERA"
                : item.data?.home?.monthly_24h_w ? "24h" : "media mensile";
            const ak = item.data?.away?.type === "arera" ? "ARERA"
                : item.data?.away?.monthly_24h_w ? "24h" : "media mensile";
            return `${item.name} — casa: ${hk}, via: ${ak}`;
        }
        return `${item.name} (${item.profile_type})`;
    }

    // ── Step 4 — Mercato ───────────────────────────────────────────────────
    let priceModelType = "escalating"; // "escalating" | "gbm" | "mean_reverting"

    // Shared
    let basePriceEur = 0.25;

    // Escalating (deterministico)
    let annualEscalation = 0.025;
    let useStochasticEscalation = true;
    let escalationP05 = -0.05;
    let escalationP95 = 0.05;

    // GBM
    let gbmDrift = 0.025;
    let gbmVolatility = 0.08;

    // Mean-reverting (Ornstein–Uhlenbeck)
    let mrMeanRevSpeed = 0.5;
    let mrLongTermPrice = 0.28;
    let mrVolatility = 0.06;

    // ── Step 5 — Investimento ──────────────────────────────────────────────
    let investmentEur = 6000;
    let nYears = 20;
    let nMc = 200;
    let scenarioName = "Mio Scenario";

    // ── Step 6 — Riepilogo & Esegui ────────────────────────────────────────
    let loading = false;
    let message = "";
    let showSaveModal = false;
    let savedScenarioId = null; // ID of the last saved configuration

    // ── Saved scenarios (load / restore) ──────────────────────────────────
    let selectedSavedScenarioId = "";

    // ── Initialise remote data ─────────────────────────────────────────────
    onMount(async () => {
        try {
            const [sp, inv, bat, lp, configs] = await Promise.all([
                api.listSolarProfiles(),
                api.listInverters(),
                api.listBatteries(),
                api.listLoadProfiles(),
                api.listConfigurations("scenario"),
            ]);
            solarProfiles = sp;
            inverters = inv;
            batteries = bat;
            loadProfiles = lp;
            savedScenarios = configs;
        } catch (e) {
            console.error("Failed to load initial data:", e);
        }
    });

    // ── Payload assembly ───────────────────────────────────────────────────

    /**
     * Build the ``price`` block of the scenario payload from the current
     * Step 4 wizard state, routing to the correct model_type.
     */
    function buildPriceBlock() {
        if (priceModelType === "gbm") {
            return {
                model_type: "gbm",
                base_price_eur_per_kwh: basePriceEur,
                drift_annual: gbmDrift,
                volatility_annual: gbmVolatility,
            };
        }
        if (priceModelType === "mean_reverting") {
            return {
                model_type: "mean_reverting",
                base_price_eur_per_kwh: basePriceEur,
                mean_reversion_speed_annual: mrMeanRevSpeed,
                long_term_price_eur_per_kwh: mrLongTermPrice,
                volatility_annual: mrVolatility,
            };
        }
        // Default: escalating (deterministic or stochastic)
        const block = {
            base_price_eur_per_kwh: basePriceEur,
            annual_escalation: annualEscalation,
            use_stochastic_escalation: useStochasticEscalation,
        };
        if (useStochasticEscalation) {
            block.escalation_variation_percentiles = [escalationP05, escalationP95];
        }
        return block;
    }

    /**
     * Build the ``load_profile`` block (or top-level ``load_profile_id``) from
     * the current Step 3 wizard state.  Mutates and returns the provided
     * scenario clone for convenience (same pattern as Phase 8 logic in the
     * legacy ScenarioBuilder).
     */
    function applyLoadProfile(scenarioClone) {
        if (loadSource === "db" && selectedLoadProfileId) {
            scenarioClone.load_profile_id = Number(selectedLoadProfileId);
            scenarioClone.min_days_home = [...minDaysHome];
            scenarioClone.max_days_home = [...maxDaysHome];
            delete scenarioClone.load_profile;
            return scenarioClone;
        }

        // Inline custom: build load_profile block
        if (homeProfileType === "weekly") {
            // Standalone weekly profile (Phase 5 «kind: weekly»)
            scenarioClone.load_profile = {
                kind: "weekly",
                type: "weekly",
                monthly_w: [...weeklyMonthlyW],
                weekly_pattern_w: weeklyPattern.map((row) => [...row]),
                min_days_home: [...minDaysHome],
                max_days_home: [...maxDaysHome],
            };
            return scenarioClone;
        }

        // Home/Away inline path
        let resolvedHomeType = homeProfileType;
        let resolvedHomeW = null;

        if (homeProfileType === "custom_24h") {
            resolvedHomeType = "custom";
            resolvedHomeW = homeProfiles24h.map((row) => [...row]);
        } else if (homeProfileType === "custom") {
            resolvedHomeW = [...homeProfilesW];
        }

        let resolvedAwayType = awayProfileType;
        let resolvedAwayW = null;
        if (awayProfileType === "custom_24h") {
            resolvedAwayType = "custom";
            resolvedAwayW = awayProfiles24h.map((row) => [...row]);
        } else if (awayProfileType === "custom") {
            resolvedAwayW = [...awayProfilesW];
        }

        scenarioClone.load_profile = {
            home_profile_type: resolvedHomeType,
            away_profile: resolvedAwayType,
            min_days_home: [...minDaysHome],
            max_days_home: [...maxDaysHome],
        };
        if (resolvedHomeW) scenarioClone.load_profile.home_profiles_w = resolvedHomeW;
        if (resolvedAwayW) scenarioClone.load_profile.away_profiles_w = resolvedAwayW;

        return scenarioClone;
    }

    /**
     * Assemble the full scenario + analysis payload from all wizard steps.
     */
    function buildPayload() {
        const solar = {
            pv_kwp: pvKwp,
            degradation_per_year: degradationPerYear,
        };
        if (selectedSolarProfileId) {
            solar.solar_profile_id = Number(selectedSolarProfileId);
        }
        if (overrideTilt) solar.panel_tilt_degrees = manualTilt;
        if (overrideAzimuth) solar.panel_azimuth_degrees = manualAzimuth;

        const energy = {
            n_years: nYears,
            battery_specs: {
                capacity_kwh: batteryCapacityKwh,
                cycles_life: batteryCyclesLife,
            },
            n_batteries: nBatteries,
            inverter_p_ac_max_kw: inverterPAcMaxKw,
        };

        let scenarioClone = {
            scenario_name: scenarioName,
            solar,
            energy,
            price: buildPriceBlock(),
            economic: {
                investment_eur: investmentEur,
                n_mc: nMc,
            },
        };

        if (selectedInverterId) scenarioClone.inverter_id = selectedInverterId;
        if (selectedBatteryId) scenarioClone.battery_id = selectedBatteryId;

        applyLoadProfile(scenarioClone);

        return {
            n_mc: nMc,
            scenario: scenarioClone,
        };
    }

    // ── Actions ────────────────────────────────────────────────────────────

    async function handleRun() {
        loading = true;
        message = "";
        try {
            const payload = buildPayload();
            const res = await api.triggerAnalysis(payload);
            // Phase 6: store the run ID and redirect to Dashboard so that the
            // auto-select logic picks it up on mount.
            if (res.run_id != null) {
                pendingRunId.set(res.run_id);
            }
            window.location.hash = "/";
        } catch (e) {
            console.error(e);
            message = "Errore durante l'analisi: " + e.message;
        } finally {
            loading = false;
        }
    }

    async function handleSave() {
        loading = true;
        message = "";
        try {
            const scenarioClone = JSON.parse(JSON.stringify(buildPayload().scenario));
            const payload = {
                name: scenarioName,
                config_type: "scenario",
                data: scenarioClone,
            };
            const saved = await api.createConfiguration(payload);
            savedScenarioId = saved.id;
            savedScenarios = await api.listConfigurations("scenario");
            message = `Scenario "${scenarioName}" salvato con successo.`;
            showSaveModal = false;
        } catch (e) {
            console.error(e);
            message = "Errore durante il salvataggio: " + e.message;
        } finally {
            loading = false;
        }
    }

    async function handleLoadScenario() {
        if (!selectedSavedScenarioId) return;
        loading = true;
        message = "";
        try {
            const targetId = Number(selectedSavedScenarioId);
            const saved = savedScenarios.find((s) => s.id === targetId);
            if (!saved) throw new Error("Scenario non trovato");

            // Restore wizard state from the saved configuration data
            const d = saved.data?.scenario ?? saved.data;
            if (!d) throw new Error("Dati scenario mancanti");

            // Step 1 — Luogo
            if (d.solar?.solar_profile_id) {
                selectedSolarProfileId = String(d.solar.solar_profile_id);
            }

            // Step 2 — Impianto
            pvKwp = d.solar?.pv_kwp ?? pvKwp;
            degradationPerYear = d.solar?.degradation_per_year ?? degradationPerYear;
            if (d.solar?.panel_tilt_degrees != null) {
                overrideTilt = true;
                manualTilt = d.solar.panel_tilt_degrees;
            }
            if (d.solar?.panel_azimuth_degrees != null) {
                overrideAzimuth = true;
                manualAzimuth = d.solar.panel_azimuth_degrees;
            }
            inverterPAcMaxKw = d.energy?.inverter_p_ac_max_kw ?? inverterPAcMaxKw;
            batteryCapacityKwh = d.energy?.battery_specs?.capacity_kwh ?? batteryCapacityKwh;
            batteryCyclesLife = d.energy?.battery_specs?.cycles_life ?? batteryCyclesLife;
            nBatteries = d.energy?.n_batteries ?? nBatteries;

            // Step 3 — Carico
            if (d.load_profile_id) {
                loadSource = "db";
                selectedLoadProfileId = String(d.load_profile_id);
            } else if (d.load_profile) {
                loadSource = "inline";
                homeProfileType = d.load_profile.home_profile_type ?? "arera";
                awayProfileType = d.load_profile.away_profile ?? "arera";
                if (d.load_profile.home_profiles_w) homeProfilesW = d.load_profile.home_profiles_w;
                if (d.load_profile.away_profiles_w) awayProfilesW = d.load_profile.away_profiles_w;
            }
            if (d.min_days_home) minDaysHome = d.min_days_home;
            if (d.max_days_home) maxDaysHome = d.max_days_home;
            if (d.load_profile?.min_days_home) minDaysHome = d.load_profile.min_days_home;
            if (d.load_profile?.max_days_home) maxDaysHome = d.load_profile.max_days_home;

            // Step 4 — Mercato
            const p = d.price ?? {};
            priceModelType = p.model_type ?? "escalating";
            basePriceEur = p.base_price_eur_per_kwh ?? basePriceEur;
            if (priceModelType === "gbm") {
                gbmDrift = p.drift_annual ?? gbmDrift;
                gbmVolatility = p.volatility_annual ?? gbmVolatility;
            } else if (priceModelType === "mean_reverting") {
                mrMeanRevSpeed = p.mean_reversion_speed_annual ?? mrMeanRevSpeed;
                mrLongTermPrice = p.long_term_price_eur_per_kwh ?? mrLongTermPrice;
                mrVolatility = p.volatility_annual ?? mrVolatility;
            } else {
                annualEscalation = p.annual_escalation ?? annualEscalation;
                useStochasticEscalation = p.use_stochastic_escalation ?? useStochasticEscalation;
                if (p.escalation_variation_percentiles) {
                    escalationP05 = p.escalation_variation_percentiles[0] ?? escalationP05;
                    escalationP95 = p.escalation_variation_percentiles[1] ?? escalationP95;
                }
            }

            // Step 5 — Investimento
            investmentEur = d.economic?.investment_eur ?? investmentEur;
            nMc = d.economic?.n_mc ?? nMc;
            nYears = d.energy?.n_years ?? nYears;

            scenarioName = saved.name ?? scenarioName;
            message = `Scenario "${saved.name}" caricato.`;
        } catch (e) {
            console.error(e);
            message = "Errore nel caricamento: " + e.message;
        } finally {
            loading = false;
        }
    }

    // ── Summary helpers ────────────────────────────────────────────────────

    $: summaryRows = [
        ["Luogo", selectedSolarProfile?.location_name ?? "Manuale (nessun profilo scelto)"],
        ["Taglia PV", `${pvKwp} kWp`],
        ["Tilt / Azimuth",
            overrideTilt || overrideAzimuth
                ? `${overrideTilt ? manualTilt : (selectedSolarProfile?.optimal_tilt_degrees ?? "—")}° / ${overrideAzimuth ? manualAzimuth : (selectedSolarProfile?.optimal_azimuth_degrees ?? "—")}°`
                : "Ottimale dal profilo"],
        ["Degradazione pannelli", `${(degradationPerYear * 100).toFixed(1)} %/anno`],
        ["Inverter",
            selectedInverterIndex >= 0
                ? inverters[selectedInverterIndex]?.name ?? "—"
                : `Manuale: ${inverterPAcMaxKw} kW AC`],
        ["Batteria",
            nBatteries === 0
                ? "Nessuna"
                : selectedBatteryIndex >= 0
                    ? `${nBatteries}× ${batteries[selectedBatteryIndex]?.name ?? "—"} (${batteryCapacityKwh} kWh, ${batteryCyclesLife} cicli)`
                    : `${nBatteries}× manuale (${batteryCapacityKwh} kWh, ${batteryCyclesLife} cicli)`],
        ["Profilo di carico",
            loadSource === "db" && selectedLoadProfileId
                ? describeLoadProfile(loadProfiles.find((p) => String(p.id) === selectedLoadProfileId))
                : `Inline — ${homeProfileType}`],
        ["Modello prezzo", priceModelType === "gbm"
                ? `GBM drift=${(gbmDrift * 100).toFixed(1)}% vol=${(gbmVolatility * 100).toFixed(1)}%`
                : priceModelType === "mean_reverting"
                    ? `Mean-rev kappa=${mrMeanRevSpeed} lt=${mrLongTermPrice.toFixed(3)} €/kWh`
                    : `Deterministico ${(annualEscalation * 100).toFixed(1)}%/anno`],
        ["Prezzo base", `€ ${basePriceEur.toFixed(3)}/kWh`],
        ["Investimento", `€ ${investmentEur.toLocaleString("it-IT")}`],
        ["Orizzonte", `${nYears} anni`],
        ["Campioni MC", nMc.toLocaleString("it-IT")],
    ];
</script>

<!-- ══════════════════════════════════════════════════════════════════════ -->
<!-- Header + Load saved                                                   -->
<!-- ══════════════════════════════════════════════════════════════════════ -->
<div class="container">
    <div class="header">
        <h1 class="page-title">Nuovo Scenario</h1>
        <p class="page-subtitle">
            Configura <strong>un</strong> impianto PV+batteria in pochi passi e
            avvia l'analisi Monte Carlo. Per esplorare più alternative vai su
            <a href="#/campaign">Campagna</a>.
        </p>
        <!-- Load saved scenario (top-bar shortcut) -->
        <div class="header-actions">
            <select
                class="select sm"
                bind:value={selectedSavedScenarioId}
                style="min-width: 220px;"
            >
                <option value="">Carica scenario salvato…</option>
                {#each savedScenarios as s}
                    <option value={String(s.id)}>{s.name}</option>
                {/each}
            </select>
            <button
                class="btn btn-outline btn-sm"
                on:click={handleLoadScenario}
                disabled={!selectedSavedScenarioId || loading}
            >Carica</button>
        </div>
        {#if message}
            <div class={`badge ${message.toLowerCase().includes("errore") ? "error" : "success"}`}>
                {message}
            </div>
        {/if}
    </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- Step progress bar                                                  -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    <div class="stepper" role="tablist">
        {#each STEPS as label, i}
            <button
                role="tab"
                class="step-btn"
                class:active={i === currentStep}
                class:done={i < currentStep}
                on:click={() => goTo(i)}
                aria-selected={i === currentStep}
            >
                <span class="step-num">{i + 1}</span>
                <span class="step-label">{label}</span>
            </button>
            {#if i < STEPS.length - 1}
                <span class="step-sep" aria-hidden="true">›</span>
            {/if}
        {/each}
    </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- STEP 1 — Luogo di installazione                                   -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    {#if currentStep === 0}
        <div class="step-content card">
            <h2 class="step-title">1. Luogo di installazione</h2>
            <p class="step-desc">
                Scegli il profilo solare del sito di installazione. Il simulatore
                userà i dati meteo storici di quel luogo per generare la
                produzione PV Monte Carlo.
            </p>

            <div class="form-group">
                <label class="label" for="solar-profile-select">
                    Località / Profilo solare
                </label>
                <select
                    id="solar-profile-select"
                    class="select"
                    bind:value={selectedSolarProfileId}
                >
                    <option value="">— Selezione manuale (nessun profilo DB) —</option>
                    {#each solarProfiles as sp}
                        <option value={String(sp.id)}>{sp.location_name}</option>
                    {/each}
                </select>
                <p class="hint">
                    I profili precaricati provengono da PVGIS. Se il tuo sito non è
                    in lista puoi procedere senza selezionarlo e inserire i parametri
                    manualmente nello step 2.
                </p>
            </div>

            {#if selectedSolarProfile}
                <!-- Read-only weather preview -->
                <div class="preview-card card subtle">
                    <div class="preview-header">
                        <strong>{selectedSolarProfile.location_name}</strong>
                        <span class="text-meta">
                            {selectedSolarProfile.latitude.toFixed(2)}° N,
                            {selectedSolarProfile.longitude.toFixed(2)}° E
                            {#if selectedSolarProfile.elevation_m != null}
                                · {selectedSolarProfile.elevation_m} m
                            {/if}
                            · Fonte: {selectedSolarProfile.source ?? "—"}
                        </span>
                    </div>
                    <div class="preview-tilt">
                        <span class="text-meta">
                            Tilt ottimale: <strong>{selectedSolarProfile.optimal_tilt_degrees}°</strong>
                            · Azimuth ottimale: <strong>{selectedSolarProfile.optimal_azimuth_degrees}°</strong>
                            (180° = sud)
                        </span>
                    </div>
                    <div class="month-grid-header">
                        <span class="row-label"></span>
                        {#each MONTHS_SHORT as m}
                            <span class="month-col">{m}</span>
                        {/each}
                    </div>
                    <div class="month-grid-row">
                        <span class="row-label" title="kWh/kWp/giorno — produzione media giornaliera">☀️ kWh/kWp/g</span>
                        {#each selectedSolarProfile.avg_daily_kwh_per_kwp as v}
                            <span class="month-col data-cell">{v.toFixed(2)}</span>
                        {/each}
                    </div>
                    <div class="month-grid-row">
                        <span class="row-label" title="Probabilità marginale di giorno soleggiato">p_sunny</span>
                        {#each selectedSolarProfile.p_sunny as v}
                            <span class="month-col data-cell">{(v * 100).toFixed(0)}%</span>
                        {/each}
                    </div>
                    {#if selectedSolarProfile.weather_persistence}
                        <div class="month-grid-row">
                            <span class="row-label" title="Persistenza meteo (Markov): 0 = iid, 1 = perfetta">persistence</span>
                            {#each selectedSolarProfile.weather_persistence as v}
                                <span class="month-col data-cell">{v.toFixed(2)}</span>
                            {/each}
                        </div>
                    {/if}
                </div>
            {:else}
                <div class="info-box">
                    <p>
                        Nessun profilo selezionato. Lo step 2 ti permetterà di
                        inserire tilt e azimuth manualmente.
                    </p>
                </div>
            {/if}
        </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- STEP 2 — Impianto                                                  -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    {:else if currentStep === 1}
        <div class="step-content card">
            <h2 class="step-title">2. Impianto</h2>
            <p class="step-desc">
                Definisci la taglia del campo PV, l'orientamento dei pannelli,
                l'inverter e il sistema di accumulo.
            </p>

            <!-- PV system -->
            <div class="section-subtitle">Campo fotovoltaico</div>
            <div class="form-row">
                <div class="form-group">
                    <label class="label" for="pv-kwp">
                        Taglia impianto (kWp)
                        <span class="tooltip" title="Potenza di picco totale del campo PV in condizioni STC.">ⓘ</span>
                    </label>
                    <input id="pv-kwp" class="input" type="number" step="0.1" min="0.5" bind:value={pvKwp} />
                </div>
                <div class="form-group">
                    <label class="label" for="degradation">
                        Degrado annuo pannelli
                        <span class="tooltip" title="Percentuale di calo della potenza ogni anno. Tipicamente 0.5–1 %/anno.">ⓘ</span>
                    </label>
                    <input id="degradation" class="input" type="number" step="0.001" min="0" max="0.05" bind:value={degradationPerYear} />
                    <p class="hint">{(degradationPerYear * 100).toFixed(2)} %/anno</p>
                </div>
            </div>

            <!-- Tilt / Azimuth -->
            <div class="form-row">
                <div class="form-group">
                    <label class="label">
                        <input type="checkbox" bind:checked={overrideTilt} />
                        Override tilt
                        <span class="tooltip" title="Lascia deselezionato per usare il valore ottimale del profilo solare.">ⓘ</span>
                    </label>
                    {#if overrideTilt}
                        <input class="input" type="number" step="1" min="0" max="90" bind:value={manualTilt} />
                        <p class="hint">Inclinazione pannelli in gradi (0° = orizzontale, 90° = verticale)</p>
                    {:else}
                        <p class="hint text-meta">
                            Ottimale dal profilo:
                            {selectedSolarProfile?.optimal_tilt_degrees ?? "—"}°
                        </p>
                    {/if}
                </div>
                <div class="form-group">
                    <label class="label">
                        <input type="checkbox" bind:checked={overrideAzimuth} />
                        Override azimuth
                        <span class="tooltip" title="Lascia deselezionato per usare il valore ottimale (tipicamente 180° = sud).">ⓘ</span>
                    </label>
                    {#if overrideAzimuth}
                        <input class="input" type="number" step="1" min="0" max="360" bind:value={manualAzimuth} />
                        <p class="hint">Orientamento pannelli in gradi (180° = sud)</p>
                    {:else}
                        <p class="hint text-meta">
                            Ottimale dal profilo:
                            {selectedSolarProfile?.optimal_azimuth_degrees ?? "—"}°
                        </p>
                    {/if}
                </div>
            </div>

            <div class="divider"></div>

            <!-- Inverter -->
            <div class="section-subtitle">Inverter</div>
            <div class="form-row">
                <div class="form-group">
                    <label class="label" for="inverter-select">Seleziona inverter</label>
                    <select id="inverter-select" class="select" bind:value={selectedInverterIndex} on:change={onInverterChange}>
                        <option value={-1}>Manuale (inserisci kW)</option>
                        {#each inverters as inv, i}
                            <option value={i}>{inv.name} ({inv.p_ac_max_kw ?? inv.nominal_power_kw} kW)</option>
                        {/each}
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="inverter-power">
                        Potenza AC max (kW)
                        <span class="tooltip" title="Potenza massima AC dell'inverter. Determina la soglia di clipping DC.">ⓘ</span>
                    </label>
                    <input id="inverter-power" class="input" type="number" step="0.1" min="0.5" bind:value={inverterPAcMaxKw} />
                </div>
            </div>

            <div class="divider"></div>

            <!-- Battery -->
            <div class="section-subtitle">Accumulo</div>
            <div class="form-row">
                <div class="form-group">
                    <label class="label" for="battery-select">Seleziona batteria</label>
                    <select id="battery-select" class="select" bind:value={selectedBatteryIndex} on:change={onBatteryChange}>
                        <option value={-1}>Manuale (inserisci specs)</option>
                        {#each batteries as bat, i}
                            <option value={i}>{bat.name} ({bat.capacity_kwh} kWh)</option>
                        {/each}
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="n-batteries">N° batterie</label>
                    <input id="n-batteries" class="input" type="number" min="0" step="1" bind:value={nBatteries} />
                    <p class="hint">0 = nessuna batteria</p>
                </div>
            </div>
            {#if nBatteries > 0}
                <div class="form-row">
                    <div class="form-group">
                        <label class="label" for="battery-cap">
                            Capacità per unità (kWh)
                            <span class="tooltip" title="Capacità nominale di ogni batteria.">ⓘ</span>
                        </label>
                        <input id="battery-cap" class="input" type="number" step="0.1" min="0.5" bind:value={batteryCapacityKwh} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="battery-cycles">
                            Cicli di vita (degrado batteria)
                            <span class="tooltip" title="Numero di cicli di carica/scarica prima che la batteria raggiunga il fine vita (tipicamente 5000–8000).">ⓘ</span>
                        </label>
                        <input id="battery-cycles" class="input" type="number" step="100" min="500" bind:value={batteryCyclesLife} />
                    </div>
                </div>
            {/if}
        </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- STEP 3 — Profilo di carico                                         -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    {:else if currentStep === 2}
        <div class="step-content card">
            <h2 class="step-title">3. Profilo di carico</h2>
            <p class="step-desc">
                Definisci come consumi energia: puoi selezionare un profilo già
                salvato nel database oppure configurarne uno personalizzato.
            </p>

            <div class="form-group">
                <label class="label">Sorgente del profilo</label>
                <div class="radio-group">
                    <label class="radio-label">
                        <input type="radio" name="load-source" value="db" bind:group={loadSource} />
                        Dal database
                    </label>
                    <label class="radio-label">
                        <input type="radio" name="load-source" value="inline" bind:group={loadSource} />
                        Personalizzato (inline)
                    </label>
                </div>
            </div>

            {#if loadSource === "db"}
                <div class="form-group">
                    <label class="label" for="load-profile-id">Profilo salvato</label>
                    <select id="load-profile-id" class="select" bind:value={selectedLoadProfileId}>
                        <option value="">— Nessuno (usa inline) —</option>
                        {#each loadProfiles as lp}
                            <option value={String(lp.id)}>{describeLoadProfile(lp)}</option>
                        {/each}
                    </select>
                    {#if selectedLoadProfileId}
                        <div class="card subtle preview-box">
                            <strong>Profilo:</strong>
                            {describeLoadProfile(loadProfiles.find((p) => String(p.id) === selectedLoadProfileId))}
                            <p class="hint">Per modificarlo apri <em>Database → Profili di carico</em>.</p>
                        </div>
                    {:else}
                        <p class="hint text-meta">
                            Nessun profilo selezionato — verranno usati i parametri inline
                            dello step 3 (passa a "Personalizzato").
                        </p>
                    {/if}
                </div>
            {:else}
                <!-- Inline custom editor -->
                <div class="form-group">
                    <label class="label" for="home-profile-type">Tipo profilo casa (home)</label>
                    <select id="home-profile-type" class="select" bind:value={homeProfileType}>
                        <option value="arera">ARERA (standard italiano)</option>
                        <option value="custom">Media mensile (W)</option>
                        <option value="custom_24h">Profilo 24h per mese (W)</option>
                        <option value="weekly">Pattern settimanale (7×24)</option>
                    </select>
                </div>

                {#if homeProfileType === "custom"}
                    <MonthInput label="Potenza media casa (W/mese)" bind:values={homeProfilesW} />
                {:else if homeProfileType === "custom_24h"}
                    <MonthlyProfileEditor label="Profilo 24h — Casa (W)" bind:values={homeProfiles24h} />
                {:else if homeProfileType === "weekly"}
                    <!-- Weekly standalone: monthly average + 7×24 pattern -->
                    <MonthInput label="Consumo medio mensile (W)" bind:values={weeklyMonthlyW} />
                    <WeeklyPatternEditor bind:values={weeklyPattern} />
                {/if}

                {#if homeProfileType !== "weekly"}
                    <!-- Away profile (not shown for pure weekly) -->
                    <div class="divider"></div>
                    <div class="form-group">
                        <label class="label" for="away-profile-type">Tipo profilo quando sei via (away)</label>
                        <select id="away-profile-type" class="select" bind:value={awayProfileType}>
                            <option value="arera">ARERA (standard italiano)</option>
                            <option value="custom">Media mensile (W)</option>
                            <option value="custom_24h">Profilo 24h per mese (W)</option>
                        </select>
                    </div>
                    {#if awayProfileType === "custom"}
                        <MonthInput label="Potenza media via (W/mese)" bind:values={awayProfilesW} />
                    {:else if awayProfileType === "custom_24h"}
                        <MonthlyProfileEditor label="Profilo 24h — Via (W)" bind:values={awayProfiles24h} />
                    {/if}
                {/if}
            {/if}

            <div class="divider"></div>
            <div class="section-subtitle">Giorni a casa per mese</div>
            <p class="hint">
                Per ogni mese il simulatore estrae uniformemente un numero di
                giorni a casa tra [min, max], poi sceglie casualmente quali.
            </p>
            <MonthInput label="Giorni minimi a casa / mese" bind:values={minDaysHome} />
            <MonthInput label="Giorni massimi a casa / mese" bind:values={maxDaysHome} />
        </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- STEP 4 — Mercato elettrico                                          -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    {:else if currentStep === 3}
        <div class="step-content card">
            <h2 class="step-title">4. Mercato elettrico</h2>
            <p class="step-desc">
                Scegli come evolverà il prezzo dell'energia durante l'orizzonte
                di simulazione. Modelli stocastici danno una stima più realistica
                del rischio di investimento.
            </p>

            <div class="form-group">
                <label class="label" for="price-model-type">Modello del prezzo</label>
                <select id="price-model-type" class="select" bind:value={priceModelType}>
                    <option value="escalating">Deterministico (escalation lineare)</option>
                    <option value="gbm">Random Walk — GBM (Geometric Brownian Motion)</option>
                    <option value="mean_reverting">Mean-reverting — Ornstein–Uhlenbeck</option>
                </select>
                <p class="hint">
                    {#if priceModelType === "escalating"}
                        Crescita deterministica con tasso fisso + variazione stocastica opzionale iid. Sottostima il rischio su orizzonti lunghi.
                    {:else if priceModelType === "gbm"}
                        Random walk in log-spazio: gli shock si accumulano. Banda di incertezza cresce con la radice del tempo. Parametri default: drift 2.5 %, vol 8 %/anno (storico EU residenziale pre-2021).
                    {:else}
                        Il prezzo tende verso un livello di equilibrio a lungo termine. Adatto se credi che le oscillazioni recenti siano anomalie temporanee.
                    {/if}
                </p>
            </div>

            <div class="form-group">
                <label class="label" for="base-price">Prezzo base attuale (€/kWh)</label>
                <input id="base-price" class="input" type="number" step="0.01" min="0.05" bind:value={basePriceEur} />
            </div>

            {#if priceModelType === "escalating"}
                <div class="form-row">
                    <div class="form-group">
                        <label class="label" for="annual-escalation">
                            Escalation annua
                            <span class="tooltip" title="Tasso di crescita annuale deterministica del prezzo (es. 0.025 = +2.5%/anno).">ⓘ</span>
                        </label>
                        <input id="annual-escalation" class="input" type="number" step="0.005" bind:value={annualEscalation} />
                        <p class="hint">{(annualEscalation * 100).toFixed(2)} %/anno</p>
                    </div>
                </div>
                <div class="form-group">
                    <label class="checkbox-label">
                        <input type="checkbox" bind:checked={useStochasticEscalation} />
                        Variazione stocastica attorno al trend
                    </label>
                </div>
                {#if useStochasticEscalation}
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="esc-p05">Variazione P05 (shock negativo)</label>
                            <input id="esc-p05" class="input" type="number" step="0.01" bind:value={escalationP05} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="esc-p95">Variazione P95 (shock positivo)</label>
                            <input id="esc-p95" class="input" type="number" step="0.01" bind:value={escalationP95} />
                        </div>
                    </div>
                {/if}

            {:else if priceModelType === "gbm"}
                <div class="form-row">
                    <div class="form-group">
                        <label class="label" for="gbm-drift">
                            Drift annuo (μ)
                            <span class="tooltip" title="Tasso di crescita atteso annuo in log-spazio. 0.025 = +2.5%/anno in media.">ⓘ</span>
                        </label>
                        <input id="gbm-drift" class="input" type="number" step="0.005" bind:value={gbmDrift} />
                        <p class="hint">{(gbmDrift * 100).toFixed(2)} %/anno</p>
                    </div>
                    <div class="form-group">
                        <label class="label" for="gbm-vol">
                            Volatilità annua (σ)
                            <span class="tooltip" title="Deviazione standard annua dei rendimenti. 0.08 = 8%/anno. Valori storici EU: 6–15%.">ⓘ</span>
                        </label>
                        <input id="gbm-vol" class="input" type="number" step="0.005" min="0" bind:value={gbmVolatility} />
                        <p class="hint">{(gbmVolatility * 100).toFixed(2)} %/anno</p>
                    </div>
                </div>

            {:else if priceModelType === "mean_reverting"}
                <div class="form-row">
                    <div class="form-group">
                        <label class="label" for="mr-speed">
                            Velocità di mean-reversion (κ)
                            <span class="tooltip" title="Velocità con cui il prezzo torna al livello di equilibrio. κ = 0.5 → semivita ~1.4 anni.">ⓘ</span>
                        </label>
                        <input id="mr-speed" class="input" type="number" step="0.05" min="0.01" bind:value={mrMeanRevSpeed} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="mr-lt-price">
                            Prezzo equilibrio di lungo periodo (€/kWh)
                            <span class="tooltip" title="Valore verso cui il prezzo tende asintoticamente.">ⓘ</span>
                        </label>
                        <input id="mr-lt-price" class="input" type="number" step="0.01" min="0.05" bind:value={mrLongTermPrice} />
                    </div>
                </div>
                <div class="form-group">
                    <label class="label" for="mr-vol">
                        Volatilità annua (σ)
                        <span class="tooltip" title="Deviazione standard del processo stocastico attorno al livello di equilibrio.">ⓘ</span>
                    </label>
                    <input id="mr-vol" class="input" type="number" step="0.005" min="0" bind:value={mrVolatility} />
                    <p class="hint">{(mrVolatility * 100).toFixed(2)} %/anno</p>
                </div>
            {/if}
        </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- STEP 5 — Investimento                                              -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    {:else if currentStep === 4}
        <div class="step-content card">
            <h2 class="step-title">5. Investimento</h2>
            <p class="step-desc">
                Inserisci il costo totale dell'impianto, l'orizzonte di analisi
                e il numero di simulazioni Monte Carlo.
            </p>

            <div class="form-row">
                <div class="form-group">
                    <label class="label" for="investment">
                        Investimento totale (€)
                        <span class="tooltip" title="Costo complessivo di acquisto e installazione: pannelli, inverter, batterie, manodopera.">ⓘ</span>
                    </label>
                    <input id="investment" class="input" type="number" step="100" min="0" bind:value={investmentEur} />
                </div>
                <div class="form-group">
                    <label class="label" for="n-years">
                        Orizzonte di analisi (anni)
                        <span class="tooltip" title="Durata della simulazione. Tipicamente 20–25 anni (vita utile dei pannelli).">ⓘ</span>
                    </label>
                    <input id="n-years" class="input" type="number" step="1" min="5" max="40" bind:value={nYears} />
                </div>
            </div>

            <div class="form-group">
                <label class="label" for="n-mc">
                    Campioni Monte Carlo (N)
                    <span class="tooltip" title="Numero di traiettorie indipendenti. Più è alto, più precisi sono i percentili. 200–500 sono un buon compromesso tempo/accuratezza.">ⓘ</span>
                </label>
                <input id="n-mc" class="input" type="number" step="50" min="50" max="5000" bind:value={nMc} />
                <p class="hint">
                    {nMc} campioni &mdash; stima tempo: {nMc <= 200 ? "< 5 s" : nMc <= 500 ? "5–15 s" : "15–60 s"}
                </p>
            </div>

            <div class="divider"></div>

            <div class="form-group">
                <label class="label" for="scenario-name">
                    Nome dello scenario
                    <span class="tooltip" title="Nome con cui salvare questa configurazione nel database per riutilizzarla in futuro.">ⓘ</span>
                </label>
                <input id="scenario-name" class="input" type="text" bind:value={scenarioName} placeholder="es. Casa Milano 3kWp GBM" />
            </div>
        </div>

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- STEP 6 — Riepilogo & Esegui                                        -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    {:else if currentStep === 5}
        <div class="step-content card">
            <h2 class="step-title">6. Riepilogo e avvio analisi</h2>
            <p class="step-desc">
                Verifica la configurazione e avvia la simulazione Monte Carlo.
                Verrai reindirizzato automaticamente alla Dashboard sul risultato
                appena creato.
            </p>

            <!-- Summary table -->
            <table class="summary-table">
                <tbody>
                    {#each summaryRows as [k, v]}
                        <tr>
                            <td class="summary-key">{k}</td>
                            <td class="summary-val">{v}</td>
                        </tr>
                    {/each}
                </tbody>
            </table>

            {#if message}
                <div class={`badge ${message.toLowerCase().includes("errore") ? "error" : "success"} summary-msg`}>
                    {message}
                </div>
            {/if}

            <div class="summary-actions">
                <button
                    class="btn btn-outline"
                    on:click={() => (showSaveModal = true)}
                    disabled={loading}
                >
                    💾 Salva scenario
                </button>
                <button
                    class="btn btn-primary btn-lg"
                    on:click={handleRun}
                    disabled={loading}
                >
                    {loading ? "Simulazione in corso…" : "▶ Esegui analisi Monte Carlo"}
                </button>
            </div>
        </div>
    {/if}

    <!-- ══════════════════════════════════════════════════════════════════ -->
    <!-- Navigation bar (Indietro / Avanti)                                 -->
    <!-- ══════════════════════════════════════════════════════════════════ -->
    <div class="nav-bar">
        <button
            class="btn btn-outline"
            on:click={prev}
            disabled={currentStep === 0}
        >← Indietro</button>
        {#if currentStep < STEPS.length - 1}
            <button class="btn btn-primary" on:click={next}>Avanti →</button>
        {/if}
    </div>
</div>

<!-- Save modal -->
{#if showSaveModal}
    <div class="modal-backdrop">
        <div class="modal card">
            <h3>Salva scenario</h3>
            <div class="form-group">
                <label class="label" for="save-name">Nome scenario</label>
                <input id="save-name" class="input" bind:value={scenarioName} />
            </div>
            <div class="modal-actions">
                <button class="btn btn-text" on:click={() => (showSaveModal = false)}>Annulla</button>
                <button class="btn btn-primary" on:click={handleSave} disabled={loading}>
                    {loading ? "Salvataggio…" : "Salva"}
                </button>
            </div>
        </div>
    </div>
{/if}

<style>
    /* ── Stepper ──────────────────────────────────────────────────────────── */
    .stepper {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
        padding: 0.75rem 1rem;
        background: var(--color-bg-secondary, #f8f9fa);
        border-radius: var(--radius-sm, 6px);
        border: 1px solid var(--color-border, #e2e8f0);
    }
    .step-btn {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.7rem;
        border: none;
        background: transparent;
        cursor: pointer;
        border-radius: var(--radius-sm, 4px);
        font-size: 0.85rem;
        color: var(--color-text-secondary, #666);
        transition: color 0.15s, background 0.15s;
    }
    .step-btn:hover { background: var(--color-bg-tertiary, #e9ecef); }
    .step-btn.done { color: var(--color-success, #198754); }
    .step-btn.active {
        color: var(--color-accent, #0d6efd);
        font-weight: 700;
        background: rgba(13, 110, 253, 0.06);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.4rem;
        height: 1.4rem;
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 700;
        background: var(--color-border, #e2e8f0);
        color: var(--color-text-secondary, #666);
    }
    .step-btn.active .step-num {
        background: var(--color-accent, #0d6efd);
        color: #fff;
    }
    .step-btn.done .step-num {
        background: var(--color-success, #198754);
        color: #fff;
    }
    .step-sep {
        color: var(--color-text-muted, #adb5bd);
        font-size: 1rem;
        user-select: none;
    }

    /* ── Step content ─────────────────────────────────────────────────────── */
    .step-content {
        margin-bottom: 1rem;
        padding: 1.5rem 1.75rem;
    }
    .step-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 0.35rem;
        color: var(--color-text-primary, inherit);
    }
    .step-desc {
        color: var(--color-text-secondary, #666);
        font-size: 0.9rem;
        margin: 0 0 1.25rem;
    }
    .section-subtitle {
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--color-text-muted, #888);
        margin: 1rem 0 0.5rem;
    }
    .form-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }

    /* ── Solar preview ────────────────────────────────────────────────────── */
    .preview-card {
        margin-top: 1rem;
        padding: 1rem;
        overflow-x: auto;
    }
    .preview-header {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        margin-bottom: 0.75rem;
    }
    .preview-tilt {
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
    }
    .month-grid-header,
    .month-grid-row {
        display: grid;
        grid-template-columns: 7.5rem repeat(12, 1fr);
        gap: 0.2rem;
        font-size: 0.78rem;
        align-items: center;
        margin-bottom: 0.2rem;
    }
    .month-grid-header {
        font-weight: 600;
        color: var(--color-text-muted, #888);
    }
    .row-label {
        font-size: 0.78rem;
        color: var(--color-text-secondary, #666);
        font-weight: 500;
        white-space: nowrap;
    }
    .month-col { text-align: center; }
    .data-cell {
        background: var(--color-bg-secondary, #f8f9fa);
        border-radius: 3px;
        padding: 0.1rem 0;
        font-variant-numeric: tabular-nums;
    }
    .info-box {
        padding: 0.75rem 1rem;
        background: var(--color-bg-secondary, #f8f9fa);
        border-radius: var(--radius-sm, 4px);
        border: 1px dashed var(--color-border, #e2e8f0);
        font-size: 0.88rem;
        color: var(--color-text-secondary, #666);
        margin-top: 0.75rem;
    }

    /* ── Load profile preview box ─────────────────────────────────────────── */
    .preview-box {
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }
    .subtle {
        background: var(--color-bg-secondary, #f8f9fa);
        border: 1px dashed var(--color-border, #e2e8f0);
    }

    /* ── Summary table ────────────────────────────────────────────────────── */
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
    }
    .summary-table tr {
        border-bottom: 1px solid var(--color-border, #e2e8f0);
    }
    .summary-table tr:last-child { border-bottom: none; }
    .summary-key {
        padding: 0.5rem 0.75rem;
        font-weight: 600;
        color: var(--color-text-secondary, #666);
        white-space: nowrap;
        width: 40%;
    }
    .summary-val {
        padding: 0.5rem 0.75rem;
        color: var(--color-text-primary, inherit);
    }
    .summary-actions {
        display: flex;
        gap: 1rem;
        justify-content: flex-end;
        margin-top: 1rem;
    }
    .summary-msg { margin-bottom: 1rem; }

    /* ── Navigation bar ───────────────────────────────────────────────────── */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        padding: 1rem 0 2rem;
    }

    /* ── Tooltip ──────────────────────────────────────────────────────────── */
    .tooltip {
        display: inline-block;
        cursor: help;
        color: var(--color-text-muted, #aaa);
        font-size: 0.8rem;
        margin-left: 0.2rem;
    }

    /* ── Radio group ──────────────────────────────────────────────────────── */
    .radio-group {
        display: flex;
        gap: 1.5rem;
    }
    .radio-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        cursor: pointer;
        font-size: 0.9rem;
    }

    /* ── Misc ─────────────────────────────────────────────────────────────── */
    .page-subtitle {
        color: var(--color-text-secondary, #666);
        margin-top: 0.25rem;
        font-size: 0.95rem;
    }
    .hint {
        color: var(--color-text-secondary, #666);
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }
    .header-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-top: 0.75rem;
    }
    .badge.error {
        background: var(--color-danger, #dc3545);
        color: #fff;
    }
    .badge.success {
        background: var(--color-success, #198754);
        color: #fff;
    }
</style>
