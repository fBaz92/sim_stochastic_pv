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
    import { get } from "svelte/store";
    import { api } from "../api";
    import { pendingRunId, activeJob, pendingConfigurationId } from "../lib/stores";
    import MonthInput from "../components/forms/MonthInput.svelte";
    import MonthlyProfileEditor from "../components/forms/MonthlyProfileEditor.svelte";
    import WeeklyPatternEditor from "../components/forms/WeeklyPatternEditor.svelte";
    // Phase 14 — Geolocation + PVGIS + Open-Meteo for the Luogo step.
    import LeafletMap from "../components/LeafletMap.svelte";
    import LocationSearch from "../components/LocationSearch.svelte";
    import ClimateNormalsPreview from "../components/ClimateNormalsPreview.svelte";
    // Phase 15 — Stochastic thermal model fan-chart.
    import TemperaturePreview from "../components/TemperaturePreview.svelte";

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
    // Phase 16 — panel catalog and known climate profiles. Both are
    // loaded on mount alongside the other catalogs so the detailed
    // electrical accordion can populate its dropdowns instantly when
    // the user toggles it on.
    let panels = [];
    let climateProfiles = [];

    // ── Step 1 — Luogo ─────────────────────────────────────────────────────
    let selectedSolarProfileId = "";

    /** The full SolarProfileResponse object for the selected location. */
    $: selectedSolarProfile = solarProfiles.find(
        (p) => String(p.id) === selectedSolarProfileId,
    ) ?? null;

    const MONTHS_SHORT = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"];

    // ── Step 1 (Phase 14) — Add a profile from map ─────────────────────────
    // Opt-in sub-flow that wraps geocoding + map + PVGIS import + Open-Meteo
    // climate normals. Stays collapsed by default so the existing dropdown
    // remains the obvious entry point. When the user expands it they can:
    //
    //   1. Type a place name → Nominatim suggests candidates
    //   2. Pick a candidate → map centres there, lat/lon update
    //   3. Drag the marker / click the map → refine the spot
    //   4. Climate normals (tmax, tmin, p_sunny) fetch in the background
    //   5. Press "Importa profilo da PVGIS" → new SolarProfileModel created
    //      on the backend and auto-selected in the dropdown.
    let showLocationFinder = false;
    let pickedLat = 44.336;        // Pavullo default
    let pickedLon = 10.831;
    let pickedDisplayName = "";
    let importTilt = 35.0;
    let importAzimuth = 180.0;
    let importLossPct = 14.0;
    let importLookbackYears = 10;
    let importName = "";
    let climateNormals = null;
    let climateLoading = false;
    let climateError = null;
    let importing = false;
    let importError = null;
    let climateDebounceTimer = null;
    // Phase 15 — also calibrate a ClimateProfileModel (stochastic thermal
    // model) when the user imports the PVGIS profile. Default ON because
    // the calibration is cheap (one extra Open-Meteo call already done for
    // the climate-normals panel) and unlocks the temperature fan chart.
    let alsoCalibrateThermal = true;
    let thermalPreview = null;
    let thermalPreviewLoading = false;
    let thermalPreviewError = null;

    function onLocationPicked(event) {
        const r = event.detail;
        pickedLat = r.latitude;
        pickedLon = r.longitude;
        pickedDisplayName = r.display_name;
        // Auto-suggest a profile name from the first token of the display
        // (the city / locality usually). The user can edit it.
        const shortName = (r.display_name.split(",")[0] || "Luogo").trim();
        if (!importName) importName = shortName;
        scheduleClimateFetch();
    }

    function onMapChange(event) {
        pickedLat = event.detail.lat;
        pickedLon = event.detail.lon;
        scheduleClimateFetch();
    }

    function scheduleClimateFetch() {
        clearTimeout(climateDebounceTimer);
        climateDebounceTimer = setTimeout(fetchClimateNormals, 600);
    }

    async function fetchClimateNormals() {
        climateLoading = true;
        climateError = null;
        try {
            climateNormals = await api.getClimateNormals(pickedLat, pickedLon, {
                lookback_years: importLookbackYears,
            });
        } catch (err) {
            climateError = err.message || "Errore di rete";
            climateNormals = null;
        } finally {
            climateLoading = false;
        }
    }

    /** Build a default profile name from lat/lon when the user didn't type one.
     *  Example: (45.3393, 10.1985) → "Pos_45.34_10.20". This guarantees the
     *  ``name`` field is never empty so the backend call always proceeds; the
     *  user can rename the profile later from the database manager. */
    function defaultImportNameFromCoords() {
        const lat = pickedLat.toFixed(2);
        const lon = pickedLon.toFixed(2);
        return `Pos_${lat}_${lon}`.replace(/-/g, "S").replace(/\./g, "_");
    }

    async function importProfileFromLocation() {
        // Fall back to a coordinate-based name instead of refusing the call:
        // makes the button behaviour predictable in the common case where the
        // user only dragged the map without picking from search.
        const effectiveName = (importName && importName.trim())
            || defaultImportNameFromCoords();
        importName = effectiveName;

        importError = null;
        importing = true;
        thermalPreview = null;
        thermalPreviewError = null;

        const locationLabel = pickedDisplayName
            || `${pickedLat.toFixed(3)}°, ${pickedLon.toFixed(3)}°`;

        try {
            // Step 1: PVGIS solar profile.
            const record = await api.createSolarProfileFromLocation({
                name: effectiveName,
                location_name: locationLabel,
                latitude: pickedLat,
                longitude: pickedLon,
                tilt_degrees: importTilt,
                azimuth_degrees: importAzimuth,
                loss_pct: importLossPct,
                lookback_years: importLookbackYears,
                overwrite: false,
            });
            // Reload list + auto-select the new record.
            solarProfiles = await api.listSolarProfiles();
            selectedSolarProfileId = String(record.id);

            // Step 2 (Phase 15, opt-in): also calibrate the stochastic
            // thermal model and show the fan-chart preview. We do this
            // AFTER the solar profile is saved so the user sees the
            // solar import as immediately successful even if the
            // thermal calibration fails (e.g. very short archive).
            if (alsoCalibrateThermal) {
                await calibrateAndPreviewThermal(effectiveName, locationLabel);
            } else {
                showLocationFinder = false;
            }
        } catch (err) {
            importError = err.message || "Errore durante l'import PVGIS.";
        } finally {
            importing = false;
        }
    }

    async function calibrateAndPreviewThermal(name, locationLabel) {
        thermalPreviewLoading = true;
        thermalPreviewError = null;
        try {
            const climateRecord = await api.createClimateProfileFromLocation({
                name: name,                  // same name across solar+climate DB tables
                location_name: locationLabel,
                latitude: pickedLat,
                longitude: pickedLon,
                lookback_years: importLookbackYears,
                climate_trend_c_per_year: 0.0,
                overwrite: true,             // overwrite to keep solar↔climate paired
            });
            // Phase 16 — keep the climate profile id for the electrical
            // accordion and refresh the cached list so the auto-match
            // reactive statement picks it up immediately.
            climateProfileId = climateRecord.id;
            climateProfiles = await api.listClimateProfiles().catch(() => climateProfiles);
            thermalPreview = await api.previewClimateProfileById(climateRecord.id, {
                n_paths: 50,
                n_years: 1,
                seed: 42,
            });
        } catch (err) {
            thermalPreviewError = err.message || "Errore durante la calibrazione termica.";
        } finally {
            thermalPreviewLoading = false;
        }
    }

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

    // ── Phase 16 — detailed electrical model (opt-in) ──────────────────────
    // The accordion appears inside step Impianto. When the user opts in,
    // we need: a panel pick (datasheet electrical specs), the inverter
    // electrical specs (already part of the selected inverter), and a
    // climate_profile_id (Phase 15) on the scenario root. The legacy flow
    // is preserved when ``electricalEnabled`` stays false.
    let electricalEnabled = false;
    let selectedPanelIndex = -1;
    let selectedPanelId = null;
    let electricalDeratingExponentK = 0.5;
    // Climate profile id chosen / created in step Luogo. Picked up
    // automatically by ``calibrateAndPreviewThermal`` and via a name
    // match between the selected solar profile and listClimateProfiles.
    let climateProfileId = null;

    function onPanelChange() {
        if (selectedPanelIndex >= 0) {
            const p = panels[selectedPanelIndex];
            selectedPanelId = p.id;
        } else {
            selectedPanelId = null;
        }
    }

    /**
     * Extract the electrical sub-block of the Phase-16 scenario JSON
     * from the currently selected panel and inverter. Returns null when
     * the user has not enabled the detailed model OR when one of the
     * required hardware picks is missing — in both cases the simulator
     * stays on the byte-identical legacy energy path.
     */
    function buildElectricalBlock() {
        if (!electricalEnabled) return null;
        if (selectedPanelIndex < 0 || selectedInverterIndex < 0) return null;
        const p = panels[selectedPanelIndex] ?? {};
        const inv = inverters[selectedInverterIndex] ?? {};
        const pSpecs = p.specs ?? {};
        const iSpecs = inv.specs ?? {};
        return {
            mode: "mppt_window",
            panel: {
                power_w: p.power_w ?? pSpecs.power_w,
                v_oc_stc_v: p.v_oc_stc_v ?? pSpecs.v_oc_stc_v,
                v_mpp_stc_v: p.v_mpp_stc_v ?? pSpecs.v_mpp_stc_v,
                i_sc_stc_a: p.i_sc_stc_a ?? pSpecs.i_sc_stc_a,
                i_mpp_stc_a: p.i_mpp_stc_a ?? pSpecs.i_mpp_stc_a,
                n_cells_series: p.n_cells_series ?? pSpecs.n_cells_series,
                beta_voc_pct_per_c: p.beta_voc_pct_per_c ?? pSpecs.beta_voc_pct_per_c,
                gamma_pmax_pct_per_c: p.gamma_pmax_pct_per_c ?? pSpecs.gamma_pmax_pct_per_c,
                noct_c: p.noct_c ?? pSpecs.noct_c,
            },
            inverter: {
                v_dc_min_v: inv.v_dc_min_v ?? iSpecs.v_dc_min_v,
                v_dc_max_v: inv.v_dc_max_v ?? iSpecs.v_dc_max_v,
                v_mppt_min_v: inv.v_mppt_min_v ?? iSpecs.v_mppt_min_v,
                v_mppt_max_v: inv.v_mppt_max_v ?? iSpecs.v_mppt_max_v,
                n_mppt_trackers: inv.n_mppt_trackers ?? iSpecs.n_mppt_trackers ?? 1,
                i_dc_max_per_mppt_a: inv.i_dc_max_per_mppt_a ?? iSpecs.i_dc_max_per_mppt_a,
            },
            derating_exponent_k: electricalDeratingExponentK,
        };
    }

    // Phase 16 — derived flag: the toggle should be DISABLED when the
    // user has not selected a Climate profile (no T_ambient available).
    $: electricalReady = selectedPanelIndex >= 0 && selectedInverterIndex >= 0 && climateProfileId != null;
    // When the selected solar profile shares a name with a climate
    // profile already in the DB, auto-link them so the user does not
    // have to re-pick. Runs whenever solarProfileId or the lists change.
    $: {
        if (selectedSolarProfile && climateProfiles.length > 0) {
            const match = climateProfiles.find(
                (c) => c.name === selectedSolarProfile.name,
            );
            if (match) climateProfileId = match.id;
        }
    }

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

    // Phase 11+ — variation percentiles applied to the inline profile.
    // The simulator interprets [p05, p95] as the bracket of multiplicative
    // perturbation applied to the base load before each Monte Carlo path.
    // Default 10% on home, 5% on away matches the backend defaults.
    let homeVariationPercent = 10;   // ±%, both sides symmetric
    let awayVariationPercent = 5;

    // ── Phase 17 — stochastic intra-day variability + HVAC additive load ──
    let stochasticLoadEnabled = false;
    let stochasticSigmaLog = 0.20;
    let stochasticPhiIntraDay = 0.5;

    let thermalLoadEnabled = false;
    let thermalFloorAreaM2 = 100;
    let thermalInsulationPreset = "standard"; // "poor" | "standard" | "good"
    let thermalCopHeating = 3.5;
    let thermalCopCooling = 3.0;
    let thermalPMaxKw = 3.0;
    let thermalTSetpointHeatingC = 20;
    let thermalTSetpointCoolingC = 26;

    // Status banner for the Excel template / import actions.
    let loadProfileStatus = "";
    let loadProfileError = "";

    /**
     * Map the wizard's profile-type identifier to the kind expected by
     * the backend Excel template/parse endpoints.
     *   - "custom"    → "monthly_avg"
     *   - "custom_24h"→ "monthly_24h"
     *   - "weekly"    → "weekly"
     *   - everything else returns null (no template available).
     */
    function profileKindForBackend(uiType) {
        if (uiType === "custom") return "monthly_avg";
        if (uiType === "custom_24h") return "monthly_24h";
        if (uiType === "weekly") return "weekly";
        return null;
    }

    /** Trigger a browser download for the matching Excel template. */
    function downloadLoadProfileTemplate(uiType) {
        const kind = profileKindForBackend(uiType);
        if (!kind) return;
        const a = document.createElement("a");
        a.href = api.loadProfileTemplateUrl(kind);
        a.download = `load_profile_${kind}_template.xlsx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    /**
     * Read a user-selected .xlsx file, POST it to the parse endpoint,
     * and merge the result into the appropriate wizard state slot.
     *
     * @param {Event} ev     Input change event.
     * @param {string} uiType  homeProfileType / awayProfileType value.
     * @param {"home"|"away"} side  Which side of the home/away pair to update.
     */
    async function handleLoadProfileUpload(ev, uiType, side) {
        loadProfileStatus = "";
        loadProfileError = "";
        const file = ev.target.files?.[0];
        if (!file) return;
        const kind = profileKindForBackend(uiType);
        if (!kind) {
            loadProfileError = "Nessun template disponibile per questo tipo di profilo.";
            return;
        }
        try {
            const data = await api.parseLoadProfileXlsx(kind, file);
            applyParsedLoadProfile(data, uiType, side);
            loadProfileStatus = `Profilo "${side === "home" ? "casa" : "via"}" importato da ${file.name}.`;
        } catch (e) {
            loadProfileError = `Errore import: ${e.message}`;
        } finally {
            // Reset the input so the same file can be re-selected.
            ev.target.value = "";
        }
    }

    function applyParsedLoadProfile(parsed, uiType, side) {
        if (uiType === "custom") {
            const arr = parsed.monthly_avg_w;
            if (!Array.isArray(arr) || arr.length !== 12) {
                throw new Error("Il file non contiene 12 valori mensili.");
            }
            if (side === "home") homeProfilesW = [...arr];
            else awayProfilesW = [...arr];
        } else if (uiType === "custom_24h") {
            const grid = parsed.monthly_24h_w;
            if (!Array.isArray(grid) || grid.length !== 12 || grid.some((r) => r.length !== 24)) {
                throw new Error("Il file non ha forma 12×24.");
            }
            if (side === "home") homeProfiles24h = grid.map((r) => [...r]);
            else awayProfiles24h = grid.map((r) => [...r]);
        } else if (uiType === "weekly") {
            const monthly = parsed.monthly_w;
            const pattern = parsed.weekly_pattern_w;
            if (!Array.isArray(monthly) || monthly.length !== 12) {
                throw new Error("La 'Scala mensile' deve avere 12 valori.");
            }
            if (!Array.isArray(pattern) || pattern.length !== 7 || pattern.some((r) => r.length !== 24)) {
                throw new Error("Il 'Pattern settimanale' deve essere 7×24.");
            }
            weeklyMonthlyW = [...monthly];
            weeklyPattern = pattern.map((r) => [...r]);
        }
    }

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

    // Phase 11 — optional tax bonus block. The UI shows percentages
    // (0–100); the payload assembly converts to fractions (0–1).
    let taxBonusEnabled = false;
    let taxBonusFractionPercent = 50;
    let taxBonusDurationYears = 10;

    // Phase 11 — optional inflation overrides. When ``inflationOverride``
    // is false the payload omits the ``economic.inflation`` block and
    // the backend falls back to the legacy deterministic behaviour.
    let inflationOverride = false;
    let inflationMode = "deterministic"; // 'deterministic' | 'stochastic'
    let inflationMeanPercent = 2.5;
    let inflationStdPercent = 1.0;
    let inflationMinClipPercent = -2.0;
    let inflationMaxClipPercent = 10.0;

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
            const [sp, inv, bat, lp, configs, pan, climate] = await Promise.all([
                api.listSolarProfiles(),
                api.listInverters(),
                api.listBatteries(),
                api.listLoadProfiles(),
                api.listConfigurations("scenario"),
                api.listPanels(),
                api.listClimateProfiles().catch(() => []),
            ]);
            solarProfiles = sp;
            inverters = inv;
            batteries = bat;
            loadProfiles = lp;
            savedScenarios = configs;
            panels = pan;
            climateProfiles = climate;
        } catch (e) {
            console.error("Failed to load initial data:", e);
        }
        // If the Database UI asked us to open a specific saved scenario,
        // pick it up here and run the existing "load" flow once the data
        // has hydrated.
        const pending = get(pendingConfigurationId);
        if (pending != null) {
            pendingConfigurationId.set(null);
            selectedSavedScenarioId = String(pending);
            await handleLoadScenario();
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
                home_variation_percentiles: [
                    -homeVariationPercent / 100,
                    homeVariationPercent / 100,
                ],
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
            home_variation_percentiles: [
                -homeVariationPercent / 100,
                homeVariationPercent / 100,
            ],
            away_variation_percentiles: [
                -awayVariationPercent / 100,
                awayVariationPercent / 100,
            ],
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

        const economic = {
            investment_eur: investmentEur,
            n_mc: nMc,
        };
        // Phase 11 — optional sub-blocks. Only included when the user
        // explicitly opted in, so legacy scenarios stay byte-identical.
        if (taxBonusEnabled) {
            economic.tax_bonus = {
                enabled: true,
                fraction_of_investment: taxBonusFractionPercent / 100,
                duration_years: taxBonusDurationYears,
            };
        }
        if (inflationOverride) {
            economic.inflation = {
                mode: inflationMode,
                mean: inflationMeanPercent / 100,
                std: inflationStdPercent / 100,
                min_clip: inflationMinClipPercent / 100,
                max_clip: inflationMaxClipPercent / 100,
            };
        }

        let scenarioClone = {
            scenario_name: scenarioName,
            solar,
            energy,
            price: buildPriceBlock(),
            economic,
        };

        if (selectedInverterId) scenarioClone.inverter_id = selectedInverterId;
        if (selectedBatteryId) scenarioClone.battery_id = selectedBatteryId;
        if (selectedPanelId) scenarioClone.panel_id = selectedPanelId;

        // Phase 16 — opt-in detailed electrical block. When the user
        // does not enable the toggle the helper returns null and the
        // legacy energy path stays byte-identical.
        const electrical = buildElectricalBlock();
        if (electrical) {
            scenarioClone.electrical = electrical;
            if (climateProfileId != null) {
                scenarioClone.climate_profile_id = climateProfileId;
            }
        }

        applyLoadProfile(scenarioClone);

        // Phase 17 — append the optional stochastic-load block under
        // load_profile.stochastic. The simulator reads it from there.
        if (stochasticLoadEnabled && scenarioClone.load_profile) {
            scenarioClone.load_profile.stochastic = {
                enabled: true,
                sigma_log: Number(stochasticSigmaLog),
                phi_intra_day: Number(stochasticPhiIntraDay),
            };
        }

        // Phase 17 — append the optional thermal_load (HVAC) block. The
        // climate_profile_id is propagated up so the backend can wire
        // the ThermalModel from the climate DB.
        if (thermalLoadEnabled) {
            scenarioClone.thermal_load = {
                enabled: true,
                house: {
                    floor_area_m2: Number(thermalFloorAreaM2),
                    insulation_preset: thermalInsulationPreset,
                },
                heat_pump: {
                    cop_heating: Number(thermalCopHeating),
                    cop_cooling: Number(thermalCopCooling),
                    p_elec_max_kw: Number(thermalPMaxKw),
                },
                setpoint: {
                    t_setpoint_heating_c: Number(thermalTSetpointHeatingC),
                    t_setpoint_cooling_c: Number(thermalTSetpointCoolingC),
                },
            };
            if (climateProfileId != null && scenarioClone.climate_profile_id == null) {
                scenarioClone.climate_profile_id = climateProfileId;
            }
        }

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
            // Phase 12 — submit the analysis as a background job. The
            // floating JobProgress widget handles the redirect when done.
            const { job_id } = await api.submitAnalysisJob(payload);
            activeJob.set({
                id: job_id,
                kind: "analysis",
                status: "pending",
                progress_done: 0,
                progress_total: payload.n_mc ?? 0,
                progress_fraction: 0,
                message: "In attesa di avvio...",
                run_id: null,
                error: null,
            });
            message = "Analisi avviata. Vedi la barra in basso a sinistra per il progresso.";
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
                // The backend stores 12×24 grids under "home_profiles_w" /
                // "away_profiles_w" regardless of whether the user typed a
                // monthly average or a 24h profile. Differentiate by shape.
                const hw = d.load_profile.home_profiles_w;
                if (Array.isArray(hw) && hw.length === 12) {
                    if (Array.isArray(hw[0])) {
                        homeProfileType = "custom_24h";
                        homeProfiles24h = hw.map((r) => [...r]);
                    } else {
                        homeProfilesW = [...hw];
                    }
                }
                const aw = d.load_profile.away_profiles_w;
                if (Array.isArray(aw) && aw.length === 12) {
                    if (Array.isArray(aw[0])) {
                        awayProfileType = "custom_24h";
                        awayProfiles24h = aw.map((r) => [...r]);
                    } else {
                        awayProfilesW = [...aw];
                    }
                }
                // Phase 11+ — restore the variation brackets when the saved
                // scenario carries them. We always store symmetric values
                // in the UI; if the JSON has asymmetric brackets we pick
                // the wider side for visual safety.
                if (Array.isArray(d.load_profile.home_variation_percentiles)) {
                    const [lo, hi] = d.load_profile.home_variation_percentiles;
                    homeVariationPercent = Math.round(
                        Math.max(Math.abs(lo ?? 0), Math.abs(hi ?? 0)) * 100,
                    );
                }
                if (Array.isArray(d.load_profile.away_variation_percentiles)) {
                    const [lo, hi] = d.load_profile.away_variation_percentiles;
                    awayVariationPercent = Math.round(
                        Math.max(Math.abs(lo ?? 0), Math.abs(hi ?? 0)) * 100,
                    );
                }
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

            // Phase 11 — restore optional sub-blocks
            if (d.economic?.tax_bonus) {
                const tb = d.economic.tax_bonus;
                taxBonusEnabled = Boolean(tb.enabled);
                if (tb.fraction_of_investment != null) {
                    taxBonusFractionPercent = tb.fraction_of_investment * 100;
                }
                if (tb.duration_years != null) {
                    taxBonusDurationYears = tb.duration_years;
                }
            } else {
                taxBonusEnabled = false;
            }
            if (d.economic?.inflation) {
                const inf = d.economic.inflation;
                inflationOverride = true;
                inflationMode = inf.mode ?? "deterministic";
                if (inf.mean != null) inflationMeanPercent = inf.mean * 100;
                if (inf.std != null) inflationStdPercent = inf.std * 100;
                if (inf.min_clip != null) inflationMinClipPercent = inf.min_clip * 100;
                if (inf.max_clip != null) inflationMaxClipPercent = inf.max_clip * 100;
            } else {
                inflationOverride = false;
            }

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
        <div class="header-text">
            <h1 class="page-title">Nuovo Scenario</h1>
            <p class="page-subtitle">
                Configura <strong>un</strong> impianto PV+batteria in pochi passi e
                avvia l'analisi Monte Carlo. Per esplorare più alternative vai su
                <a href="#/design">Design</a>.
            </p>
        </div>
        <!-- Load saved scenario (top-bar shortcut) -->
        <div class="header-actions">
            <select
                class="select sm"
                bind:value={selectedSavedScenarioId}
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
    </div>
    {#if message}
        <div class={`badge ${message.toLowerCase().includes("errore") ? "error" : "success"} header-message`}>
            {message}
        </div>
    {/if}

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
                    in lista puoi crearne uno nuovo dalla mappa qui sotto, oppure
                    procedere senza selezionare e inserire i parametri manualmente
                    nello step 2.
                </p>
            </div>

            <!-- Phase 14 — Add a profile from map -->
            <div class="form-group">
                <button
                    type="button"
                    class="link-btn"
                    on:click={() => (showLocationFinder = !showLocationFinder)}
                >
                    {showLocationFinder ? "▾" : "▸"}
                    Aggiungi un nuovo profilo da mappa (PVGIS + Open-Meteo)
                </button>
            </div>

            {#if showLocationFinder}
                <div class="step-content card subtle">
                    <h3 class="step-subtitle">Nuovo profilo da posizione</h3>
                    <p class="step-desc">
                        Cerca una località o trascina il marker sulla mappa.
                        Il backend chiamerà <strong>PVGIS</strong> per i dati di
                        produzione solare e <strong>Open-Meteo</strong> per le
                        normali climatiche (probabilità di giorno sereno).
                    </p>

                    <div class="form-group">
                        <LocationSearch on:select={onLocationPicked} />
                    </div>

                    <div class="form-group">
                        <LeafletMap
                            bind:lat={pickedLat}
                            bind:lon={pickedLon}
                            on:change={onMapChange}
                        />
                        <p class="hint">
                            Lat <strong>{pickedLat.toFixed(4)}°</strong>,
                            Lon <strong>{pickedLon.toFixed(4)}°</strong>
                            {#if pickedDisplayName} · {pickedDisplayName}{/if}
                        </p>
                    </div>

                    <ClimateNormalsPreview
                        data={climateNormals}
                        loading={climateLoading}
                        error={climateError}
                    />

                    <div class="form-grid two-cols">
                        <div class="form-group">
                            <label class="label" for="import-name">Nome profilo (opzionale)</label>
                            <input
                                id="import-name"
                                class="input"
                                type="text"
                                bind:value={importName}
                                placeholder="es. Pavullo"
                            />
                            <p class="hint">
                                Se lasciato vuoto verrà generato un nome
                                automatico tipo <code>Pos_45_34_10_20</code>.
                            </p>
                        </div>
                        <div class="form-group">
                            <label class="label" for="import-tilt">Tilt (°)</label>
                            <input
                                id="import-tilt"
                                class="input"
                                type="number"
                                step="1"
                                min="0"
                                max="90"
                                bind:value={importTilt}
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="import-azimuth">Azimuth (°, 180 = sud)</label>
                            <input
                                id="import-azimuth"
                                class="input"
                                type="number"
                                step="1"
                                min="0"
                                max="360"
                                bind:value={importAzimuth}
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="import-loss">Perdite PVGIS (%)</label>
                            <input
                                id="import-loss"
                                class="input"
                                type="number"
                                step="0.5"
                                min="0"
                                max="100"
                                bind:value={importLossPct}
                            />
                            <p class="hint">14% = PR ≈ 0.86 (impianti residenziali standard).</p>
                        </div>
                    </div>

                    <div class="form-group">
                        <label class="checkbox-label">
                            <input
                                type="checkbox"
                                bind:checked={alsoCalibrateThermal}
                            />
                            <span>
                                Calibra anche il <strong>modello termico stocastico</strong>
                                (Open-Meteo Archive, {importLookbackYears} anni)
                            </span>
                        </label>
                        <p class="hint">
                            Fitta stagionalità armonica + residuo AR(1) + code GPD
                            per ondate di calore e gelate. Necessario per il
                            derating elettrico e per modellare il carico HVAC.
                        </p>
                    </div>

                    {#if importError}
                        <div class="info-box error">
                            <p>{importError}</p>
                        </div>
                    {/if}

                    <button
                        type="button"
                        class="btn primary"
                        on:click={importProfileFromLocation}
                        disabled={importing}
                    >
                        {importing ? "Importazione in corso…" : "Importa profilo da PVGIS"}
                    </button>

                    {#if alsoCalibrateThermal && (thermalPreviewLoading || thermalPreview || thermalPreviewError)}
                        <TemperaturePreview
                            data={thermalPreview}
                            loading={thermalPreviewLoading}
                            error={thermalPreviewError}
                        />
                    {/if}
                </div>
            {/if}

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

            <!-- Phase 16 — opt-in detailed electrical model. -->
            <hr class="section-divider" />
            <div class="form-group">
                <label class="toggle-row">
                    <input type="checkbox" bind:checked={electricalEnabled} />
                    <span class="toggle-label">Modello elettrico dettagliato (Phase 16 — opzionale)</span>
                </label>
                <p class="hint">
                    Quando attivo, lo scenario controlla ora-per-ora la finestra MPPT
                    dell'inverter, deratina per temperatura cella e segnala le ore
                    di shutdown V_dc. Richiede un pannello dal catalogo (con dati
                    elettrici completi), un inverter dal catalogo (idem) e un
                    profilo climatico Phase 15 dallo step Luogo.
                </p>
            </div>
            {#if electricalEnabled}
                <div class="electrical-section">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="panel-select">
                                Pannello (dal catalogo)
                                <span class="tooltip" title="Il modello MPPT usa V_oc, V_mpp e coefficienti termici dal datasheet.">ⓘ</span>
                            </label>
                            <select id="panel-select" class="select" bind:value={selectedPanelIndex} on:change={onPanelChange}>
                                <option value={-1}>— Scegli un pannello —</option>
                                {#each panels as p, i (p.id)}
                                    <option value={i}>{p.name} ({p.power_w} W)</option>
                                {/each}
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="label" for="derating-exp">
                                Esponente derating MPPT (k)
                                <span class="tooltip" title="Penalità soft fuori finestra MPPT: power × (V_target / V_string)^k. k=0 disattiva il derating MPPT.">ⓘ</span>
                            </label>
                            <input id="derating-exp" class="input" type="number" step="0.1" min="0" bind:value={electricalDeratingExponentK} />
                        </div>
                    </div>
                    <p class="hint">
                        {#if climateProfileId == null}
                            ⚠️ Manca il profilo climatico: torna allo step Luogo e
                            spunta "Calibra anche il modello termico stocastico" prima
                            di importare un profilo da PVGIS.
                        {:else if selectedPanelIndex < 0}
                            Seleziona un pannello con dati elettrici completi (v_oc,
                            v_mpp, β, γ, NOCT).
                        {:else if selectedInverterIndex < 0}
                            Seleziona un inverter dal catalogo per ottenere la
                            finestra MPPT e i limiti V_dc.
                        {:else}
                            ✓ Profilo climatico {climateProfileId} · pannello
                            <strong>{panels[selectedPanelIndex]?.name}</strong> ·
                            inverter <strong>{inverters[selectedInverterIndex]?.name}</strong> —
                            il modello elettrico dettagliato è pronto.
                        {/if}
                    </p>
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

                {#if homeProfileType !== "arera"}
                    <div class="excel-tools">
                        <button
                            type="button"
                            class="btn btn-outline btn-sm"
                            on:click={() => downloadLoadProfileTemplate(homeProfileType)}
                            title="Scarica un template Excel per questo tipo di profilo"
                        >📥 Scarica template</button>
                        <label class="btn btn-outline btn-sm import-btn">
                            📤 Importa Excel
                            <input
                                type="file"
                                accept=".xlsx"
                                on:change={(ev) => handleLoadProfileUpload(ev, homeProfileType, "home")}
                            />
                        </label>
                    </div>
                {/if}

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
                    {#if awayProfileType !== "arera"}
                        <div class="excel-tools">
                            <button
                                type="button"
                                class="btn btn-outline btn-sm"
                                on:click={() => downloadLoadProfileTemplate(awayProfileType)}
                                title="Scarica un template Excel per questo tipo di profilo"
                            >📥 Scarica template</button>
                            <label class="btn btn-outline btn-sm import-btn">
                                📤 Importa Excel
                                <input
                                    type="file"
                                    accept=".xlsx"
                                    on:change={(ev) => handleLoadProfileUpload(ev, awayProfileType, "away")}
                                />
                            </label>
                        </div>
                    {/if}
                    {#if awayProfileType === "custom"}
                        <MonthInput label="Potenza media via (W/mese)" bind:values={awayProfilesW} />
                    {:else if awayProfileType === "custom_24h"}
                        <MonthlyProfileEditor label="Profilo 24h — Via (W)" bind:values={awayProfiles24h} />
                    {/if}
                {/if}

                {#if loadProfileStatus}
                    <p class="hint success-text">{loadProfileStatus}</p>
                {/if}
                {#if loadProfileError}
                    <p class="hint error-text">{loadProfileError}</p>
                {/if}

                <!-- Phase 11+ — variation brackets for the load profile -->
                <div class="divider"></div>
                <div class="section-subtitle">Variazione del consumo (Monte Carlo)</div>
                <p class="hint">
                    Per ogni traiettoria il simulatore moltiplica il profilo per
                    un fattore casuale all'interno di una banda simmetrica
                    [-x %, +x %], catturando l'incertezza dei consumi reali.
                </p>
                <div class="form-row">
                    <div class="form-group">
                        <label class="label" for="home-variation">Variazione casa (±%)</label>
                        <input id="home-variation" class="input" type="number" step="1" min="0" max="80" bind:value={homeVariationPercent} />
                    </div>
                    {#if homeProfileType !== "weekly"}
                        <div class="form-group">
                            <label class="label" for="away-variation">Variazione via (±%)</label>
                            <input id="away-variation" class="input" type="number" step="1" min="0" max="80" bind:value={awayVariationPercent} />
                        </div>
                    {/if}
                </div>
            {/if}

            <div class="divider"></div>
            <div class="section-subtitle">Giorni a casa per mese</div>
            <p class="hint">
                Per ogni mese il simulatore estrae uniformemente un numero di
                giorni a casa tra [min, max], poi sceglie casualmente quali.
            </p>
            <MonthInput label="Giorni minimi a casa / mese" bind:values={minDaysHome} />
            <MonthInput label="Giorni massimi a casa / mese" bind:values={maxDaysHome} />

            <!-- Phase 17 — opt-in stochastic intra-day variability. -->
            <hr class="section-divider" />
            <div class="form-group">
                <label class="toggle-row">
                    <input type="checkbox" bind:checked={stochasticLoadEnabled} />
                    <span class="toggle-label">Variabilità giornaliera del consumo (Phase 17 — opzionale)</span>
                </label>
                <p class="hint">
                    Aggiunge un moltiplicatore stocastico LogN ora-per-ora con
                    autocorrelazione AR(1). Preserva per costruzione il consumo
                    medio mensile. Default ±20% 1-σ con φ=0.5.
                </p>
            </div>
            {#if stochasticLoadEnabled}
                <div class="thermal-section">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="stoch-sigma">σ log-multiplier</label>
                            <input id="stoch-sigma" class="input" type="number" step="0.01" min="0"
                                   bind:value={stochasticSigmaLog} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="stoch-phi">φ autocorrelazione AR(1)</label>
                            <input id="stoch-phi" class="input" type="number" step="0.05" min="-0.99" max="0.99"
                                   bind:value={stochasticPhiIntraDay} />
                        </div>
                    </div>
                </div>
            {/if}

            <!-- Phase 17 — opt-in HVAC additive load. -->
            <hr class="section-divider" />
            <div class="form-group">
                <label class="toggle-row">
                    <input type="checkbox" bind:checked={thermalLoadEnabled} />
                    <span class="toggle-label">Pompa di calore / HVAC con modello casa (Phase 17 — opzionale)</span>
                </label>
                <p class="hint">
                    Calcola il consumo elettrico orario di una pompa di calore
                    che mantiene la casa al setpoint con un modello RC del 1° ordine.
                    Richiede un profilo climatico Phase 15 dallo step Luogo.
                </p>
            </div>
            {#if thermalLoadEnabled}
                <div class="thermal-section">
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="th-area">Superficie pavimento (m²)</label>
                            <input id="th-area" class="input" type="number" step="5" min="20"
                                   bind:value={thermalFloorAreaM2} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="th-preset">Isolamento dell'edificio</label>
                            <select id="th-preset" class="select" bind:value={thermalInsulationPreset}>
                                <option value="poor">Scarso (~2.5 W/°C/m²) — case anni '60-'70</option>
                                <option value="standard">Standard (~1.5 W/°C/m²) — anni '90</option>
                                <option value="good">Buono (~0.8 W/°C/m²) — NZEB / classe A</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="th-cop-h">COP riscaldamento</label>
                            <input id="th-cop-h" class="input" type="number" step="0.1" min="0.5"
                                   bind:value={thermalCopHeating} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="th-cop-c">COP raffrescamento</label>
                            <input id="th-cop-c" class="input" type="number" step="0.1" min="0.5"
                                   bind:value={thermalCopCooling} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="th-pmax">P_elec max (kW)</label>
                            <input id="th-pmax" class="input" type="number" step="0.1" min="0.5"
                                   bind:value={thermalPMaxKw} />
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="th-tset-h">Setpoint riscaldamento (°C)</label>
                            <input id="th-tset-h" class="input" type="number" step="0.5"
                                   bind:value={thermalTSetpointHeatingC} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="th-tset-c">Setpoint raffrescamento (°C)</label>
                            <input id="th-tset-c" class="input" type="number" step="0.5"
                                   bind:value={thermalTSetpointCoolingC} />
                        </div>
                    </div>
                    <p class="hint">
                        {#if climateProfileId == null}
                            ⚠️ Manca il profilo climatico — torna allo step Luogo,
                            attiva la checkbox "Calibra anche il modello termico stocastico"
                            e importa il profilo PVGIS prima di proseguire.
                        {:else}
                            ✓ Profilo climatico {climateProfileId} pronto. La pompa
                            di calore aggiungerà il proprio consumo elettrico orario
                            al profilo base nelle ore in cui T_ambient esce dal
                            dead-band {thermalTSetpointHeatingC}°C–{thermalTSetpointCoolingC}°C.
                        {/if}
                    </p>
                </div>
            {/if}
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

            <!-- Phase 11 — optional tax bonus block ─────────────────────── -->
            <details class="adv-block">
                <summary>
                    Bonus fiscale (opzionale)
                    <span class="tooltip" title="Detrazione fiscale come quella italiana del 50% in 10 anni. Quando attiva, ogni anno (fine anno) ti viene restituita una quota dell'investimento.">ⓘ</span>
                </summary>
                <div class="form-row">
                    <div class="form-group">
                        <label class="label">
                            <input type="checkbox" bind:checked={taxBonusEnabled} />
                            Attiva bonus fiscale
                        </label>
                    </div>
                </div>
                {#if taxBonusEnabled}
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="bonus-fraction">
                                Percentuale dell'investimento (%)
                            </label>
                            <input id="bonus-fraction" class="input" type="number" step="1" min="0" max="100" bind:value={taxBonusFractionPercent} />
                            <p class="hint">
                                Es. 50 = 50% dell'investimento restituito complessivamente
                            </p>
                        </div>
                        <div class="form-group">
                            <label class="label" for="bonus-years">
                                Durata (anni)
                            </label>
                            <input id="bonus-years" class="input" type="number" step="1" min="1" max="20" bind:value={taxBonusDurationYears} />
                            <p class="hint">
                                Importo annuo = € {(investmentEur * (taxBonusFractionPercent / 100) / Math.max(1, taxBonusDurationYears)).toLocaleString("it-IT", {maximumFractionDigits: 0})}
                            </p>
                        </div>
                    </div>
                {/if}
            </details>

            <!-- Phase 11 — optional inflation override ──────────────────── -->
            <details class="adv-block">
                <summary>
                    Inflazione (opzionale)
                    <span class="tooltip" title="Sostituisce il tasso di inflazione predefinito (2,5%). In modalità stocastica viene estratto annualmente da una Normale troncata.">ⓘ</span>
                </summary>
                <div class="form-row">
                    <div class="form-group">
                        <label class="label">
                            <input type="checkbox" bind:checked={inflationOverride} />
                            Personalizza inflazione
                        </label>
                    </div>
                </div>
                {#if inflationOverride}
                    <div class="form-row">
                        <div class="form-group">
                            <label class="label" for="inflation-mode">Modalità</label>
                            <select id="inflation-mode" class="input" bind:value={inflationMode}>
                                <option value="deterministic">Deterministica</option>
                                <option value="stochastic">Stocastica (Normale troncata)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="label" for="inflation-mean">Media annua (%)</label>
                            <input id="inflation-mean" class="input" type="number" step="0.1" min="-5" max="20" bind:value={inflationMeanPercent} />
                        </div>
                    </div>
                    {#if inflationMode === "stochastic"}
                        <div class="form-row">
                            <div class="form-group">
                                <label class="label" for="inflation-std">Deviazione standard (%)</label>
                                <input id="inflation-std" class="input" type="number" step="0.1" min="0" max="10" bind:value={inflationStdPercent} />
                            </div>
                            <div class="form-group">
                                <label class="label" for="inflation-min">Limite inferiore (%)</label>
                                <input id="inflation-min" class="input" type="number" step="0.5" bind:value={inflationMinClipPercent} />
                            </div>
                            <div class="form-group">
                                <label class="label" for="inflation-max">Limite superiore (%)</label>
                                <input id="inflation-max" class="input" type="number" step="0.5" bind:value={inflationMaxClipPercent} />
                            </div>
                        </div>
                    {/if}
                {/if}
            </details>

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
    /* Phase 11+ — Excel template / import buttons */
    .excel-tools {
        display: flex;
        gap: 0.5rem;
        margin: 0.5rem 0 0.75rem;
        flex-wrap: wrap;
    }
    .import-btn {
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    .import-btn input[type="file"] {
        position: absolute;
        inset: 0;
        opacity: 0;
        cursor: pointer;
    }
    .success-text { color: var(--color-success, #198754); }
    .error-text   { color: var(--color-danger,  #dc3545); }

    /* ── Phase 11 — optional advanced blocks (collapsed by default) ──────── */
    .adv-block {
        margin-top: 1rem;
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: var(--radius-sm, 6px);
        padding: 0.75rem 1rem;
        background: var(--color-bg-secondary, #f8f9fa);
    }
    .adv-block summary {
        cursor: pointer;
        font-weight: 600;
        color: var(--color-text, #1f2937);
        margin: -0.25rem 0;
        padding: 0.25rem 0;
    }
    .adv-block[open] summary {
        margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border, #e2e8f0);
        padding-bottom: 0.5rem;
    }

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
    .header-text {
        flex: 1 1 360px;
        min-width: 0;
    }
    .header-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .header-actions .select {
        width: auto;
        min-width: 220px;
        max-width: 260px;
        flex: 0 1 auto;
    }
    .header-message {
        display: block;
        margin: 0 0 1rem auto;
        width: fit-content;
    }
    .badge.error {
        background: var(--color-danger, #dc3545);
        color: #fff;
    }
    .badge.success {
        background: var(--color-success, #198754);
        color: #fff;
    }

    /* ── Phase 14 — Luogo step add-from-map sub-flow ────────────────────── */
    .info-box.error {
        border-color: var(--color-danger, #dc3545);
        background: var(--color-danger-bg, #fde8ea);
        color: var(--color-danger, #b3261e);
    }
    .info-box.error p {
        margin: 0;
        font-weight: 500;
    }
    .link-btn {
        background: none;
        border: none;
        padding: 0;
        margin: 0;
        font-size: 0.9rem;
        color: var(--color-primary, #2563eb);
        cursor: pointer;
        text-align: left;
    }
    .link-btn:hover {
        text-decoration: underline;
    }
    .step-subtitle {
        margin: 0 0 0.5rem;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .form-grid {
        display: grid;
        gap: 0.75rem;
        margin: 0.5rem 0;
    }
    .form-grid.two-cols {
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    /* Phase 15 — thermal calibration toggle */
    .checkbox-label {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        cursor: pointer;
        font-size: 0.92rem;
    }
    .checkbox-label input[type="checkbox"] {
        margin-top: 0.2rem;
    }
    /* Phase 16 — detailed electrical model accordion. */
    .section-divider {
        border: 0;
        border-top: 1px solid var(--color-border, #e2e8f0);
        margin: 1.5rem 0;
    }
    .toggle-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
        font-size: 0.95rem;
    }
    .toggle-label { font-weight: 500; }
    .electrical-section,
    .thermal-section {
        padding: 0.75rem 1rem;
        background-color: var(--color-bg-secondary, #f8f9fb);
        border: 1px solid var(--color-border, #e2e8f0);
        border-radius: 6px;
        margin-top: 0.5rem;
    }
</style>
