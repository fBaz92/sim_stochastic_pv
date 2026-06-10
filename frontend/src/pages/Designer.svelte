<script>
    /**
     * Designer — "Progettazione": reactive electrical sizing sheet.
     *
     * The page behaves like the reference spreadsheet, but live: every
     * input change posts the full design to /api/designs/evaluate and
     * repaints the derived cells (admissible string range, voltage and
     * current checks with margins, plant sizing, temperature margins,
     * fuse sizing, cable comparison table). Two things a static sheet
     * cannot do are wired on top: the hourly Monte Carlo production
     * preview (clipping, I² cable losses, MPPT/temperature derating on
     * the site's stochastic weather) and saving the result as a
     * *detailed* plant design ready for the economic analysis.
     */
    import { onMount } from "svelte";
    import { api } from "../api.js";

    // ── Catalogues ──────────────────────────────────────────────────────
    let panels = [];
    let inverters = [];
    let locations = [];
    let cables = [];
    let loadError = "";

    // ── Component selection + editable datasheet ────────────────────────
    let selectedPanelId = "";
    let selectedInverterId = "";
    let panel = {
        power_w: 505, v_oc_stc_v: 40.14, v_mpp_stc_v: 33.9,
        i_sc_stc_a: 15.88, i_mpp_stc_a: 14.9,
        beta_voc_pct_per_c: -0.25, gamma_pmax_pct_per_c: -0.29,
        alpha_isc_pct_per_c: 0.045, v_system_max_v: 1500,
        max_series_fuse_a: 30, noct_c: 45, n_cells_series: 108,
    };
    let inverter = {
        p_ac_nom_kw: 3.0, efficiency_max: 0.972,
        v_dc_max_v: 600, v_dc_min_v: 90,
        v_mppt_min_v: 90, v_mppt_max_v: 580,
        v_mppt_full_load_min_v: 160, v_mppt_full_load_max_v: 520,
        n_mppt_trackers: 2, i_dc_max_per_mppt_a: 12,
        i_sc_max_per_mppt_a: 15, max_strings_per_mppt: 1,
    };
    let showPanelSheet = false;
    let showInverterSheet = false;

    // ── Site ────────────────────────────────────────────────────────────
    let selectedLocationId = "";
    let tMin = -10;
    let tMax = 40;
    let deltaTCell = 30;
    let siteHint = "";

    // ── Requirements ────────────────────────────────────────────────────
    let pAcRequired = 3.0;
    let targetRatio = 1.2;
    let nPanelsPerString = 6;
    let showAdvanced = false;
    let safetyFactor = 1.25;
    let maxLossPct = 1.0; // percent in the UI, fraction to the API
    let fuseMin = 1.5;
    let fuseMax = 2.4;

    // ── Cable run ───────────────────────────────────────────────────────
    let cableLength = 30;
    let cableTemp = 70;
    let chosenSection = null; // user-picked section for preview/save

    // ── Evaluation state ────────────────────────────────────────────────
    let result = null;
    let evalError = "";
    let evaluating = false;
    let debounceTimer = null;

    // ── Production preview state ────────────────────────────────────────
    let preview = null;
    let previewLoading = false;
    let previewError = "";

    // ── Save state ──────────────────────────────────────────────────────
    let designName = "";
    let totalCost = 9000;
    let storageKwh = 0;
    let saveMsg = "";
    let saveError = "";

    function numbersOf(obj) {
        const out = {};
        for (const [k, v] of Object.entries(obj)) {
            out[k] = v === "" || v === null ? null : Number(v);
        }
        return out;
    }

    function buildEvaluatePayload() {
        return {
            panel: numbersOf(panel),
            inverter: numbersOf(inverter),
            site: {
                t_min_c: Number(tMin),
                t_max_c: Number(tMax),
                delta_t_cell_c: Number(deltaTCell),
            },
            requirements: {
                p_ac_required_kw: Number(pAcRequired),
                target_dc_ac_ratio: Number(targetRatio),
                n_panels_per_string: Number(nPanelsPerString),
                safety_factor_isc: Number(safetyFactor),
                max_cable_loss_fraction: Number(maxLossPct) / 100,
                fuse_factor_min: Number(fuseMin),
                fuse_factor_max: Number(fuseMax),
            },
            cable: {
                length_one_way_m: Number(cableLength),
                operating_temperature_c: Number(cableTemp),
                sections: cables.length
                    ? cables.map((c) => ({
                          section_mm2: c.section_mm2,
                          price_eur_per_m: c.price_eur_per_m,
                          iz_a: c.iz_a,
                      }))
                    : null,
            },
        };
    }

    async function evaluateNow() {
        evaluating = true;
        evalError = "";
        try {
            result = await api.evaluateDesign(buildEvaluatePayload());
            // Keep the chosen section valid; default to the recommendation.
            const sections = result.cables.rows.map((r) => r.section_mm2);
            if (chosenSection == null || !sections.includes(chosenSection)) {
                chosenSection = result.cables.recommended_section_mm2;
            }
        } catch (e) {
            evalError = e.message || "Errore di valutazione.";
            result = null;
        } finally {
            evaluating = false;
        }
    }

    function scheduleEvaluate() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(evaluateNow, 300);
    }

    // Reactive trigger: any change of the serialized inputs re-evaluates.
    $: inputsKey = JSON.stringify([
        panel, inverter, tMin, tMax, deltaTCell, pAcRequired, targetRatio,
        nPanelsPerString, safetyFactor, maxLossPct, fuseMin, fuseMax,
        cableLength, cableTemp, cables.length,
    ]);
    let lastKey = "";
    $: if (inputsKey !== lastKey) {
        lastKey = inputsKey;
        scheduleEvaluate();
        preview = null; // any input change invalidates the MC preview
    }

    // Constrained dropdown options for moduli-per-stringa.
    $: nOptions = result && result.bounds.feasible
        ? Array.from(
              { length: result.bounds.n_max - result.bounds.n_min + 1 },
              (_, i) => result.bounds.n_min + i,
          )
        : [Number(nPanelsPerString)];

    function applyPanel() {
        const p = panels.find((x) => String(x.id) === String(selectedPanelId));
        if (!p) return;
        const s = p.specs || {};
        panel = {
            power_w: s.power_w ?? p.power_w,
            v_oc_stc_v: s.v_oc_stc_v ?? null,
            v_mpp_stc_v: s.v_mpp_stc_v ?? null,
            i_sc_stc_a: s.i_sc_stc_a ?? null,
            i_mpp_stc_a: s.i_mpp_stc_a ?? null,
            beta_voc_pct_per_c: s.beta_voc_pct_per_c ?? null,
            gamma_pmax_pct_per_c: s.gamma_pmax_pct_per_c ?? null,
            alpha_isc_pct_per_c: s.alpha_isc_pct_per_c ?? null,
            v_system_max_v: s.v_system_max_v ?? null,
            max_series_fuse_a: s.max_series_fuse_a ?? null,
            noct_c: s.noct_c ?? null,
            n_cells_series: s.n_cells_series ?? null,
        };
    }

    function applyInverter() {
        const inv = inverters.find(
            (x) => String(x.id) === String(selectedInverterId),
        );
        if (!inv) return;
        const s = inv.specs || {};
        inverter = {
            p_ac_nom_kw: s.p_ac_max_kw ?? inv.nominal_power_kw,
            efficiency_max: s.efficiency_max ?? null,
            v_dc_max_v: s.v_dc_max_v ?? null,
            v_dc_min_v: s.v_dc_min_v ?? null,
            v_mppt_min_v: s.v_mppt_min_v ?? null,
            v_mppt_max_v: s.v_mppt_max_v ?? null,
            v_mppt_full_load_min_v: s.v_mppt_full_load_min_v ?? null,
            v_mppt_full_load_max_v: s.v_mppt_full_load_max_v ?? null,
            n_mppt_trackers: s.n_mppt_trackers ?? 1,
            i_dc_max_per_mppt_a: s.i_dc_max_per_mppt_a ?? null,
            i_sc_max_per_mppt_a: s.i_sc_max_per_mppt_a ?? null,
            max_strings_per_mppt: s.max_strings_per_mppt ?? 1,
        };
        if (!pAcRequired || Number(pAcRequired) <= 0) {
            pAcRequired = inverter.p_ac_nom_kw;
        }
    }

    $: selectedLocation = locations.find(
        (l) => String(l.id) === String(selectedLocationId),
    );

    /**
     * Propose Tmin/Tmax from the site's calibrated climate model: the
     * extremes of the simulated hourly temperatures (50 paths × 1 year).
     */
    async function applySiteTemperatures() {
        siteHint = "";
        // Local lookup: inside an on:change handler the reactive
        // `selectedLocation` has not been recomputed yet.
        const loc = locations.find(
            (l) => String(l.id) === String(selectedLocationId),
        );
        const climateId = loc?.climate_profiles?.[0]?.id;
        if (!climateId) {
            siteHint = "La posizione non ha un clima calibrato: Tmin/Tmax manuali.";
            return;
        }
        try {
            const prev = await api.previewClimateProfileById(climateId, {
                n_paths: 50, n_years: 1, seed: 42,
            });
            tMin = Math.round(Math.min(...prev.monthly_min_c));
            tMax = Math.round(Math.max(...prev.monthly_max_c));
            siteHint = `Tmin/Tmax proposti dal clima stocastico del sito (50 path).`;
        } catch (e) {
            siteHint = "Anteprima clima non disponibile: " + e.message;
        }
    }

    async function runPreview() {
        if (!result) return;
        const solarId = selectedLocation?.solar_profiles?.[0]?.id;
        if (!solarId) {
            previewError =
                "Seleziona una posizione con profilo solare per la produzione attesa.";
            return;
        }
        previewLoading = true;
        previewError = "";
        try {
            const climateId = selectedLocation?.climate_profiles?.[0]?.id ?? null;
            preview = await api.productionPreview({
                panel: numbersOf(panel),
                inverter: numbersOf(inverter),
                n_panels_per_string: Number(nPanelsPerString),
                n_strings: result.plant.n_strings,
                solar_profile_id: solarId,
                climate_profile_id: climateId,
                use_electrical_model: climateId != null,
                cable: chosenSection
                    ? {
                          section_mm2: chosenSection,
                          length_one_way_m: Number(cableLength),
                          operating_temperature_c: Number(cableTemp),
                      }
                    : null,
                n_paths: 30,
                seed: 42,
            });
        } catch (e) {
            previewError = e.message || "Errore nella produzione attesa.";
            preview = null;
        } finally {
            previewLoading = false;
        }
    }

    async function saveDesign() {
        saveMsg = "";
        saveError = "";
        if (!designName.trim()) {
            saveError = "Dai un nome al progetto.";
            return;
        }
        if (!result) {
            saveError = "Attendi la valutazione del design.";
            return;
        }
        try {
            const data = {
                p_ac_kw: Number(inverter.p_ac_nom_kw),
                p_dc_kwp: result.plant.p_dc_installed_kwp,
                total_cost_eur: Number(totalCost),
                designer: {
                    panel: numbersOf(panel),
                    inverter: numbersOf(inverter),
                    site: {
                        t_min_c: Number(tMin),
                        t_max_c: Number(tMax),
                        delta_t_cell_c: Number(deltaTCell),
                    },
                    n_panels_per_string: Number(nPanelsPerString),
                    n_strings: result.plant.n_strings,
                    total_panels: result.plant.total_panels,
                    dc_ac_ratio: result.plant.dc_ac_ratio,
                    cable_section_mm2: chosenSection,
                    cable_length_one_way_m: Number(cableLength),
                    recommended_fuse_a: result.protection.recommended_fuse_a,
                    all_checks_ok: result.all_checks_ok,
                },
            };
            if (Number(storageKwh) > 0) data.storage_kwh = Number(storageKwh);
            const payload = {
                name: designName.trim(),
                design_level: "detailed",
                data,
            };
            if (selectedLocationId) payload.location_id = Number(selectedLocationId);
            const record = await api.upsertDesign(payload);
            saveMsg = `Impianto "${record.name}" salvato: lo trovi in Database → Impianti e nella pagina Offerta per l'analisi economica.`;
        } catch (e) {
            saveError = e.message || "Errore nel salvataggio.";
        }
    }

    onMount(async () => {
        try {
            const [p, inv, locs, cab] = await Promise.all([
                api.listPanels(),
                api.listInverters(),
                api.listLocations(),
                api.listCables(),
            ]);
            panels = p;
            inverters = inv;
            locations = locs;
            cables = cab;
        } catch (e) {
            loadError = "Errore nel caricamento dei cataloghi: " + e.message;
        }
        evaluateNow();
    });

    const fmt = (v, digits = 1) =>
        v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);

    // Indicative energy value used for the cable-upgrade payback column
    // (the full economics live in the Confronto/Analisi pages).
    const PAYBACK_PRICE_EUR_PER_KWH = 0.25;

    /**
     * Years to repay the copper upgrade from `row` to the recommended ★
     * section, valuing the avoided ohmic loss at the indicative price.
     * Requires the MC preview (for the annual DC energy) and a priced
     * catalogue; null hides the cell.
     */
    function upgradePaybackYears(row) {
        if (!preview || !result?.cables?.recommended_section_mm2) return null;
        const rec = result.cables.rows.find(
            (r) => r.section_mm2 === result.cables.recommended_section_mm2,
        );
        if (!rec || row.section_mm2 >= rec.section_mm2) return null;
        if (row.cost_total_eur == null || rec.cost_total_eur == null) return null;
        const dCost = rec.cost_total_eur - row.cost_total_eur;
        const dLossKwh =
            (row.loss_fraction_of_dc - rec.loss_fraction_of_dc) *
            preview.annual_dc_kwh_mean;
        if (dCost <= 0 || dLossKwh <= 0) return null;
        return dCost / (dLossKwh * PAYBACK_PRICE_EUR_PER_KWH);
    }
</script>

<div class="page">
    <h1 class="page-title">Progettazione</h1>
    <p class="hint">
        Dimensionamento elettrico di dettaglio: stringhe, verifiche di
        tensione e corrente, margini di temperatura, fusibili e cavi DC.
        Ogni cella derivata si ricalcola a ogni modifica; la produzione
        attesa usa il meteo stocastico del sito (clipping e perdite
        orarie reali).
    </p>

    {#if loadError}<p class="error">{loadError}</p>{/if}

    {#if result}
        <div class="verdict-bar" class:ok={result.all_checks_ok} class:ko={!result.all_checks_ok}>
            {#if result.all_checks_ok}
                ✔ Tutte le verifiche superate
            {:else}
                ✖ Verifiche non superate — controlla i semafori rossi
            {/if}
            {#if evaluating}<span class="evaluating">ricalcolo…</span>{/if}
        </div>
    {/if}
    {#if evalError}<p class="error">{evalError}</p>{/if}

    <div class="grid">
        <!-- ── INPUT COLUMN ─────────────────────────────────────────── -->
        <div class="col">
            <div class="card">
                <h3>Componenti</h3>
                <div class="form-group">
                    <label class="label" for="sel-panel">Modulo fotovoltaico</label>
                    <select id="sel-panel" class="input" bind:value={selectedPanelId} on:change={applyPanel}>
                        <option value="">— Datasheet manuale —</option>
                        {#each panels as p}
                            <option value={String(p.id)}>{p.name}</option>
                        {/each}
                    </select>
                    <button type="button" class="link-btn" on:click={() => (showPanelSheet = !showPanelSheet)}>
                        {showPanelSheet ? "Nascondi" : "Mostra"} datasheet modulo
                    </button>
                </div>
                {#if showPanelSheet}
                    <div class="sheet">
                        <label>Pmax (W)<input class="input" type="number" step="5" bind:value={panel.power_w} /></label>
                        <label>Voc STC (V)<input class="input" type="number" step="0.01" bind:value={panel.v_oc_stc_v} /></label>
                        <label>Vmp STC (V)<input class="input" type="number" step="0.01" bind:value={panel.v_mpp_stc_v} /></label>
                        <label>Isc STC (A)<input class="input" type="number" step="0.01" bind:value={panel.i_sc_stc_a} /></label>
                        <label>Imp STC (A)<input class="input" type="number" step="0.01" bind:value={panel.i_mpp_stc_a} /></label>
                        <label>β Voc (%/°C)<input class="input" type="number" step="0.01" bind:value={panel.beta_voc_pct_per_c} /></label>
                        <label>γ Pmax (%/°C)<input class="input" type="number" step="0.01" bind:value={panel.gamma_pmax_pct_per_c} /></label>
                        <label>α Isc (%/°C)<input class="input" type="number" step="0.005" bind:value={panel.alpha_isc_pct_per_c} /></label>
                        <label>V sistema (V)<input class="input" type="number" step="100" bind:value={panel.v_system_max_v} /></label>
                        <label>Max fusibile (A)<input class="input" type="number" step="1" bind:value={panel.max_series_fuse_a} /></label>
                        <label>NOCT (°C)<input class="input" type="number" step="0.5" bind:value={panel.noct_c} /></label>
                        <label>Celle in serie<input class="input" type="number" step="1" bind:value={panel.n_cells_series} /></label>
                    </div>
                {/if}
                <div class="form-group">
                    <label class="label" for="sel-inverter">Inverter</label>
                    <select id="sel-inverter" class="input" bind:value={selectedInverterId} on:change={applyInverter}>
                        <option value="">— Datasheet manuale —</option>
                        {#each inverters as inv}
                            <option value={String(inv.id)}>{inv.name}</option>
                        {/each}
                    </select>
                    <button type="button" class="link-btn" on:click={() => (showInverterSheet = !showInverterSheet)}>
                        {showInverterSheet ? "Nascondi" : "Mostra"} datasheet inverter
                    </button>
                </div>
                {#if showInverterSheet}
                    <div class="sheet">
                        <label>P AC nom (kW)<input class="input" type="number" step="0.1" bind:value={inverter.p_ac_nom_kw} /></label>
                        <label>Rendimento max<input class="input" type="number" step="0.001" bind:value={inverter.efficiency_max} /></label>
                        <label>Vdc max (V)<input class="input" type="number" step="10" bind:value={inverter.v_dc_max_v} /></label>
                        <label>Vdc min (V)<input class="input" type="number" step="10" bind:value={inverter.v_dc_min_v} /></label>
                        <label>MPPT min (V)<input class="input" type="number" step="10" bind:value={inverter.v_mppt_min_v} /></label>
                        <label>MPPT max (V)<input class="input" type="number" step="10" bind:value={inverter.v_mppt_max_v} /></label>
                        <label>Pieno carico min (V)<input class="input" type="number" step="10" bind:value={inverter.v_mppt_full_load_min_v} /></label>
                        <label>Pieno carico max (V)<input class="input" type="number" step="10" bind:value={inverter.v_mppt_full_load_max_v} /></label>
                        <label>N. MPPT<input class="input" type="number" step="1" min="1" bind:value={inverter.n_mppt_trackers} /></label>
                        <label>I max op/MPPT (A)<input class="input" type="number" step="0.5" bind:value={inverter.i_dc_max_per_mppt_a} /></label>
                        <label>Isc max/MPPT (A)<input class="input" type="number" step="0.5" bind:value={inverter.i_sc_max_per_mppt_a} /></label>
                        <label>Stringhe/MPPT<input class="input" type="number" step="1" min="1" bind:value={inverter.max_strings_per_mppt} /></label>
                    </div>
                {/if}
            </div>

            <div class="card">
                <h3>Sito</h3>
                <div class="form-group">
                    <label class="label" for="sel-location">Posizione</label>
                    <select id="sel-location" class="input" bind:value={selectedLocationId} on:change={applySiteTemperatures}>
                        <option value="">— Nessuna (Tmin/Tmax manuali) —</option>
                        {#each locations as l}
                            <option value={String(l.id)}>{l.name}</option>
                        {/each}
                    </select>
                    {#if siteHint}<p class="hint small">{siteHint}</p>{/if}
                </div>
                <div class="sheet three">
                    <label>T min sito (°C)<input class="input" type="number" step="1" bind:value={tMin} /></label>
                    <label>T max sito (°C)<input class="input" type="number" step="1" bind:value={tMax} /></label>
                    <label>ΔT cella (°C)<input class="input" type="number" step="1" bind:value={deltaTCell} /></label>
                </div>
            </div>

            <div class="card">
                <h3>Requisiti</h3>
                <div class="sheet three">
                    <label>P AC richiesta (kW)<input class="input" type="number" step="0.5" bind:value={pAcRequired} /></label>
                    <label>DC/AC target<input class="input" type="number" step="0.05" bind:value={targetRatio} /></label>
                    <label>Moduli per stringa
                        <select class="input" bind:value={nPanelsPerString}>
                            {#each nOptions as n}
                                <option value={n}>{n}</option>
                            {/each}
                        </select>
                    </label>
                </div>
                {#if result}
                    <p class="hint small">
                        Range ammissibile: {result.bounds.n_min}–{result.bounds.n_max}
                        (Voc fredda ≤ {fmt(result.bounds.v_limit_v, 0)} V
                        → max {result.bounds.n_max_voc}; inseguimento MPPT
                        → max {result.bounds.n_max_mppt}).
                        {#if !result.bounds.feasible}
                            <strong class="ko-text">Accoppiamento non fattibile.</strong>
                        {/if}
                    </p>
                {/if}
                <button type="button" class="link-btn" on:click={() => (showAdvanced = !showAdvanced)}>
                    {showAdvanced ? "Nascondi" : "Mostra"} parametri di verifica
                </button>
                {#if showAdvanced}
                    <div class="sheet">
                        <label>Fattore sicurezza Isc<input class="input" type="number" step="0.05" bind:value={safetyFactor} /></label>
                        <label>Perdita max cavi (%)<input class="input" type="number" step="0.1" bind:value={maxLossPct} /></label>
                        <label>Fusibile min × Isc<input class="input" type="number" step="0.1" bind:value={fuseMin} /></label>
                        <label>Fusibile max × Isc<input class="input" type="number" step="0.1" bind:value={fuseMax} /></label>
                    </div>
                {/if}
                <div class="sheet">
                    <label>Lunghezza cavo (m, sola andata)<input class="input" type="number" step="1" bind:value={cableLength} /></label>
                    <label>T operativa cavo (°C)<input class="input" type="number" step="5" bind:value={cableTemp} /></label>
                </div>
            </div>
        </div>

        <!-- ── RESULTS COLUMN ───────────────────────────────────────── -->
        <div class="col">
            {#if result}
                <div class="card">
                    <h3>Taglia impianto</h3>
                    <table class="kv">
                        <tbody>
                            <tr><td>Potenza DC target</td><td>{fmt(result.plant.p_dc_target_kwp, 2)} kWp</td></tr>
                            <tr><td>Stringhe ({nPanelsPerString} moduli)</td><td>{result.plant.n_strings} × {fmt(result.plant.string_power_kwp, 2)} kWp</td></tr>
                            <tr><td>Moduli totali</td><td>{result.plant.total_panels}</td></tr>
                            <tr><td>Potenza DC installata</td><td><strong>{fmt(result.plant.p_dc_installed_kwp, 2)} kWp</strong></td></tr>
                            <tr><td>Rapporto DC/AC effettivo</td><td>{fmt(result.plant.dc_ac_ratio, 2)}</td></tr>
                        </tbody>
                    </table>
                </div>

                <div class="card">
                    <h3>Verifiche</h3>
                    <ul class="checks">
                        <li class:ok={result.voltages.n_in_range} class:ko={!result.voltages.n_in_range}>
                            Moduli per stringa nel range ammissibile
                        </li>
                        <li class:ok={result.voltages.v_oc_margin_v >= 0} class:ko={result.voltages.v_oc_margin_v < 0}>
                            Voc stringa a freddo {fmt(result.voltages.v_oc_string_cold_v)} V
                            (margine {fmt(result.voltages.v_oc_margin_v)} V)
                        </li>
                        <li class:ok={result.voltages.v_mp_hot_margin_v >= 0} class:ko={result.voltages.v_mp_hot_margin_v < 0}>
                            Vmp stringa a caldo {fmt(result.voltages.v_mp_string_hot_v)} V
                            (margine {fmt(result.voltages.v_mp_hot_margin_v)} V)
                        </li>
                        <li class:ok={result.voltages.v_mp_cold_margin_v >= 0} class:ko={result.voltages.v_mp_cold_margin_v < 0}>
                            Vmp stringa a freddo {fmt(result.voltages.v_mp_string_cold_v)} V
                            (margine inseguimento {fmt(result.voltages.v_mp_cold_margin_v)} V)
                        </li>
                        <li class:ok={result.currents.inputs_ok} class:ko={!result.currents.inputs_ok}>
                            Ingressi fisici: {result.currents.strings_per_mppt} stringa/e per MPPT
                        </li>
                        <li class:ok={result.currents.i_operating_margin_a >= 0} class:ko={result.currents.i_operating_margin_a < 0}>
                            Corrente operativa {fmt(result.currents.i_operating_a)} A per MPPT
                            (margine {fmt(result.currents.i_operating_margin_a)} A)
                        </li>
                        <li class:ok={result.currents.i_sc_margin_a >= 0} class:ko={result.currents.i_sc_margin_a < 0}>
                            Corrente di cortocircuito {fmt(result.currents.i_sc_a)} A per MPPT
                            (margine {fmt(result.currents.i_sc_margin_a)} A)
                        </li>
                        <li class:ok={result.margins.robust} class:ko={!result.margins.robust}>
                            Margini di temperatura: freddo {fmt(result.margins.margin_cold_c, 0)} °C,
                            caldo {fmt(result.margins.margin_hot_c, 1)} °C
                        </li>
                    </ul>
                    <p class="hint small">
                        T ambiente massima ammissibile {fmt(result.margins.t_amb_max_admissible_c, 1)} °C;
                        sotto {fmt(result.margins.t_min_admissible_c, 0)} °C la Voc supererebbe il limite.
                    </p>
                </div>

                <div class="card">
                    <h3>Protezioni di stringa</h3>
                    <p class="meta">
                        {result.protection.protection_required
                            ? "Richieste (≥ 3 stringhe in parallelo, CEI EN 62548)."
                            : "Non richieste (< 3 stringhe in parallelo)."}
                        Finestra fusibile {fmt(result.protection.i_fuse_min_a)}–{fmt(result.protection.i_fuse_max_norm_a)} A
                        {#if result.protection.recommended_fuse_a}
                            → consigliato <strong>{result.protection.recommended_fuse_a} A gPV</strong>
                            {#if result.protection.fuse_within_module_limit === false}
                                <span class="ko-text">(supera il max del modulo!)</span>
                            {/if}
                        {/if}
                    </p>
                </div>
            {/if}
        </div>
    </div>

    {#if result}
        <div class="card">
            <h3>Cavi DC — confronto sezioni</h3>
            <table class="cable-table">
                <thead>
                    <tr>
                        <th></th><th>Sezione</th><th>ΔV</th><th>Caduta</th>
                        <th>Perdita totale</th><th>% su DC</th><th>Costo rame</th>
                        <th>Iz</th><th>Esito</th>
                        {#if preview}<th title="In quanti anni l'upgrade alla sezione ★ si ripaga con le perdite evitate (a {PAYBACK_PRICE_EUR_PER_KWH} €/kWh)">Upgrade a ★</th>{/if}
                    </tr>
                </thead>
                <tbody>
                    {#each result.cables.rows as row (row.section_mm2)}
                        <tr
                            class:recommended={row.section_mm2 === result.cables.recommended_section_mm2}
                            class:chosen={row.section_mm2 === chosenSection}
                        >
                            <td><input type="radio" name="section" value={row.section_mm2} bind:group={chosenSection} /></td>
                            <td>{row.section_mm2} mm²{row.section_mm2 === result.cables.recommended_section_mm2 ? " ★" : ""}</td>
                            <td>{fmt(row.voltage_drop_v, 2)} V</td>
                            <td>{fmt(row.voltage_drop_fraction * 100, 2)}%</td>
                            <td>{fmt(row.loss_total_kw * 1000, 0)} W</td>
                            <td>{fmt(row.loss_fraction_of_dc * 100, 2)}%</td>
                            <td>{row.cost_total_eur != null ? fmt(row.cost_total_eur, 0) + " €" : "—"}</td>
                            <td>{row.iz_a != null ? fmt(row.iz_a, 0) + " A" : "—"}</td>
                            <td class:ok-text={row.loss_ok && row.iz_ok !== false} class:ko-text={!row.loss_ok || row.iz_ok === false}>
                                {row.loss_ok && row.iz_ok !== false ? "OK" : (row.iz_ok === false ? "Iz insuff." : "Oltre soglia")}
                            </td>
                            {#if preview}
                                {@const pb = upgradePaybackYears(row)}
                                <td>{pb != null ? fmt(pb, 1) + " anni" : "—"}</td>
                            {/if}
                        </tr>
                    {/each}
                </tbody>
            </table>
            <p class="hint small">
                ★ = sezione minima che rispetta la soglia di perdita
                ({fmt(Number(maxLossPct), 1)}%) e la portata Iz ≥ Isc di progetto
                ({fmt(result.corrected.i_sc_design_a)} A). Il pallino seleziona la
                sezione usata per produzione attesa e salvataggio.
            </p>
        </div>

        <div class="bottom-grid">
            <div class="card">
                <h3>Produzione attesa (Monte Carlo orario)</h3>
                <p class="hint small">
                    Meteo stocastico del sito, derating MPPT/temperatura
                    {selectedLocation?.climate_profiles?.length ? "(clima calibrato attivo)" : "(senza clima: solo clipping e cavi)"},
                    perdite cavo ∝ I², clipping all'AC nominale.
                </p>
                <button class="btn btn-primary" on:click={runPreview} disabled={previewLoading || !selectedLocation}>
                    {previewLoading ? "Simulazione…" : "Calcola produzione attesa"}
                </button>
                {#if !selectedLocation}<p class="hint small">Seleziona prima una posizione.</p>{/if}
                {#if previewError}<p class="error">{previewError}</p>{/if}
                {#if preview}
                    <table class="kv">
                        <tbody>
                            <tr><td>Energia AC attesa</td>
                                <td><strong>{fmt(preview.annual_ac_kwh_mean, 0)} kWh/anno</strong>
                                    [{fmt(preview.annual_ac_kwh_p05, 0)}–{fmt(preview.annual_ac_kwh_p95, 0)}]</td></tr>
                            <tr><td>Energia DC lorda</td><td>{fmt(preview.annual_dc_kwh_mean, 0)} kWh/anno</td></tr>
                            <tr><td>Clipping inverter</td>
                                <td>{fmt(preview.clipping_kwh_mean, 0)} kWh ({fmt(preview.clipping_fraction * 100, 1)}%)</td></tr>
                            <tr><td>Perdite cavi DC</td>
                                <td>{fmt(preview.cable_loss_kwh_mean, 0)} kWh ({fmt(preview.cable_loss_fraction * 100, 2)}%)</td></tr>
                            {#if preview.electrical_derating_kwh_mean > 0}
                                <tr><td>Derating elettrico (MPPT + T)</td>
                                    <td>{fmt(preview.electrical_derating_kwh_mean, 0)} kWh
                                        · {fmt(preview.hours_outside_mppt_per_year_mean, 0)} h/anno fuori MPPT</td></tr>
                            {/if}
                        </tbody>
                    </table>
                {/if}
            </div>

            <div class="card">
                <h3>Salva come impianto</h3>
                <div class="form-group">
                    <label class="label" for="design-name">Nome progetto *</label>
                    <input id="design-name" class="input" bind:value={designName} placeholder='es. "Casa Pavullo 6 kWp"' />
                </div>
                <div class="sheet">
                    <label>Costo totale (€)<input class="input" type="number" step="100" bind:value={totalCost} /></label>
                    <label>Accumulo (kWh, 0 = no)<input class="input" type="number" step="0.5" bind:value={storageKwh} /></label>
                </div>
                <button class="btn btn-primary" on:click={saveDesign}>Salva impianto (dettagliato)</button>
                {#if saveError}<p class="error">{saveError}</p>{/if}
                {#if saveMsg}<p class="success">{saveMsg}</p>{/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .page { max-width: 1100px; margin: 0 auto; }
    .hint { color: var(--color-text-secondary); font-size: 0.9rem; max-width: 75ch; }
    .hint.small { font-size: 0.82rem; margin-top: 0.4rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 1rem; }
    .bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 1.25rem; }
    @media (max-width: 980px) { .grid, .bottom-grid { grid-template-columns: 1fr; } }
    .col { display: flex; flex-direction: column; gap: 1.25rem; }
    .card { padding: 1.1rem 1.25rem; }
    .card h3 { margin: 0 0 0.75rem; }
    .sheet {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem 1rem;
        margin: 0.5rem 0;
    }
    .sheet.three { grid-template-columns: 1fr 1fr 1fr; }
    .sheet label {
        display: flex; flex-direction: column; gap: 0.2rem;
        font-size: 0.8rem; color: var(--color-text-secondary);
    }
    .link-btn {
        background: none; border: none; padding: 0;
        color: var(--color-primary, #1d4ed8); cursor: pointer;
        font-size: 0.82rem; text-decoration: underline;
    }
    .verdict-bar {
        margin: 1rem 0 0.5rem; padding: 0.6rem 1rem; border-radius: 8px;
        font-weight: 600; border: 1px solid;
    }
    .verdict-bar.ok { color: var(--color-success, #28a745); border-color: var(--color-success, #28a745); }
    .verdict-bar.ko { color: var(--color-danger, #dc3545); border-color: var(--color-danger, #dc3545); }
    .evaluating { font-weight: 400; font-size: 0.82rem; margin-left: 0.75rem; color: var(--color-text-secondary); }
    .checks { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.35rem; }
    .checks li { font-size: 0.88rem; padding-left: 1.4rem; position: relative; }
    .checks li::before { position: absolute; left: 0; }
    .checks li.ok::before { content: "🟢"; }
    .checks li.ko::before { content: "🔴"; }
    .checks li.ko { color: var(--color-danger, #dc3545); }
    .kv { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .kv td { padding: 0.25rem 0.4rem; border-bottom: 1px solid var(--color-border, #eee); }
    .kv td:last-child { text-align: right; }
    .cable-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .cable-table th, .cable-table td { padding: 0.35rem 0.5rem; text-align: right; border-bottom: 1px solid var(--color-border, #eee); }
    .cable-table th:nth-child(2), .cable-table td:nth-child(2) { text-align: left; }
    .cable-table tr.recommended { background: rgba(40, 167, 69, 0.08); }
    .cable-table tr.chosen td:nth-child(2) { font-weight: 600; }
    .ok-text { color: var(--color-success, #28a745); }
    .ko-text { color: var(--color-danger, #dc3545); }
    .meta { color: var(--color-text-secondary); font-size: 0.88rem; }
    .form-group { margin-bottom: 0.6rem; }
    .error { color: var(--color-danger, #dc3545); margin-top: 0.6rem; }
    .success { color: var(--color-success, #28a745); margin-top: 0.6rem; }
</style>
