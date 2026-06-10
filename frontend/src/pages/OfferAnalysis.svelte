<script>
    /**
     * OfferAnalysis — one-page flow to evaluate a received commercial
     * offer ("Analizza un'offerta").
     *
     * The user enters the offer's nameplate data (AC power, optional DC
     * power, optional storage, turn-key cost, incentive) plus the
     * installation site and the economic context, in a single form. On
     * submit the page saves the offer as an *essential* plant design
     * (upsert by name) and launches a Monte Carlo analysis whose
     * scenario references it via `plant_design_id` — the backend expands
     * the design into the simulator configuration. The floating
     * JobProgress widget tracks the run and opens the Dashboard when
     * done.
     */
    import { onMount } from "svelte";
    import { api } from "../api.js";

    // ── Offer fields ────────────────────────────────────────────────────
    let offerName = "";
    let pAcKw = 6.0;
    let pDcKwp = null; // null → "= P_AC"
    let storageKwh = 0;
    let totalCostEur = 14000;
    let taxBonusEnabled = true;
    let taxBonusPercent = 50;
    let taxBonusYears = 10;
    let notes = "";

    // ── Context ─────────────────────────────────────────────────────────
    let locations = [];
    let selectedLocationId = "";
    let loadProfiles = [];
    let selectedLoadProfileId = ""; // "" → standard ARERA inline profile
    let priceProfiles = [];
    let selectedPriceProfileId = ""; // "" → default inline price model
    let nYears = 20;
    let nMc = 200;

    // ── Saved designs (reload an offer) ─────────────────────────────────
    let designs = [];
    let selectedDesignId = "";

    let loading = false;
    let submitting = false;
    let message = "";
    let error = "";

    async function loadAll() {
        loading = true;
        try {
            const [locs, loads, prices, des] = await Promise.all([
                api.listLocations(),
                api.listLoadProfiles(),
                api.listPriceProfiles(),
                api.listDesigns(),
            ]);
            locations = locs;
            loadProfiles = loads;
            priceProfiles = prices;
            designs = des.filter((d) => d.design_level === "essential");
        } catch (e) {
            error = "Errore nel caricamento dei dati: " + e.message;
        } finally {
            loading = false;
        }
    }

    // Only sites with a downloaded solar profile can feed the simulation.
    $: usableLocations = locations.filter((l) => l.solar_profiles.length > 0);
    $: selectedLocation = usableLocations.find(
        (l) => String(l.id) === String(selectedLocationId),
    );

    function loadDesign() {
        const d = designs.find((x) => String(x.id) === String(selectedDesignId));
        if (!d) return;
        offerName = d.name;
        pAcKw = d.data.p_ac_kw;
        pDcKwp = d.data.p_dc_kwp ?? null;
        storageKwh = d.data.storage_kwh ?? 0;
        totalCostEur = d.data.total_cost_eur;
        notes = d.data.notes ?? "";
        if (d.data.tax_bonus) {
            taxBonusEnabled = !!d.data.tax_bonus.enabled;
            taxBonusPercent = (d.data.tax_bonus.fraction_of_investment ?? 0.5) * 100;
            taxBonusYears = d.data.tax_bonus.duration_years ?? 10;
        } else {
            taxBonusEnabled = false;
        }
        if (d.location_id != null) selectedLocationId = String(d.location_id);
    }

    function buildDesignPayload() {
        const data = {
            p_ac_kw: Number(pAcKw),
            total_cost_eur: Number(totalCostEur),
        };
        if (pDcKwp != null && pDcKwp !== "") data.p_dc_kwp = Number(pDcKwp);
        if (Number(storageKwh) > 0) data.storage_kwh = Number(storageKwh);
        if (taxBonusEnabled) {
            data.tax_bonus = {
                enabled: true,
                fraction_of_investment: Number(taxBonusPercent) / 100,
                duration_years: Number(taxBonusYears),
            };
        }
        if (notes.trim()) data.notes = notes.trim();
        const payload = {
            name: offerName.trim(),
            design_level: "essential",
            data,
        };
        if (selectedLocationId) payload.location_id = Number(selectedLocationId);
        return payload;
    }

    async function saveDesign() {
        error = "";
        message = "";
        if (!offerName.trim()) {
            error = "Dai un nome all'offerta (es. \"Offerta Rossi 6 kW\").";
            return null;
        }
        try {
            const record = await api.upsertDesign(buildDesignPayload());
            designs = (await api.listDesigns()).filter(
                (d) => d.design_level === "essential",
            );
            selectedDesignId = String(record.id);
            return record;
        } catch (e) {
            error = "Errore nel salvataggio dell'impianto: " + e.message;
            return null;
        }
    }

    async function saveOnly() {
        const record = await saveDesign();
        if (record) message = `Impianto "${record.name}" salvato.`;
    }

    async function saveAndAnalyze() {
        if (!selectedLocationId) {
            error = "Seleziona la posizione di installazione (serve il profilo solare del sito).";
            return;
        }
        submitting = true;
        try {
            const record = await saveDesign();
            if (!record) return;

            const scenario = {
                scenario_name: `Offerta — ${record.name}`,
                plant_design_id: record.id,
                energy: { n_years: Number(nYears) },
                economic: { n_mc: Number(nMc) },
            };
            if (selectedLoadProfileId) {
                scenario.load_profile_id = Number(selectedLoadProfileId);
            } else {
                // Standard Italian residential baseline (ARERA), occupied
                // all year: enough for a first verdict on the offer.
                scenario.load_profile = {
                    kind: "home_away",
                    home: { type: "arera" },
                    away: { type: "arera" },
                };
            }
            if (selectedPriceProfileId) {
                scenario.price_profile_id = Number(selectedPriceProfileId);
            } else {
                // Default market context: mild stochastic escalation around
                // today's typical residential price.
                scenario.price = {
                    base_price_eur_per_kwh: 0.25,
                    annual_escalation: 0.03,
                    use_stochastic_escalation: true,
                };
            }

            await api.submitAnalysisJob({ n_mc: Number(nMc), scenario });
            message =
                "Analisi avviata: al termine si apre la dashboard con il verdetto sull'offerta.";
        } catch (e) {
            error = "Errore nell'avvio dell'analisi: " + e.message;
        } finally {
            submitting = false;
        }
    }

    onMount(loadAll);
</script>

<div class="page">
    <h1 class="page-title">Analizza un'offerta</h1>
    <p class="hint">
        Hai ricevuto un preventivo? Bastano i dati di targa: potenza,
        accumulo, costo e incentivo. L'offerta viene salvata come
        <strong>impianto</strong> riusabile e valutata con il Monte Carlo
        (sole, consumo e prezzi stocastici del tuo sito).
    </p>

    {#if loading}
        <p>Caricamento…</p>
    {:else}
        {#if designs.length > 0}
            <div class="card reload-card">
                <label class="label" for="saved-design">Riparti da un'offerta salvata</label>
                <div class="reload-row">
                    <select id="saved-design" class="input" bind:value={selectedDesignId}>
                        <option value="">— Nuova offerta —</option>
                        {#each designs as d}
                            <option value={String(d.id)}>{d.name}</option>
                        {/each}
                    </select>
                    <button class="btn btn-ghost" on:click={loadDesign} disabled={!selectedDesignId}>
                        Carica
                    </button>
                </div>
            </div>
        {/if}

        <div class="grid">
            <div class="card">
                <h3>L'offerta</h3>
                <div class="form-group">
                    <label class="label" for="offer-name">Nome offerta *</label>
                    <input id="offer-name" class="input" bind:value={offerName}
                        placeholder='es. "Offerta Rossi 6 kW"' />
                </div>
                <div class="grid-2col">
                    <div class="form-group">
                        <label class="label" for="p-ac">Potenza AC (kW) *</label>
                        <input id="p-ac" class="input" type="number" step="0.1" min="0.5"
                            bind:value={pAcKw} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="p-dc">Potenza DC (kWp)</label>
                        <input id="p-dc" class="input" type="number" step="0.1" min="0"
                            bind:value={pDcKwp} placeholder="= potenza AC" />
                        <p class="hint small">Lascia vuoto se il preventivo non la indica.</p>
                    </div>
                </div>
                <div class="grid-2col">
                    <div class="form-group">
                        <label class="label" for="storage">Accumulo (kWh)</label>
                        <input id="storage" class="input" type="number" step="0.5" min="0"
                            bind:value={storageKwh} />
                        <p class="hint small">0 = nessuna batteria.</p>
                    </div>
                    <div class="form-group">
                        <label class="label" for="cost">Costo chiavi in mano (€) *</label>
                        <input id="cost" class="input" type="number" step="100" min="1"
                            bind:value={totalCostEur} />
                    </div>
                </div>
                <label class="checkbox-label">
                    <input type="checkbox" bind:checked={taxBonusEnabled} />
                    Detrazione fiscale
                </label>
                {#if taxBonusEnabled}
                    <div class="grid-2col">
                        <div class="form-group">
                            <label class="label" for="bonus-pct">Quota detratta (%)</label>
                            <input id="bonus-pct" class="input" type="number" step="5" min="1" max="100"
                                bind:value={taxBonusPercent} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="bonus-years">In quanti anni</label>
                            <input id="bonus-years" class="input" type="number" step="1" min="1" max="30"
                                bind:value={taxBonusYears} />
                        </div>
                    </div>
                {/if}
                <div class="form-group">
                    <label class="label" for="offer-notes">Note</label>
                    <textarea id="offer-notes" class="input" rows="2" bind:value={notes}
                        placeholder="es. installatore, scadenza dell'offerta…"></textarea>
                </div>
            </div>

            <div class="card">
                <h3>Il contesto</h3>
                <div class="form-group">
                    <label class="label" for="location">Posizione di installazione *</label>
                    <select id="location" class="input" bind:value={selectedLocationId}>
                        <option value="">— Seleziona —</option>
                        {#each usableLocations as l}
                            <option value={String(l.id)}>{l.name}</option>
                        {/each}
                    </select>
                    {#if usableLocations.length === 0}
                        <p class="hint small warn">
                            Nessuna posizione con profilo solare: aggiungine una in
                            <a href="#/database">Database → Posizioni</a>.
                        </p>
                    {:else if selectedLocation}
                        <p class="hint small">
                            ☀ profilo solare del sito
                            {#if selectedLocation.climate_profiles.length > 0}
                                · 🌡 clima stocastico calibrato
                            {/if}
                        </p>
                    {/if}
                </div>
                <div class="form-group">
                    <label class="label" for="load">Profilo di consumo</label>
                    <select id="load" class="input" bind:value={selectedLoadProfileId}>
                        <option value="">Standard ARERA (residenziale tipico)</option>
                        {#each loadProfiles as p}
                            <option value={String(p.id)}>{p.name}</option>
                        {/each}
                    </select>
                    <p class="hint small">
                        Per un consumo su misura crea un profilo in
                        <a href="#/database">Database → Profili di carico</a>.
                    </p>
                </div>
                <div class="form-group">
                    <label class="label" for="price">Prezzo dell'energia</label>
                    <select id="price" class="input" bind:value={selectedPriceProfileId}>
                        <option value="">Predefinito (0,25 €/kWh, +3%/anno stocastico)</option>
                        {#each priceProfiles as p}
                            <option value={String(p.id)}>{p.name}</option>
                        {/each}
                    </select>
                </div>
                <div class="grid-2col">
                    <div class="form-group">
                        <label class="label" for="n-years">Orizzonte (anni)</label>
                        <input id="n-years" class="input" type="number" step="1" min="1" max="40"
                            bind:value={nYears} />
                    </div>
                    <div class="form-group">
                        <label class="label" for="n-mc">Simulazioni Monte Carlo</label>
                        <input id="n-mc" class="input" type="number" step="50" min="10" max="2000"
                            bind:value={nMc} />
                    </div>
                </div>
            </div>
        </div>

        {#if error}<p class="error">{error}</p>{/if}
        {#if message}<p class="success">{message}</p>{/if}

        <div class="form-actions">
            <button class="btn btn-ghost" on:click={saveOnly} disabled={submitting}>
                Salva soltanto
            </button>
            <button class="btn btn-primary" on:click={saveAndAnalyze} disabled={submitting}>
                {submitting ? "Avvio analisi…" : "Salva e analizza"}
            </button>
        </div>
    {/if}
</div>

<style>
    .page { max-width: 980px; margin: 0 auto; }
    .hint { color: var(--color-text-secondary); font-size: 0.9rem; max-width: 70ch; }
    .hint.small { font-size: 0.82rem; margin-top: 0.25rem; }
    .hint.small.warn { color: var(--color-warning-text, #b8860b); }
    .grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 1rem;
    }
    @media (max-width: 900px) {
        .grid { grid-template-columns: 1fr; }
    }
    .card { padding: 1.25rem; }
    .card h3 { margin-top: 0; }
    .grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .checkbox-label { display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0; }
    .reload-card { padding: 1rem 1.25rem; margin-bottom: 1rem; }
    .reload-row { display: flex; gap: 0.5rem; }
    .reload-row select { flex: 1; }
    .form-actions {
        margin-top: 1.25rem;
        display: flex;
        gap: 0.75rem;
        justify-content: flex-end;
    }
    textarea.input { resize: vertical; }
    .error { color: var(--color-danger, #dc3545); margin-top: 1rem; }
    .success { color: var(--color-success, #28a745); margin-top: 1rem; }
</style>
