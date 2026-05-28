<script>
    import { onMount } from "svelte";
    import { get } from "svelte/store";
    import { api } from "../api";
    import { activeJob, pendingConfigurationId } from "../lib/stores";

    let loadingParams = true;
    let running = false;
    let message = "";

    // Available Hardware options
    let inverters = [];
    let panels = [];
    let batteries = [];

    // Selection State
    let selectedInverterIds = new Set();
    let selectedPanelIds = new Set();
    let selectedBatteryIds = new Set();

    // Campaign Management
    let savedCampaigns = [];
    let selectedSavedCampaignId = "";
    let newCampaignName = "My Campaign";
    let showSaveModal = false;

    // Phase 11 — optional tax bonus and inflation overrides applied to
    // every scenario in the campaign sweep. Mirrors the ScenarioBuilder
    // UI: percentages in the UI, fractions in the payload.
    let taxBonusEnabled = false;
    let taxBonusFractionPercent = 50;
    let taxBonusDurationYears = 10;

    let inflationOverride = false;
    let inflationMode = "deterministic";
    let inflationMeanPercent = 2.5;
    let inflationStdPercent = 1.0;
    let inflationMinClipPercent = -2.0;
    let inflationMaxClipPercent = 10.0;

    function buildEconomicBlock() {
        const econ = { ...config.economic };
        if (taxBonusEnabled) {
            econ.tax_bonus = {
                enabled: true,
                fraction_of_investment: taxBonusFractionPercent / 100,
                duration_years: taxBonusDurationYears,
            };
        }
        if (inflationOverride) {
            econ.inflation = {
                mode: inflationMode,
                mean: inflationMeanPercent / 100,
                std: inflationStdPercent / 100,
                min_clip: inflationMinClipPercent / 100,
                max_clip: inflationMaxClipPercent / 100,
            };
        }
        return econ;
    }

    // Configuration - Full scenario structure required by backend
    let config = {
        panel_counts: "1, 2, 3",
        battery_counts: "0, 1, 2",
        include_no_battery: true,
        scenario_name: "Design",
        economic: {
            n_mc: 20,
            investment_eur: 0, // Will be calculated by optimizer
        },
        load_profile: {
            home_profile_type: "arera",
            away_profile: "arera",
            min_days_home: Array(12).fill(25),
        },
        solar: {
            pv_kwp: 3.0,
            degradation_per_year: 0.007,
        },
        energy: {
            n_years: 20,
            pv_kwp: 3.0,
            battery_specs: {
                capacity_kwh: 5.0,
                cycles_life: 5000,
            },
            n_batteries: 0,
            inverter_p_ac_max_kw: 3.0,
        },
        price: {
            base_price_eur_per_kwh: 0.20,
            annual_escalation: 0.02,
            use_stochastic_escalation: true,
            escalation_variation_percentiles: [-0.05, 0.05],
        },
    };

    async function loadHardware() {
        try {
            const [inv, pan, bat, configs] = await Promise.all([
                api.listInverters(),
                api.listPanels(),
                api.listBatteries(),
                api.listConfigurations("campaign"),
            ]);
            inverters = inv;
            panels = pan;
            batteries = bat;
            savedCampaigns = configs;
        } catch (e) {
            console.error(e);
            alert("Failed to load hardware options");
        } finally {
            loadingParams = false;
        }
    }

    async function handleSaveCampaign() {
        if (!newCampaignName) return;
        try {
            // Build campaign data with hardware IDs for database-driven workflow
            const configClone = JSON.parse(JSON.stringify(config));
            // Phase 11 — persist the optional sub-blocks alongside the rest
            // of the economic configuration so reloading the campaign
            // restores them.
            configClone.economic = buildEconomicBlock();
            const campaignData = {
                ...configClone,
                // Store hardware selections as IDs instead of full objects
                hardware_selections: {
                    inverter_ids: [...selectedInverterIds],
                    panel_ids: [...selectedPanelIds],
                    battery_ids: [...selectedBatteryIds],
                },
            };

            const payload = {
                name: newCampaignName,
                config_type: "campaign",
                data: campaignData,
            };
            const saved = await api.createConfiguration(payload);
            savedCampaigns = await api.listConfigurations("campaign");
            selectedSavedCampaignId = String(saved.id);
            message = "Design salvato!";
            showSaveModal = false;
        } catch (e) {
            message = "Error saving: " + e.message;
        }
    }

    async function handleRunSavedCampaign() {
        if (!selectedSavedCampaignId) return;
        running = true;
        message = "";
        try {
            const targetId = Number(selectedSavedCampaignId);
            const saved = savedCampaigns.find((c) => c.id === targetId);
            if (!saved) {
                throw new Error("Campaign not found");
            }

            // Run the saved campaign directly using the DB-driven endpoint
            // This will hydrate hardware IDs from the database automatically
            const result = await api.runSavedCampaign(targetId, {
                seed: config.economic.seed,
                n_mc: config.economic.n_mc,
            });

            message = `Design '${saved.name}' eseguito con successo. Valutazioni: ${result.evaluations}`;
        } catch (e) {
            console.error(e);
            message = "Errore nell'esecuzione del design salvato: " + e.message;
        } finally {
            running = false;
        }
    }

    async function handleLoadCampaign() {
        if (!selectedSavedCampaignId) return;
        const saved = savedCampaigns.find(
            (c) => c.id === Number(selectedSavedCampaignId),
        );
        if (saved && saved.data) {
            if (saved.data.config)
                config = JSON.parse(JSON.stringify(saved.data.config));
            if (saved.data.selections) {
                selectedInverterIds = new Set(saved.data.selections.inverters);
                selectedPanelIds = new Set(saved.data.selections.panels);
                selectedBatteryIds = new Set(saved.data.selections.batteries);
            }
            message = `Design caricato: ${saved.name}`;
        }
    }

    function toggleSelection(set, id) {
        if (set.has(id)) set.delete(id);
        else set.add(id);
        // Force reactivity for Set
        if (set === selectedInverterIds) selectedInverterIds = new Set(set);
        if (set === selectedPanelIds) selectedPanelIds = new Set(set);
        if (set === selectedBatteryIds) selectedBatteryIds = new Set(set);
    }

    async function handleRun() {
        running = true;
        message = "";
        try {
            const inverter_options = inverters.filter((i) =>
                selectedInverterIds.has(i.id),
            );
            const panel_options = panels.filter((p) =>
                selectedPanelIds.has(p.id),
            );
            const battery_options = batteries.filter((b) =>
                selectedBatteryIds.has(b.id),
            );

            if (inverter_options.length === 0 || panel_options.length === 0) {
                throw new Error(
                    "Please select at least one inverter and one panel option.",
                );
            }

            const parseCounts = (str) =>
                str
                    .split(",")
                    .map((s) => parseInt(s.trim()))
                    .filter((n) => !isNaN(n));
            const panel_count_options = parseCounts(config.panel_counts);
            const battery_count_options = parseCounts(config.battery_counts);

            // Build complete scenario structure with all required sections
            const payload = {
                n_mc: config.economic.n_mc,
                scenario: {
                    optimization: {
                        inverter_options,
                        panel_options,
                        battery_options,
                        panel_count_options,
                        battery_count_options,
                        include_no_battery: config.include_no_battery,
                    },
                    scenario_name: config.scenario_name,
                    economic: buildEconomicBlock(),
                    load_profile: config.load_profile,
                    solar: config.solar,
                    energy: config.energy,
                    price: config.price,
                },
            };

            // Phase 12 — submit as a background job. The floating
            // JobProgress widget shows progress (configurations completed
            // out of total) and redirects to the Dashboard when done.
            const { job_id } = await api.submitOptimizationJob(payload);
            const totalConfigs =
                inverter_options.length *
                panel_options.length *
                Math.max(1, battery_options.length) *
                Math.max(1, panel_count_options.length) *
                Math.max(1, battery_count_options.length);
            activeJob.set({
                id: job_id,
                kind: "optimization",
                status: "pending",
                progress_done: 0,
                progress_total: totalConfigs,
                progress_fraction: 0,
                message: "In attesa di avvio...",
                run_id: null,
                error: null,
            });
            message = "Design avviato. Vedi la barra in basso a sinistra per il progresso.";
        } catch (e) {
            console.error(e);
            message = "Errore: " + e.message;
        } finally {
            running = false;
        }
    }

    onMount(async () => {
        await loadHardware();
        const pending = get(pendingConfigurationId);
        if (pending != null) {
            pendingConfigurationId.set(null);
            selectedSavedCampaignId = String(pending);
            await handleLoadCampaign();
        }
    });
</script>

<div class="container">
    <div class="header">
        <div class="header-text">
            <h1>Design — esplorazione economica</h1>
            <p class="page-subtitle">
                Un <strong>Design</strong> è un'esplorazione economica al variare
                delle configurazioni hardware <strong>(CAPEX:</strong> inverter,
                pannelli, batterie<strong>)</strong> e dei parametri operativi
                <strong>(OPEX:</strong> profilo di consumo, prezzo dell'energia,
                inflazione<strong>)</strong>. Per ogni combinazione la Monte
                Carlo gira uno scenario completo e i risultati sono confrontati
                per individuare l'opzione che massimizza il rendimento. Per
                analizzare un <em>singolo</em> impianto già definito usa
                la pagina <a href="#/scenario">Scenario</a>.
            </p>
        </div>
        <div class="header-actions">
            <select class="select sm" bind:value={selectedSavedCampaignId}>
                <option value="">Carica design salvato...</option>
                {#each savedCampaigns as s}
                    <option value={String(s.id)}>{s.name}</option>
                {/each}
            </select>
            <button
                class="btn btn-outline btn-sm"
                on:click={handleLoadCampaign}
                disabled={!selectedSavedCampaignId}>Carica</button
            >
            <button
                class="btn btn-primary btn-sm"
                on:click={handleRunSavedCampaign}
                disabled={!selectedSavedCampaignId}
                title="Esegue questo design dal database con le specifiche hardware correnti"
                >Esegui salvato</button
            >
            <button
                class="btn btn-outline btn-sm"
                on:click={() => (showSaveModal = true)}>Salva corrente</button
            >
        </div>
    </div>
    {#if message}
        <div
            class={`badge ${message.toLowerCase().includes("error") ? "error" : "success"} header-message`}
        >
            {message}
        </div>
    {/if}

    {#if showSaveModal}
        <div class="modal-backdrop">
            <div class="modal card">
                <h3>Salva design</h3>
                <div class="form-group">
                    <label class="label" for="campaign-save-name">Nome del design</label>
                    <input id="campaign-save-name" class="input" bind:value={newCampaignName} />
                </div>
                <div class="modal-actions">
                    <button
                        class="btn btn-text"
                        on:click={() => (showSaveModal = false)}>Annulla</button
                    >
                    <button
                        class="btn btn-primary"
                        on:click={handleSaveCampaign}>Salva</button
                    >
                </div>
            </div>
        </div>
    {/if}

    {#if loadingParams}
        <p>Caricamento database...</p>
    {:else}
        <div class="grid-2col">
            <div class="card section">
                <h2>1. Seleziona hardware</h2>

                <div class="selection-group">
                    <h3>Inverter</h3>
                    <div class="scrollable-list">
                        {#each inverters as item}
                            <label class="checkbox-label">
                                <input
                                    type="checkbox"
                                    checked={selectedInverterIds.has(item.id)}
                                    on:change={() =>
                                        toggleSelection(
                                            selectedInverterIds,
                                            item.id,
                                        )}
                                />
                                <span
                                    >{item.name} ({item.nominal_power_kw ||
                                        item.p_ac_max_kw} kW)</span
                                >
                            </label>
                        {/each}
                    </div>
                </div>

                <div class="selection-group">
                    <h3>Pannelli</h3>
                    <div class="scrollable-list">
                        {#each panels as item}
                            <label class="checkbox-label">
                                <input
                                    type="checkbox"
                                    checked={selectedPanelIds.has(item.id)}
                                    on:change={() =>
                                        toggleSelection(
                                            selectedPanelIds,
                                            item.id,
                                        )}
                                />
                                <span>{item.name} ({item.power_w} W)</span>
                            </label>
                        {/each}
                    </div>
                </div>

                <div class="selection-group">
                    <h3>Batterie</h3>
                    <div class="scrollable-list">
                        {#each batteries as item}
                            <label class="checkbox-label">
                                <input
                                    type="checkbox"
                                    checked={selectedBatteryIds.has(item.id)}
                                    on:change={() =>
                                        toggleSelection(
                                            selectedBatteryIds,
                                            item.id,
                                        )}
                                />
                                <span
                                    >{item.name} ({item.capacity_kwh} kWh)</span
                                >
                            </label>
                        {/each}
                    </div>
                </div>
            </div>

            <div class="card section">
                <h2>2. Configurazione</h2>

                <div class="form-group">
                    <label class="label" for="campaign-config-name">Nome del design</label>
                    <input id="campaign-config-name" class="input" bind:value={config.scenario_name} />
                </div>

                <div class="form-group">
                    <label class="label" for="campaign-panel-counts">Numero pannelli (separati da virgola)</label>
                    <input
                        id="campaign-panel-counts"
                        class="input"
                        bind:value={config.panel_counts}
                        placeholder="es. 6, 8, 10, 12"
                    />
                    <small class="text-meta"
                        >Numero di pannelli per stringa/inverter</small
                    >
                </div>

                <div class="form-group">
                    <label class="label" for="campaign-battery-counts"
                        >Numero batterie (separati da virgola)</label
                    >
                    <input
                        id="campaign-battery-counts"
                        class="input"
                        bind:value={config.battery_counts}
                        placeholder="es. 1, 2"
                    />
                </div>

                <div class="form-group">
                    <label class="checkbox-label">
                        <input
                            type="checkbox"
                            bind:checked={config.include_no_battery}
                        />
                        Includi scenario "senza batteria"
                    </label>
                </div>

                <div class="divider"></div>

                <div class="form-group">
                    <label class="label" for="campaign-mc-runs">Campioni Monte Carlo (per scenario)</label>
                    <input
                        id="campaign-mc-runs"
                        class="input"
                        type="number"
                        bind:value={config.economic.n_mc}
                    />
                </div>

                <!-- Phase 11 — optional tax bonus block ─────────────────── -->
                <details class="adv-block">
                    <summary>Bonus fiscale (opzionale)</summary>
                    <label class="checkbox-label">
                        <input type="checkbox" bind:checked={taxBonusEnabled} />
                        Attiva bonus fiscale (applicato a ogni scenario)
                    </label>
                    {#if taxBonusEnabled}
                        <div class="form-group">
                            <label class="label" for="campaign-bonus-fraction">Percentuale dell'investimento (%)</label>
                            <input id="campaign-bonus-fraction" class="input" type="number" step="1" min="0" max="100" bind:value={taxBonusFractionPercent} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="campaign-bonus-years">Durata (anni)</label>
                            <input id="campaign-bonus-years" class="input" type="number" step="1" min="1" max="20" bind:value={taxBonusDurationYears} />
                        </div>
                    {/if}
                </details>

                <!-- Phase 11 — optional inflation override ──────────────── -->
                <details class="adv-block">
                    <summary>Inflazione (opzionale)</summary>
                    <label class="checkbox-label">
                        <input type="checkbox" bind:checked={inflationOverride} />
                        Personalizza inflazione
                    </label>
                    {#if inflationOverride}
                        <div class="form-group">
                            <label class="label" for="campaign-inflation-mode">Modalità</label>
                            <select id="campaign-inflation-mode" class="input" bind:value={inflationMode}>
                                <option value="deterministic">Deterministica</option>
                                <option value="stochastic">Stocastica (Normale troncata)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="label" for="campaign-inflation-mean">Media annua (%)</label>
                            <input id="campaign-inflation-mean" class="input" type="number" step="0.1" bind:value={inflationMeanPercent} />
                        </div>
                        {#if inflationMode === "stochastic"}
                            <div class="form-group">
                                <label class="label" for="campaign-inflation-std">Deviazione standard (%)</label>
                                <input id="campaign-inflation-std" class="input" type="number" step="0.1" min="0" bind:value={inflationStdPercent} />
                            </div>
                            <div class="form-group">
                                <label class="label" for="campaign-inflation-min">Limite inferiore (%)</label>
                                <input id="campaign-inflation-min" class="input" type="number" step="0.5" bind:value={inflationMinClipPercent} />
                            </div>
                            <div class="form-group">
                                <label class="label" for="campaign-inflation-max">Limite superiore (%)</label>
                                <input id="campaign-inflation-max" class="input" type="number" step="0.5" bind:value={inflationMaxClipPercent} />
                            </div>
                        {/if}
                    {/if}
                </details>

                <div class="actions">
                    <button
                        class="btn btn-primary btn-lg"
                        on:click={handleRun}
                        disabled={running}
                    >
                        {running ? "Esecuzione in corso..." : "Esegui design"}
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .header-text {
        flex: 1 1 360px;
        min-width: 0;
    }
    .page-subtitle {
        color: var(--color-text-secondary);
        margin-top: 0.25rem;
        font-size: 0.95rem;
    }
    .header-message {
        display: block;
        margin: 0 0 1rem auto;
        width: fit-content;
    }
    .header-actions {
        justify-content: flex-end;
    }
    .header-actions .select {
        width: auto;
        min-width: 200px;
        max-width: 240px;
        flex: 0 1 auto;
    }
    .actions {
        margin-top: 1.25rem;
        display: flex;
        justify-content: flex-end;
    }
    /* Phase 11 — optional advanced blocks (collapsed by default) */
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
</style>
