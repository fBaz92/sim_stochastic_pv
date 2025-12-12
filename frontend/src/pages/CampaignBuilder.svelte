<script>
    import { onMount } from "svelte";
    import { api } from "../api";

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

    // Configuration - Full scenario structure required by backend
    let config = {
        panel_counts: "1, 2, 3",
        battery_counts: "0, 1, 2",
        include_no_battery: true,
        scenario_name: "Optimization Campaign",
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
            const campaignData = {
                ...JSON.parse(JSON.stringify(config)),
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
            message = "Campaign saved!";
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

            message = `Campaign '${saved.name}' executed successfully! Evaluations: ${result.evaluations}`;
        } catch (e) {
            console.error(e);
            message = "Error running saved campaign: " + e.message;
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
            message = `Loaded campaign: ${saved.name}`;
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
                    economic: config.economic,
                    load_profile: config.load_profile,
                    solar: config.solar,
                    energy: config.energy,
                    price: config.price,
                },
            };

            const res = await api.triggerOptimization(payload);
            message = `Optimization Started! Evaluations: ${res.evaluations}`;
        } catch (e) {
            console.error(e);
            message = "Error: " + e.message;
        } finally {
            running = false;
        }
    }

    onMount(loadHardware);
</script>

<div class="container">
    <div class="header">
        <h1>Campaign Builder</h1>
        <div class="header-actions">
            <div class="load-group">
                <select class="select sm" bind:value={selectedSavedCampaignId}>
                    <option value="">Load saved campaign...</option>
                    {#each savedCampaigns as s}
                        <option value={String(s.id)}>{s.name}</option>
                    {/each}
                </select>
                <button
                    class="btn btn-outline btn-sm"
                    on:click={handleLoadCampaign}
                    disabled={!selectedSavedCampaignId}>Load</button
                >
                <button
                    class="btn btn-primary btn-sm"
                    on:click={handleRunSavedCampaign}
                    disabled={!selectedSavedCampaignId}
                    title="Run this campaign directly from the database (uses current hardware specs)"
                    >Run Saved</button
                >
            </div>
            <button
                class="btn btn-outline btn-sm"
                on:click={() => (showSaveModal = true)}>Save Current</button
            >
        </div>
        {#if message}
            <div
                class={`badge ${message.includes("Error") ? "error" : "success"}`}
            >
                {message}
            </div>
        {/if}
    </div>

    {#if showSaveModal}
        <div class="modal-backdrop">
            <div class="modal card">
                <h3>Save Campaign</h3>
                <div class="form-group">
                    <label class="label" for="campaign-save-name">Campaign Name</label>
                    <input id="campaign-save-name" class="input" bind:value={newCampaignName} />
                </div>
                <div class="modal-actions">
                    <button
                        class="btn btn-text"
                        on:click={() => (showSaveModal = false)}>Cancel</button
                    >
                    <button
                        class="btn btn-primary"
                        on:click={handleSaveCampaign}>Save</button
                    >
                </div>
            </div>
        </div>
    {/if}

    {#if loadingParams}
        <p>Loading database...</p>
    {:else}
        <div class="grid-2col">
            <div class="card section">
                <h2>1. Select Hardware Scope</h2>

                <div class="selection-group">
                    <h3>Inverters</h3>
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
                    <h3>Panels</h3>
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
                    <h3>Batteries</h3>
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
                <h2>2. Configuration</h2>

                <div class="form-group">
                    <label class="label" for="campaign-config-name">Campaign Name</label>
                    <input id="campaign-config-name" class="input" bind:value={config.scenario_name} />
                </div>

                <div class="form-group">
                    <label class="label" for="campaign-panel-counts">Panel Counts (comma separated)</label>
                    <input
                        id="campaign-panel-counts"
                        class="input"
                        bind:value={config.panel_counts}
                        placeholder="e.g. 6, 8, 10, 12"
                    />
                    <small class="text-meta"
                        >Number of panels per string/inverter</small
                    >
                </div>

                <div class="form-group">
                    <label class="label" for="campaign-battery-counts"
                        >Battery Counts (comma separated)</label
                    >
                    <input
                        id="campaign-battery-counts"
                        class="input"
                        bind:value={config.battery_counts}
                        placeholder="e.g. 1, 2"
                    />
                </div>

                <div class="form-group">
                    <label class="checkbox-label">
                        <input
                            type="checkbox"
                            bind:checked={config.include_no_battery}
                        />
                        Include "No Battery" scenario?
                    </label>
                </div>

                <div class="divider"></div>

                <div class="form-group">
                    <label class="label" for="campaign-mc-runs">Monte Carlo Runs (per scenario)</label>
                    <input
                        id="campaign-mc-runs"
                        class="input"
                        type="number"
                        bind:value={config.economic.n_mc}
                    />
                </div>

                <div class="actions">
                    <button
                        class="btn btn-primary btn-lg"
                        on:click={handleRun}
                        disabled={running}
                    >
                        {running ? "Running..." : "Run Optimization"}
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>
