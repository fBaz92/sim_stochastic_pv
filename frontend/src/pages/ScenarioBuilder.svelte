<script>
    import { onMount } from "svelte";
    import { api } from "../api";
    import MonthInput from "../components/forms/MonthInput.svelte";
    import MonthlyProfileEditor from "../components/forms/MonthlyProfileEditor.svelte";

    let loading = false;
    let message = "";

    // Hardware Lists
    let inverters = [];
    let batteries = [];
    // let panels = []; // Assuming we might want to pick panels later, but currently ScenarioBuilder uses simple pv_kwp

    // Selection state
    let selectedInverterIndex = -1;
    let selectedBatteryIndex = -1;

    // Track selected hardware IDs for database-driven workflow
    let selectedInverterId = null;
    let selectedBatteryId = null;

    // Scenario Management
    let savedScenarios = [];
    let selectedSavedScenarioId = "";
    let newScenarioName = "My Scenario";
    let showSaveModal = false;

    // Default Scenario State
    let scenario = {
        load_profile: {
            home_profile_type: "arera",
            home_profiles_w: Array(12).fill(300), // Custom Monthly
            home_profiles_24h: Array.from({ length: 12 }, () =>
                Array(24).fill(200),
            ), // Custom 24h (Monthly)
            away_profile: "arera",
            away_profiles_w: Array(12).fill(100),
            away_profiles_24h: Array.from({ length: 12 }, () =>
                Array(24).fill(100),
            ),
            min_days_home: Array(12).fill(25),
        },
        solar: {
            pv_kwp: 3.0,
            degradation_per_year: 0.007,
        },
        energy: {
            n_years: 20,
            battery_specs: {
                capacity_kwh: 5.0,
                cycles_life: 5000,
            },
            n_batteries: 0,
            inverter_p_ac_max_kw: 3.0,
        },
        price: {
            base_price_eur_per_kwh: 0.25,
            annual_escalation: 0.02,
            use_stochastic_escalation: true,
            escalation_variation_percentiles: [-0.05, 0.05],
        },
        economic: {
            investment_eur: 6000,
            n_mc: 100,
        },
    };

    // Load hardware on mount
    onMount(async () => {
        try {
            const [inv, bat, configs] = await Promise.all([
                api.listInverters(),
                api.listBatteries(),
                api.listConfigurations("scenario"),
            ]);
            inverters = inv;
            batteries = bat;
            savedScenarios = configs;
        } catch (e) {
            console.error(e);
        }
    });

    // Cost Calculation Logic - Removed as per new code
    // $: {
    //     if (!manualCostOverride) {
    //         let total = 0;
    //         // Inverter Cost (using price_eur if available, else placeholder)
    //         if (selectedInverterIndex >= 0) {
    //             // Inverter price usually ~150-200 EUR/kW? Or use DB field
    //             // Since DB doesn't have price field exposed in schema list_inverters,
    //             // we assume user must input or we guess.
    //             // Wait, InverterResponse doesn't have price.
    //             // So we can't auto-calc cost accurately without that data.
    //             // Retain manual input or assume standard costs.
    //         }
    //         // Panels: ~1000 EUR/kWp installed?
    //         total += scenario.solar.pv_kwp * 1000;

    //         // Batteries
    //         if (scenario.energy.n_batteries > 0) {
    //             // ~400 EUR/kWh?
    //             total +=
    //                 scenario.energy.battery_specs.capacity_kwh *
    //                 scenario.energy.n_batteries *
    //                 400;
    //         }

    //         // scenario.economic.investment_eur = total; // Disabled auto-calc for now as it overrides user input too aggressively without real data
    //     }
    // }

    function onInverterSelect() {
        if (selectedInverterIndex >= 0) {
            const inv = inverters[selectedInverterIndex];
            selectedInverterId = inv.id; // Track the ID for DB-driven workflow
            scenario = {
                ...scenario,
                energy: {
                    ...scenario.energy,
                    inverter_p_ac_max_kw:
                        inv.p_ac_max_kw || inv.nominal_power_kw,
                },
            };
        }
    }

    function onBatterySelect() {
        if (selectedBatteryIndex >= 0) {
            const bat = batteries[selectedBatteryIndex];
            selectedBatteryId = bat.id; // Track the ID for DB-driven workflow
            const batterySpecs = {
                ...scenario.energy.battery_specs,
                capacity_kwh: bat.capacity_kwh,
                cycles_life:
                    bat.specs?.cycles_life ??
                    scenario.energy.battery_specs.cycles_life,
            };
            scenario = {
                ...scenario,
                energy: {
                    ...scenario.energy,
                    battery_specs: batterySpecs,
                    n_batteries:
                        scenario.energy.n_batteries === 0
                            ? 1
                            : scenario.energy.n_batteries,
                },
            };
        }
    }

    async function handleRun() {
        loading = true;
        message = "";
        try {
            // Clone configuration
            // Note: $state objects can be cloned via structuredClone or JSON parse/stringify
            const finalScenario = JSON.parse(JSON.stringify(scenario));

            // Map the selected profile data to the 'home_profiles_w' field expected by backend
            if (finalScenario.load_profile.home_profile_type === "custom_24h") {
                finalScenario.load_profile.home_profile_type = "custom";
                // Backend expects (12, 24) array or (12,) or (24,).
                // Our new home_profiles_24h is (12, 24).
                finalScenario.load_profile.home_profiles_w =
                    scenario.load_profile.home_profiles_24h;
            }
            if (finalScenario.load_profile.away_profile === "custom_24h") {
                finalScenario.load_profile.away_profile = "custom";
                finalScenario.load_profile.away_profiles_w =
                    scenario.load_profile.away_profiles_24h;
            }

            const payload = {
                n_mc: scenario.economic.n_mc,
                scenario: finalScenario,
            };

            const res = await api.triggerAnalysis(payload);
            message = `Analysis Complete! Mean Gain: €${res.final_gain_mean_eur.toFixed(0)}`;
            // We could trigger a dashboard refresh or redirect?
            // window.location.hash = '/';
        } catch (e) {
            console.error(e);
            message = "Error: " + e.message;
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
            if (!saved) {
                throw new Error("Scenario not found");
            }
            const savedScenario = saved.data?.scenario ?? saved.data;
            scenario = JSON.parse(JSON.stringify(savedScenario));
            message = `Scenario '${saved.name}' loaded successfully.`;
            // Reset selected hardware indices as they might not match the loaded scenario
            selectedInverterIndex = -1;
            selectedBatteryIndex = -1;
        } catch (e) {
            console.error(e);
            message = "Error loading scenario: " + e.message;
        } finally {
            loading = false;
        }
    }

    async function handleRunSavedScenario() {
        if (!selectedSavedScenarioId) return;
        loading = true;
        message = "";
        try {
            const targetId = Number(selectedSavedScenarioId);
            const saved = savedScenarios.find((s) => s.id === targetId);
            if (!saved) {
                throw new Error("Scenario not found");
            }

            // Run the saved scenario directly using the DB-driven endpoint
            // This will hydrate hardware IDs from the database automatically
            const result = await api.runSavedScenario(targetId, {
                seed: scenario.economic.seed,
                n_mc: scenario.economic.n_mc,
            });

            message = `Scenario '${saved.name}' executed successfully! Final gain (mean): €${result.final_gain_mean_eur?.toFixed(2) || "N/A"}`;
        } catch (e) {
            console.error(e);
            message = "Error running saved scenario: " + e.message;
        } finally {
            loading = false;
        }
    }

    async function handleSaveScenario() {
        loading = true;
        message = "";
        try {
            // Build scenario data with hardware IDs instead of embedded values
            const scenarioData = JSON.parse(JSON.stringify(scenario));

            // Add hardware IDs to the scenario data for database-driven workflow
            if (selectedInverterId) {
                scenarioData.inverter_id = selectedInverterId;
            }
            if (selectedBatteryId) {
                scenarioData.battery_id = selectedBatteryId;
            }

            const payload = {
                name: newScenarioName,
                config_type: "scenario",
                data: scenarioData,
            };
            const savedConfig = await api.createConfiguration(payload);
            savedScenarios = await api.listConfigurations("scenario");
            selectedSavedScenarioId = String(savedConfig.id);
            message = `Scenario '${newScenarioName}' saved successfully!`;
            showSaveModal = false;
        } catch (e) {
            console.error(e);
            message = "Error saving scenario: " + e.message;
        } finally {
            loading = false;
        }
    }
</script>

<div class="container">
    <div class="header">
        <h1 class="page-title">Scenario Builder</h1>
        <div class="header-actions">
            <div
                class="load-group"
                style="display: flex; gap: 0.5rem; align-items: center;"
            >
                <select
                    class="select sm"
                    bind:value={selectedSavedScenarioId}
                    style="min-width: 200px;"
                >
                    <option value="">Load saved scenario...</option>
                    {#each savedScenarios as s}
                        <option value={String(s.id)}>{s.name}</option>
                    {/each}
                </select>
                <button
                    class="btn btn-outline btn-sm"
                    on:click={handleLoadScenario}
                    disabled={!selectedSavedScenarioId}>Load</button
                >
                <button
                    class="btn btn-primary btn-sm"
                    on:click={handleRunSavedScenario}
                    disabled={!selectedSavedScenarioId}
                    title="Run this scenario directly from the database (uses current hardware specs)"
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
                <h3>Save Scenario</h3>
                <div class="form-group">
                    <label class="label" for="scenario-save-name"
                        >Scenario Name</label
                    >
                    <input
                        id="scenario-save-name"
                        class="input"
                        bind:value={newScenarioName}
                    />
                </div>
                <div class="modal-actions">
                    <button
                        class="btn btn-text"
                        on:click={() => (showSaveModal = false)}>Cancel</button
                    >
                    <button
                        class="btn btn-primary"
                        on:click={handleSaveScenario}>Save</button
                    >
                </div>
            </div>
        </div>
    {/if}
    <div class="grid-2col">
        <!-- Column 1 -->
        <div class="col">
            <div class="card section">
                <h2 class="section-title">Load Profiles</h2>

                <!-- Home Profile -->
                <div class="form-group">
                    <label class="label" for="scenario-home-profile-type"
                        >Home Profile Type</label
                    >
                    <select
                        id="scenario-home-profile-type"
                        class="select"
                        bind:value={scenario.load_profile.home_profile_type}
                    >
                        <option value="arera">ARERA (Standard)</option>
                        <option value="custom">Flat Monthly Average (W)</option>
                        <option value="custom_24h"
                            >Custom 24h Profile (W)</option
                        >
                    </select>
                </div>
                {#if scenario.load_profile.home_profile_type === "custom"}
                    <MonthInput
                        label="Monthly Power (W)"
                        bind:values={scenario.load_profile.home_profiles_w}
                    />
                {:else if scenario.load_profile.home_profile_type === "custom_24h"}
                    <MonthlyProfileEditor
                        label="Home 24h Profile (W)"
                        bind:values={scenario.load_profile.home_profiles_24h}
                    />
                {/if}

                <!-- Away Profile -->
                {#if scenario.load_profile.home_profile_type !== "arera"}
                    <!-- Show Away profile options only if user might want to customize, or always show?
                     Usually 'Away' is simpler, but user asked for flexibility.
                     Let's keep it visible.
                -->
                {/if}

                <div class="form-group" style="margin-top: 1rem;">
                    <label class="label" for="scenario-away-profile-type"
                        >Away Profile Type</label
                    >
                    <select
                        id="scenario-away-profile-type"
                        class="select"
                        bind:value={scenario.load_profile.away_profile}
                    >
                        <option value="arera">ARERA (Standard)</option>
                        <option value="custom">Flat Monthly Average (W)</option>
                        <option value="custom_24h"
                            >Custom 24h Profile (W)</option
                        >
                    </select>
                </div>
                {#if scenario.load_profile.away_profile === "custom"}
                    <MonthInput
                        label="Monthly Power (W)"
                        bind:values={scenario.load_profile.away_profiles_w}
                    />
                {:else if scenario.load_profile.away_profile === "custom_24h"}
                    <MonthlyProfileEditor
                        label="Away 24h Profile (W)"
                        bind:values={scenario.load_profile.away_profiles_24h}
                    />
                {/if}

                <div class="divider"></div>
                <MonthInput
                    label="Min Days Home / Month"
                    bind:values={scenario.load_profile.min_days_home}
                />
            </div>

            <div class="card section">
                <h2 class="section-title">Solar System</h2>
                <div class="form-group">
                    <label class="label" for="scenario-pv-kwp"
                        >PV System Size (kWp)</label
                    >
                    <input
                        id="scenario-pv-kwp"
                        class="input"
                        type="number"
                        step="0.1"
                        bind:value={scenario.solar.pv_kwp}
                    />
                </div>
                <div class="form-group">
                    <label class="label" for="scenario-degradation"
                        >Annual Degradation (0-1)</label
                    >
                    <input
                        id="scenario-degradation"
                        class="input"
                        type="number"
                        step="0.001"
                        bind:value={scenario.solar.degradation_per_year}
                    />
                </div>
                <!-- Future: Panel selection -->
            </div>
        </div>

        <!-- Column 2 -->
        <div class="col">
            <div class="card section">
                <h2 class="section-title">Hardware Selection</h2>

                <div class="form-group">
                    <label class="label" for="scenario-inverter-select"
                        >Select Inverter</label
                    >
                    <select
                        id="scenario-inverter-select"
                        class="select"
                        bind:value={selectedInverterIndex}
                        on:change={onInverterSelect}
                    >
                        <option value={-1}>Custom (Enter specs manually)</option
                        >
                        {#each inverters as inv, i}
                            <option value={i}
                                >{inv.name} ({inv.p_ac_max_kw ||
                                    inv.nominal_power_kw} kW)</option
                            >
                        {/each}
                    </select>
                </div>
                <div class="form-group">
                    <label class="label" for="scenario-inverter-power"
                        >Max AC Power (kW)</label
                    >
                    <input
                        id="scenario-inverter-power"
                        class="input"
                        type="number"
                        step="0.1"
                        bind:value={scenario.energy.inverter_p_ac_max_kw}
                    />
                </div>

                <div class="divider"></div>

                <div class="form-group">
                    <label class="label" for="scenario-battery-select"
                        >Select Battery</label
                    >
                    <select
                        id="scenario-battery-select"
                        class="select"
                        bind:value={selectedBatteryIndex}
                        on:change={onBatterySelect}
                    >
                        <option value={-1}>Custom (Enter specs manually)</option
                        >
                        {#each batteries as bat, i}
                            <option value={i}
                                >{bat.name} ({bat.capacity_kwh} kWh)</option
                            >
                        {/each}
                    </select>
                </div>
                <div
                    class="grid-responsive"
                    style="grid-template-columns: 1fr 1fr;"
                >
                    <div class="form-group">
                        <label class="label" for="scenario-battery-count"
                            >Count</label
                        >
                        <input
                            id="scenario-battery-count"
                            class="input"
                            type="number"
                            bind:value={scenario.energy.n_batteries}
                        />
                    </div>
                    {#if scenario.energy.n_batteries > 0}
                        <div class="form-group">
                            <label class="label" for="scenario-battery-capacity"
                                >Capacity (kWh)</label
                            >
                            <input
                                id="scenario-battery-capacity"
                                class="input"
                                type="number"
                                step="0.1"
                                bind:value={
                                    scenario.energy.battery_specs.capacity_kwh
                                }
                            />
                        </div>
                    {/if}
                </div>
            </div>

            <div class="card section">
                <h2 class="section-title">Economics & Price</h2>
                <div class="form-group">
                    <label class="label" for="scenario-base-price"
                        >Base Price (€/kWh)</label
                    >
                    <input
                        id="scenario-base-price"
                        class="input"
                        type="number"
                        step="0.01"
                        bind:value={scenario.price.base_price_eur_per_kwh}
                    />
                </div>
                <div class="form-group">
                    <label class="label" for="scenario-annual-escalation"
                        >Annual Escalation (rate)</label
                    >
                    <input
                        id="scenario-annual-escalation"
                        class="input"
                        type="number"
                        step="0.01"
                        bind:value={scenario.price.annual_escalation}
                    />
                </div>

                <div class="form-group">
                    <label class="checkbox-label">
                        <input
                            type="checkbox"
                            bind:checked={
                                scenario.price.use_stochastic_escalation
                            }
                        />
                        Stochastic Price Variation?
                    </label>
                </div>
                {#if scenario.price.use_stochastic_escalation}
                    <div
                        class="grid-responsive"
                        style="grid-template-columns: 1fr 1fr;"
                    >
                        <div class="form-group">
                            <label class="label" for="scenario-p05-variation"
                                >P05 Variation</label
                            >
                            <input
                                id="scenario-p05-variation"
                                class="input"
                                type="number"
                                step="0.01"
                                bind:value={
                                    scenario.price
                                        .escalation_variation_percentiles[0]
                                }
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="scenario-p95-variation"
                                >P95 Variation</label
                            >
                            <input
                                id="scenario-p95-variation"
                                class="input"
                                type="number"
                                step="0.01"
                                bind:value={
                                    scenario.price
                                        .escalation_variation_percentiles[1]
                                }
                            />
                        </div>
                    </div>
                {/if}

                <div class="divider"></div>

                <div class="form-group">
                    <label class="label" for="scenario-investment"
                        >Initial Investment (€)</label
                    >
                    <input
                        id="scenario-investment"
                        class="input"
                        type="number"
                        step="100"
                        bind:value={scenario.economic.investment_eur}
                    />
                    <small class="text-meta">Override calculated cost</small>
                </div>
                <div class="form-group">
                    <label class="label" for="scenario-mc-runs"
                        >Monte Carlo Runs</label
                    >
                    <input
                        id="scenario-mc-runs"
                        class="input"
                        type="number"
                        step="10"
                        bind:value={scenario.economic.n_mc}
                    />
                </div>
            </div>

            <div
                class="actions-bar"
                style="display: flex; justify-content: flex-end; margin-top: 1rem;"
            >
                <button
                    class="btn btn-primary btn-lg"
                    on:click={handleRun}
                    disabled={loading}
                >
                    {loading ? "Running Simulation..." : "Run Simulation"}
                </button>
            </div>
        </div>
    </div>
</div>
