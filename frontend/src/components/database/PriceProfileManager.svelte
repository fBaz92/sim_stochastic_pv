<script>
    import { onMount } from "svelte";
    import { api } from "../../api";
    import ResultsChart from "../ResultsChart.svelte";

    /**
     * Price Profile Manager — Phase 10 edition.
     *
     * Adds live Monte Carlo fan-chart preview to the create/edit form,
     * and a click-to-preview affordance on saved profiles. The user can
     * judge how aggressive a volatility setting is *before* committing to
     * a 20-year economic simulation.
     */

    let items = [];
    let showForm = false;

    // Default state of the form. We model parameters generically so the
    // form can carry all three model_type variants (escalating, gbm,
    // mean_reverting) and we only serialise the relevant subset.
    let newItem = {
        name: "",
        model_type: "gbm",
        base_price_eur_per_kwh: 0.25,

        // Escalating (legacy)
        annual_escalation: 0.025,
        use_stochastic_escalation: true,
        escalation_p05: -0.05,
        escalation_p95: 0.05,

        // GBM
        drift_annual: 0.025,
        volatility_annual: 0.10,

        // Mean reverting
        long_term_price_eur_per_kwh: 0.28,
        mean_reversion_speed_annual: 0.3,
        // (volatility_annual reused)
    };

    // Live preview state for the form
    let formPreview = null;
    let formPreviewLoading = false;
    let formPreviewError = "";
    let formPreviewTimer = null;

    // Saved-profile preview state
    let selectedSavedId = null;
    let savedPreview = null;
    let savedPreviewLoading = false;
    let savedPreviewError = "";

    async function load() {
        items = await api.listPriceProfiles();
    }

    /**
     * Build the JSON ``data`` blob expected by the backend from the form
     * state. Only relevant fields per model_type are emitted to keep the
     * payload tidy and round-trip clean.
     */
    function buildDataPayload() {
        const base = {
            model_type: newItem.model_type,
            base_price_eur_per_kwh: Number(newItem.base_price_eur_per_kwh),
        };
        if (newItem.model_type === "escalating") {
            return {
                ...base,
                annual_escalation: Number(newItem.annual_escalation),
                use_stochastic_escalation: !!newItem.use_stochastic_escalation,
                escalation_variation_percentiles: [
                    Number(newItem.escalation_p05),
                    Number(newItem.escalation_p95),
                ],
            };
        }
        if (newItem.model_type === "gbm") {
            return {
                ...base,
                drift_annual: Number(newItem.drift_annual),
                volatility_annual: Number(newItem.volatility_annual),
            };
        }
        // mean_reverting
        return {
            ...base,
            long_term_price_eur_per_kwh: Number(
                newItem.long_term_price_eur_per_kwh,
            ),
            mean_reversion_speed_annual: Number(
                newItem.mean_reversion_speed_annual,
            ),
            volatility_annual: Number(newItem.volatility_annual),
        };
    }

    async function handleSubmit() {
        await api.createPriceProfile({
            name: newItem.name,
            data: buildDataPayload(),
        });
        showForm = false;
        load();
    }

    onMount(load);

    /**
     * Convert the API preview payload (Phase 3 shape) into a Chart.js dataset.
     * Same visual recipe as Dashboard.getPriceChart:
     *   - p05/p95 band as filled area
     *   - sample paths as thin semi-transparent strokes
     *   - mean as the dominant solid line
     */
    function previewToChartData(p) {
        if (!p) return null;
        const datasets = [
            {
                label: "P05",
                data: p.p05_eur_per_kwh,
                borderColor: "transparent",
                backgroundColor: "rgba(13, 110, 253, 0.18)",
                fill: "+1",
                pointRadius: 0,
                type: "line",
                order: 3,
            },
            {
                label: "P95",
                data: p.p95_eur_per_kwh,
                borderColor: "transparent",
                backgroundColor: "transparent",
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 3,
            },
        ];
        const samples = Array.isArray(p.sample_paths) ? p.sample_paths : [];
        for (let i = 0; i < samples.length; i++) {
            datasets.push({
                label: i === 0 ? "Path simulati" : `_path_${i}`,
                data: samples[i],
                borderColor: "rgba(108, 117, 125, 0.35)",
                backgroundColor: "transparent",
                borderWidth: 1,
                fill: false,
                pointRadius: 0,
                type: "line",
                order: 2,
            });
        }
        datasets.push({
            label: "Prezzo medio (€/kWh)",
            data: p.mean_eur_per_kwh,
            borderColor: "#0d6efd",
            backgroundColor: "#0d6efd",
            borderWidth: 2.5,
            fill: false,
            pointRadius: 0,
            type: "line",
            order: 1,
        });
        return { labels: p.months, datasets };
    }

    const previewChartOptions = {
        plugins: {
            legend: {
                labels: {
                    filter: (item) => !item.text.startsWith("_path_"),
                },
            },
        },
        scales: {
            x: { title: { display: true, text: "Mese" } },
            y: { title: { display: true, text: "EUR/kWh" } },
        },
    };

    /**
     * Debounced live preview for the form. Called whenever a form value
     * changes; waits 500 ms of stillness before hitting the backend.
     */
    async function refreshFormPreview() {
        clearTimeout(formPreviewTimer);
        formPreviewTimer = setTimeout(async () => {
            formPreviewLoading = true;
            formPreviewError = "";
            try {
                const payload = {
                    name: newItem.name || "preview",
                    data: buildDataPayload(),
                };
                formPreview = await api.previewPriceParameters(payload, {
                    n_paths: 150,
                    n_years: 15,
                    seed: 42,
                });
            } catch (e) {
                console.error(e);
                formPreviewError = e.message || "Preview failed";
                formPreview = null;
            } finally {
                formPreviewLoading = false;
            }
        }, 500);
    }

    // Re-trigger live preview whenever ANY form input changes
    $: if (showForm) {
        // Touch each field so Svelte tracks the dependency
        const _ = [
            newItem.model_type,
            newItem.base_price_eur_per_kwh,
            newItem.annual_escalation,
            newItem.use_stochastic_escalation,
            newItem.escalation_p05,
            newItem.escalation_p95,
            newItem.drift_annual,
            newItem.volatility_annual,
            newItem.long_term_price_eur_per_kwh,
            newItem.mean_reversion_speed_annual,
        ];
        refreshFormPreview();
    }

    /**
     * On click of a saved-profile card, fetch its preview from the backend
     * (uses the saved data verbatim — useful even when the form is closed).
     */
    async function selectSaved(item) {
        selectedSavedId = item.id;
        savedPreview = null;
        savedPreviewError = "";
        savedPreviewLoading = true;
        try {
            savedPreview = await api.previewPriceProfileById(item.id, {
                n_paths: 200,
                n_years: 20,
                seed: 42,
            });
        } catch (e) {
            console.error(e);
            savedPreviewError = e.message || "Preview failed";
        } finally {
            savedPreviewLoading = false;
        }
    }
</script>

<div class="manager">
    <div class="toolbar">
        <h2>Price Profiles</h2>
        <button class="btn btn-primary" on:click={() => (showForm = !showForm)}>
            {showForm ? "Cancel" : "Add Profile"}
        </button>
    </div>

    {#if showForm}
        <div class="card form-card">
            <form
                on:submit={(e) => {
                    e.preventDefault();
                    handleSubmit();
                }}
            >
                <div class="form-group">
                    <label class="label" for="price-profile-name">Name</label>
                    <input
                        id="price-profile-name"
                        class="input"
                        bind:value={newItem.name}
                        required
                    />
                </div>

                <div class="group-2col">
                    <div class="form-group">
                        <label class="label" for="price-profile-model"
                            >Modello prezzo</label
                        >
                        <select
                            id="price-profile-model"
                            class="select"
                            bind:value={newItem.model_type}
                        >
                            <option value="escalating"
                                >Escalating (deterministico + jitter)</option
                            >
                            <option value="gbm">GBM (random walk log-prezzo)</option>
                            <option value="mean_reverting"
                                >Mean reverting (Ornstein-Uhlenbeck)</option
                            >
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="label" for="price-profile-base"
                            >Base Price (€/kWh)</label
                        >
                        <input
                            id="price-profile-base"
                            class="input"
                            type="number"
                            step="0.01"
                            bind:value={newItem.base_price_eur_per_kwh}
                            required
                        />
                    </div>
                </div>

                {#if newItem.model_type === "escalating"}
                    <div class="group-2col">
                        <div class="form-group">
                            <label class="label" for="price-profile-escalation"
                                >Annual Escalation</label
                            >
                            <input
                                id="price-profile-escalation"
                                class="input"
                                type="number"
                                step="0.005"
                                bind:value={newItem.annual_escalation}
                                required
                            />
                        </div>
                        <div class="form-group">
                            <label class="checkbox-label">
                                <input
                                    type="checkbox"
                                    bind:checked={newItem.use_stochastic_escalation}
                                />
                                Stochastic jitter
                            </label>
                        </div>
                    </div>
                    {#if newItem.use_stochastic_escalation}
                        <div class="group-2col">
                            <div class="form-group">
                                <label class="label" for="price-profile-p05"
                                    >P05 (negative)</label
                                >
                                <input
                                    id="price-profile-p05"
                                    class="input"
                                    type="number"
                                    step="0.01"
                                    bind:value={newItem.escalation_p05}
                                />
                            </div>
                            <div class="form-group">
                                <label class="label" for="price-profile-p95"
                                    >P95 (positive)</label
                                >
                                <input
                                    id="price-profile-p95"
                                    class="input"
                                    type="number"
                                    step="0.01"
                                    bind:value={newItem.escalation_p95}
                                />
                            </div>
                        </div>
                    {/if}
                {:else if newItem.model_type === "gbm"}
                    <div class="group-2col">
                        <div class="form-group">
                            <label class="label" for="price-profile-drift"
                                >Drift annuo (μ)</label
                            >
                            <input
                                id="price-profile-drift"
                                class="input"
                                type="number"
                                step="0.005"
                                bind:value={newItem.drift_annual}
                                required
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="price-profile-vol"
                                >Volatilità annua (σ)</label
                            >
                            <input
                                id="price-profile-vol"
                                class="input"
                                type="number"
                                step="0.01"
                                min="0"
                                bind:value={newItem.volatility_annual}
                                required
                            />
                        </div>
                    </div>
                {:else}
                    <div class="group-2col">
                        <div class="form-group">
                            <label class="label" for="price-profile-ltp"
                                >Prezzo lungo periodo (€/kWh)</label
                            >
                            <input
                                id="price-profile-ltp"
                                class="input"
                                type="number"
                                step="0.01"
                                bind:value={newItem.long_term_price_eur_per_kwh}
                                required
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="price-profile-kappa"
                                >Mean reversion κ (1/anno)</label
                            >
                            <input
                                id="price-profile-kappa"
                                class="input"
                                type="number"
                                step="0.05"
                                min="0.01"
                                bind:value={newItem.mean_reversion_speed_annual}
                                required
                            />
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="label" for="price-profile-vol-ou"
                            >Volatilità annua (σ)</label
                        >
                        <input
                            id="price-profile-vol-ou"
                            class="input"
                            type="number"
                            step="0.01"
                            min="0"
                            bind:value={newItem.volatility_annual}
                            required
                        />
                    </div>
                {/if}

                <!-- Live preview block -->
                <div class="preview-block">
                    <div class="preview-header">
                        <strong>Anteprima Monte Carlo</strong>
                        {#if formPreviewLoading}
                            <span class="muted">Calcolo in corso…</span>
                        {/if}
                        {#if formPreviewError}
                            <span class="error">{formPreviewError}</span>
                        {/if}
                    </div>
                    {#if formPreview}
                        <ResultsChart
                            type="line"
                            data={previewToChartData(formPreview)}
                            options={previewChartOptions}
                        />
                        <p class="muted small">
                            150 traiettorie su 15 anni — la banda è il 5°-95°
                            percentile. Se collassa sulla linea blu, il
                            modello non ha incertezza.
                        </p>
                    {/if}
                </div>

                <button class="btn btn-primary" type="submit"
                    >Save Profile</button
                >
            </form>
        </div>
    {/if}

    <div class="list">
        {#each items as item}
            <button
                type="button"
                class="card item-card item-card-button"
                class:selected={selectedSavedId === item.id}
                on:click={() => selectSaved(item)}
            >
                <h3>{item.name}</h3>
                <p class="meta">
                    {item.data?.model_type ?? "escalating"} · Base €{
                        item.data?.base_price_eur_per_kwh ?? "?"
                    }
                </p>
            </button>
        {/each}
    </div>

    {#if selectedSavedId}
        <div class="card preview-saved">
            <h3>
                Anteprima — {items.find((i) => i.id === selectedSavedId)?.name}
            </h3>
            {#if savedPreviewLoading}
                <p class="muted">Calcolo in corso…</p>
            {:else if savedPreviewError}
                <p class="error">{savedPreviewError}</p>
            {:else if savedPreview}
                <ResultsChart
                    type="line"
                    data={previewToChartData(savedPreview)}
                    options={previewChartOptions}
                />
                <p class="muted small">
                    200 traiettorie su 20 anni con seed=42 (riproducibile).
                </p>
            {/if}
        </div>
    {/if}
</div>

<style>
    .toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    .item-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .item-card-button {
        display: block;
        width: 100%;
        text-align: left;
        cursor: pointer;
        border: 1px solid var(--color-border, #e2e8f0);
        background: var(--color-bg, #fff);
        transition: border-color 0.15s;
    }
    .item-card-button:hover {
        border-color: var(--color-primary, #0d6efd);
    }
    .item-card-button.selected {
        border-color: var(--color-primary, #0d6efd);
        box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.15);
    }
    .form-card {
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .group-2col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
    }
    .checkbox-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .preview-block {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-top: 1rem;
        border-top: 1px dashed var(--color-border, #e2e8f0);
    }
    .preview-header {
        display: flex;
        gap: 1rem;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }
    .preview-saved {
        padding: 1.5rem;
        margin-top: 2rem;
    }
    .muted {
        color: var(--color-text-secondary);
    }
    .small {
        font-size: 0.85rem;
    }
    .error {
        color: #dc3545;
    }
</style>
