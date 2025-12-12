<script>
    import { onMount } from "svelte";
    import { api } from "../../api";

    let items = [];
    let showForm = false;

    let newItem = {
        name: "",
        data: {
            base_price_eur_per_kwh: 0.25,
            annual_escalation: 0.02,
            use_stochastic: false,
            p05: -0.05,
            p95: 0.05,
        },
    };

    async function load() {
        items = await api.listPriceProfiles();
    }

    async function handleSubmit() {
        await api.createPriceProfile({
            name: newItem.name,
            data: newItem.data,
        });
        showForm = false;
        load();
    }

    onMount(load);
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
                    <input id="price-profile-name" class="input" bind:value={newItem.name} required />
                </div>
                <div class="group-2col">
                    <div class="form-group">
                        <label class="label" for="price-profile-base">Base Price (€/kWh)</label>
                        <input
                            id="price-profile-base"
                            class="input"
                            type="number"
                            step="0.01"
                            bind:value={newItem.data.base_price_eur_per_kwh}
                            required
                        />
                    </div>
                    <div class="form-group">
                        <label class="label" for="price-profile-escalation">Annual Escalation</label>
                        <input
                            id="price-profile-escalation"
                            class="input"
                            type="number"
                            step="0.01"
                            bind:value={newItem.data.annual_escalation}
                            required
                        />
                    </div>
                </div>

                <div class="form-group">
                    <label class="checkbox-label">
                        <input
                            type="checkbox"
                            bind:checked={newItem.data.use_stochastic}
                        />
                        Stochastic?
                    </label>
                </div>
                {#if newItem.data.use_stochastic}
                    <div class="group-2col">
                        <div class="form-group">
                            <label class="label" for="price-profile-p05">P05</label>
                            <input
                                id="price-profile-p05"
                                class="input"
                                type="number"
                                step="0.01"
                                bind:value={newItem.data.p05}
                            />
                        </div>
                        <div class="form-group">
                            <label class="label" for="price-profile-p95">P95</label>
                            <input
                                id="price-profile-p95"
                                class="input"
                                type="number"
                                step="0.01"
                                bind:value={newItem.data.p95}
                            />
                        </div>
                    </div>
                {/if}

                <button class="btn btn-primary" type="submit"
                    >Save Profile</button
                >
            </form>
        </div>
    {/if}

    <div class="list">
        {#each items as item}
            <div class="card item-card">
                <h3>{item.name}</h3>
                <p class="meta">
                    Base: €{item.data.base_price_eur_per_kwh} | Esc: {(
                        item.data.annual_escalation * 100
                    ).toFixed(1)}%
                </p>
            </div>
        {/each}
    </div>
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
</style>
