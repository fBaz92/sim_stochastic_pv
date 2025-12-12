<script>
    import { onMount } from "svelte";
    import { api } from "../../api";
    import MonthInput from "../forms/MonthInput.svelte";
    import MonthlyProfileEditor from "../forms/MonthlyProfileEditor.svelte";

    let items = [];
    let showForm = false;

    let newItem = {
        name: "",
        profile_type: "custom",
        data: {
            // For simplicity, we can store different struct based on type
            monthly_w: Array(12).fill(100),
            monthly_24h_w: Array.from({ length: 12 }, () =>
                Array(24).fill(100),
            ),
        },
    };

    async function load() {
        items = await api.listLoadProfiles();
    }

    async function handleSubmit() {
        // Construct standard data payload
        const payloadData = {};
        if (newItem.profile_type === "custom") {
            payloadData.monthly_w = newItem.data.monthly_w;
        } else if (newItem.profile_type === "custom_24h") {
            payloadData.monthly_24h_w = newItem.data.monthly_24h_w;
        }

        await api.createLoadProfile({
            name: newItem.name,
            profile_type: newItem.profile_type,
            data: payloadData,
        });
        showForm = false;
        load();
    }

    onMount(load);
</script>

<div class="manager">
    <div class="toolbar">
        <h2>Load Profiles</h2>
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
                    <label class="label" for="load-profile-name">Name</label>
                    <input id="load-profile-name" class="input" bind:value={newItem.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="load-profile-type">Type</label>
                    <select id="load-profile-type" class="select" bind:value={newItem.profile_type}>
                        <option value="custom">Monthly Average (W)</option>
                        <option value="custom_24h">Monthly 24h (W)</option>
                    </select>
                </div>

                {#if newItem.profile_type === "custom"}
                    <MonthInput
                        label="Average Watts"
                        bind:values={newItem.data.monthly_w}
                    />
                {:else if newItem.profile_type === "custom_24h"}
                    <MonthlyProfileEditor
                        label="24h Pattern"
                        bind:values={newItem.data.monthly_24h_w}
                    />
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
                <p class="meta">{item.profile_type}</p>
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
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
    }
</style>
