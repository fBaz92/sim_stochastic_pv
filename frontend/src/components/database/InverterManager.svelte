<script>
    import { onMount } from 'svelte';
    import { api } from '../../api';

    let inverters = [];
    let loading = false;
    let showForm = false;
    
    // New Inverter Form Data
    let newItem = {
        name: '',
        manufacturer: '',
        model_number: '',
        p_ac_max_kw: 0,
        datasheet: '',
        specs: {}
    };

    async function loadItems() {
        loading = true;
        try {
            inverters = await api.listInverters();
        } catch (e) {
            console.error(e);
            alert('Error loading inverters');
        } finally {
            loading = false;
        }
    }

    async function handleSubmit() {
        try {
            await api.createInverter(newItem);
            showForm = false;
            newItem = { name: '', manufacturer: '', model_number: '', p_ac_max_kw: 0, datasheet: '', specs: {} };
            await loadItems();
        } catch (e) {
            console.error(e);
            alert('Error creating inverter: ' + e.message);
        }
    }

    onMount(loadItems);
</script>

<div>
    <div class="header-actions">
        <h2>Inverters</h2>
        <button class="btn btn-primary" on:click={() => showForm = !showForm}>
            {showForm ? 'Cancel' : 'Add Inverter'}
        </button>
    </div>

    {#if showForm}
        <div class="card form-card">
            <h3>New Inverter</h3>
            <form on:submit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <div class="form-group">
                    <label class="label" for="new-inverter-name">Name</label>
                    <input id="new-inverter-name" class="input" bind:value={newItem.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="new-inverter-power">Specify Max Power (kW)</label>
                    <input id="new-inverter-power" class="input" type="number" step="0.1" bind:value={newItem.p_ac_max_kw} required />
                </div>
                <div class="form-group">
                    <label class="label" for="new-inverter-manufacturer">Manufacturer</label>
                    <input id="new-inverter-manufacturer" class="input" bind:value={newItem.manufacturer} />
                </div>
                <div class="form-group">
                    <label class="label" for="new-inverter-model">Model Number</label>
                    <input id="new-inverter-model" class="input" bind:value={newItem.model_number} />
                </div>
                <div class="form-group">
                    <label class="label" for="new-inverter-datasheet">Datasheet URL</label>
                    <input id="new-inverter-datasheet" class="input" bind:value={newItem.datasheet} />
                </div>
                <div class="form-actions">
                    <button type="submit" class="btn btn-primary">Save</button>
                </div>
            </form>
        </div>
    {/if}

    {#if loading}
        <p>Loading...</p>
    {:else}
        <div class="grid">
            {#each inverters as inverter}
                <div class="card item-card">
                    <h4>{inverter.name}</h4>
                    <p class="specs">{inverter.p_ac_max_kw || inverter.nominal_power_kw} kW</p>
                    {#if inverter.manufacturer}
                        <p class="meta">{inverter.manufacturer} {inverter.model_number || ''}</p>
                    {/if}
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .header-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
    }
    .form-card {
        margin-bottom: 2rem;
        background-color: var(--color-bg-tertiary); /* Slightly different to stand out */
    }
    .specs {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--color-accent);
    }
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
    }
    .form-actions {
        margin-top: 1rem;
        display: flex;
        justify-content: flex-end;
    }
</style>
