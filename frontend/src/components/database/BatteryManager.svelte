<script>
    import { onMount } from 'svelte';
    import { api } from '../../api';

    let items = [];
    let loading = false;
    let showForm = false;
    
    let newItem = {
        name: '',
        manufacturer: '',
        model_number: '',
        capacity_kwh: 0,
        datasheet: '',
        specs: {}
    };

    async function loadItems() {
        loading = true;
        try {
            items = await api.listBatteries();
        } catch (e) {
            console.error(e);
            alert('Error loading batteries');
        } finally {
            loading = false;
        }
    }

    async function handleSubmit() {
        try {
            await api.createBattery(newItem);
            showForm = false;
            newItem = { name: '', manufacturer: '', model_number: '', capacity_kwh: 0, datasheet: '', specs: {} };
            await loadItems();
        } catch (e) {
            console.error(e);
            alert('Error creating battery: ' + e.message);
        }
    }

    onMount(loadItems);
</script>

<div>
    <div class="header-actions">
        <h2>Batteries</h2>
        <button class="btn btn-primary" on:click={() => showForm = !showForm}>
            {showForm ? 'Cancel' : 'Add Battery'}
        </button>
    </div>

    {#if showForm}
        <div class="card form-card">
            <h3>New Battery</h3>
            <form on:submit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <div class="form-group">
                    <label class="label" for="new-battery-name">Name</label>
                    <input id="new-battery-name" class="input" bind:value={newItem.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="new-battery-capacity">Capacity (kWh)</label>
                    <input id="new-battery-capacity" class="input" type="number" step="0.1" bind:value={newItem.capacity_kwh} required />
                </div>
                <div class="form-group">
                    <label class="label" for="new-battery-manufacturer">Manufacturer</label>
                    <input id="new-battery-manufacturer" class="input" bind:value={newItem.manufacturer} />
                </div>
                <div class="form-group">
                    <label class="label" for="new-battery-model">Model Number</label>
                    <input id="new-battery-model" class="input" bind:value={newItem.model_number} />
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
            {#each items as item}
                <div class="card item-card">
                    <h4>{item.name}</h4>
                    <p class="specs">{item.capacity_kwh} kWh</p>
                    {#if item.manufacturer}
                        <p class="meta">{item.manufacturer} {item.model_number || ''}</p>
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
        background-color: var(--color-bg-tertiary);
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
