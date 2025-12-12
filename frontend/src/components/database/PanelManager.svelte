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
        power_w: 0,
        datasheet: '',
        specs: {}
    };

    async function loadItems() {
        loading = true;
        try {
            items = await api.listPanels();
        } catch (e) {
            console.error(e);
            alert('Error loading panels');
        } finally {
            loading = false;
        }
    }

    async function handleSubmit() {
        try {
            await api.createPanel(newItem);
            showForm = false;
            newItem = { name: '', manufacturer: '', model_number: '', power_w: 0, datasheet: '', specs: {} };
            await loadItems();
        } catch (e) {
            console.error(e);
            alert('Error creating panel: ' + e.message);
        }
    }

    onMount(loadItems);
</script>

<div>
    <div class="header-actions">
        <h2>PV Panels</h2>
        <button class="btn btn-primary" on:click={() => showForm = !showForm}>
            {showForm ? 'Cancel' : 'Add Panel'}
        </button>
    </div>

    {#if showForm}
        <div class="card form-card">
            <h3>New Panel</h3>
            <form on:submit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <div class="form-group">
                    <label class="label" for="new-panel-name">Name</label>
                    <input id="new-panel-name" class="input" bind:value={newItem.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="new-panel-power">Power (W)</label>
                    <input id="new-panel-power" class="input" type="number" step="1" bind:value={newItem.power_w} required />
                </div>
                <div class="form-group">
                    <label class="label" for="new-panel-manufacturer">Manufacturer</label>
                    <input id="new-panel-manufacturer" class="input" bind:value={newItem.manufacturer} />
                </div>
                <div class="form-group">
                    <label class="label" for="new-panel-model">Model Number</label>
                    <input id="new-panel-model" class="input" bind:value={newItem.model_number} />
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
                    <p class="specs">{item.power_w} W</p>
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
