<script>
    import { onMount } from 'svelte';
    import { api } from '../../api';
    import BatteryDetail from './BatteryDetail.svelte';

    let items = [];
    let loading = false;
    let showForm = false;
    let editingId = null;
    let deleteConfirmId = null;
    // Product sheet toggled per item.
    let detailItem = null;

    const emptyForm = () => ({
        name: '',
        manufacturer: '',
        model_number: '',
        capacity_kwh: 0,
        cycles_life: 6000,
        datasheet: '',
        specs: {},
    });

    let formData = emptyForm();

    async function loadItems() {
        loading = true;
        try {
            items = await api.listBatteries();
        } catch (e) {
            console.error(e);
            alert('Errore nel caricamento delle batterie');
        } finally {
            loading = false;
        }
    }

    function startEdit(item) {
        editingId = item.id;
        formData = {
            name: item.name ?? '',
            manufacturer: item.manufacturer ?? '',
            model_number: item.model_number ?? '',
            capacity_kwh: item.capacity_kwh ?? 0,
            cycles_life: item.specs?.cycles_life ?? item.cycles_life ?? 6000,
            datasheet: item.datasheet ?? '',
            specs: item.specs ?? {},
        };
        showForm = true;
        deleteConfirmId = null;
        document.getElementById('battery-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function cancelForm() {
        showForm = false;
        editingId = null;
        formData = emptyForm();
    }

    async function handleSubmit() {
        try {
            if (editingId != null) {
                await api.updateBattery(editingId, formData);
            } else {
                await api.createBattery(formData);
            }
            cancelForm();
            await loadItems();
        } catch (e) {
            console.error(e);
            alert('Errore: ' + e.message);
        }
    }

    async function handleDelete(id) {
        try {
            await api.deleteBattery(id);
            deleteConfirmId = null;
            await loadItems();
        } catch (e) {
            console.error(e);
            alert('Errore nella cancellazione: ' + e.message);
        }
    }

    onMount(loadItems);
</script>

<div>
    <div class="header-actions">
        <h2>Batterie</h2>
        <button class="btn btn-primary" on:click={() => {
            if (showForm && editingId === null) { cancelForm(); }
            else { cancelForm(); showForm = true; }
        }}>
            {showForm ? 'Annulla' : '+ Aggiungi'}
        </button>
    </div>

    {#if showForm}
        <div id="battery-form" class="card form-card">
            <h3>{editingId ? 'Modifica batteria' : 'Nuova batteria'}</h3>
            {#if editingId}
                <p class="edit-hint">Stai modificando una batteria esistente. Puoi anche rinominarla.</p>
            {/if}
            <form on:submit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <div class="form-group">
                    <label class="label" for="bat-name">Nome *</label>
                    <input id="bat-name" class="input" bind:value={formData.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="bat-capacity">Capacità (kWh) *</label>
                    <input id="bat-capacity" class="input" type="number" step="0.1" min="0"
                           bind:value={formData.capacity_kwh} required />
                </div>
                <div class="form-group">
                    <label class="label" for="bat-cycles">Cicli di vita *</label>
                    <input id="bat-cycles" class="input" type="number" step="100" min="100"
                           bind:value={formData.cycles_life} required
                           title="Numero di cicli carica/scarica completi prima di raggiungere fine vita (tipico: 3000–10000)" />
                </div>
                <div class="form-group">
                    <label class="label" for="bat-manufacturer">Produttore</label>
                    <input id="bat-manufacturer" class="input" bind:value={formData.manufacturer} />
                </div>
                <div class="form-group">
                    <label class="label" for="bat-model">Numero modello</label>
                    <input id="bat-model" class="input" bind:value={formData.model_number} />
                </div>
                <div class="form-actions">
                    <button type="button" class="btn btn-ghost" on:click={cancelForm}>Annulla</button>
                    <button type="submit" class="btn btn-primary">
                        {editingId ? 'Aggiorna' : 'Salva'}
                    </button>
                </div>
            </form>
        </div>
    {/if}

    {#if loading}
        <p>Caricamento…</p>
    {:else if items.length === 0}
        <p class="empty">Nessuna batteria nel catalogo. Aggiungine una.</p>
    {:else}
        <div class="grid">
            {#each items as item (item.id)}
                <div class="card item-card" class:editing={editingId === item.id}>
                    <div class="item-body">
                        <h4>{item.name}</h4>
                        <p class="specs">{item.capacity_kwh} kWh</p>
                        <p class="meta">
                            {item.specs?.cycles_life ?? item.cycles_life ?? '–'} cicli
                            {#if item.manufacturer} · {item.manufacturer}{/if}
                        </p>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmId === item.id}
                            <span class="confirm-label">Eliminare?</span>
                            <button class="btn btn-sm btn-danger"
                                    on:click={() => handleDelete(item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost"
                                    on:click={() => deleteConfirmId = null}>No</button>
                        {:else}
                            <button class="btn btn-sm btn-ghost" title="Scheda con curva di degrado"
                                    on:click={() => detailItem = detailItem?.id === item.id ? null : item}>📊</button>
                            <button class="btn btn-sm btn-ghost" title="Modifica"
                                    on:click={() => startEdit(item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina"
                                    on:click={() => { deleteConfirmId = item.id; editingId = null; }}>🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
        {#if detailItem}
            {#key detailItem.id}
                <BatteryDetail battery={detailItem} />
            {/key}
        {/if}
    {/if}
</div>

<style>
    .header-actions { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1rem; }
    .form-card { margin-bottom: 2rem; background-color: var(--color-bg-tertiary); }
    .edit-hint { font-size: 0.82rem; color: var(--color-text-muted, #6c757d); margin: 0 0 1rem; }
    .item-card { display: flex; flex-direction: column; gap: 0.5rem; }
    .item-card.editing { border-color: var(--color-accent, #0d6efd); outline: 2px solid var(--color-accent, #0d6efd); outline-offset: 2px; }
    .item-body { flex: 1; }
    .item-actions { display: flex; align-items: center; gap: 0.4rem; justify-content: flex-end; border-top: 1px solid var(--color-border, #e2e8f0); padding-top: 0.5rem; }
    .confirm-label { font-size: 0.82rem; color: var(--color-danger, #dc3545); font-weight: 500; }
    .specs { font-size: 1.2rem; font-weight: bold; color: var(--color-accent); }
    .meta { color: var(--color-text-secondary); font-size: 0.9rem; }
    .form-actions { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: flex-end; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
</style>
