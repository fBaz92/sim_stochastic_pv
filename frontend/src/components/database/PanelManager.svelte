<script>
    import { onMount } from 'svelte';
    import { api } from '../../api';

    let items = [];
    let loading = false;
    let showForm = false;
    let editingId = null;
    let deleteConfirmId = null;

    const emptyForm = () => ({
        name: '',
        manufacturer: '',
        model_number: '',
        power_w: 0,
        datasheet: '',
        specs: {},
    });

    let formData = emptyForm();

    async function loadItems() {
        loading = true;
        try {
            items = await api.listPanels();
        } catch (e) {
            console.error(e);
            alert('Errore nel caricamento dei pannelli');
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
            power_w: item.power_w ?? 0,
            datasheet: item.datasheet ?? '',
            specs: item.specs ?? {},
        };
        showForm = true;
        deleteConfirmId = null;
        document.getElementById('panel-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function cancelForm() {
        showForm = false;
        editingId = null;
        formData = emptyForm();
    }

    async function handleSubmit() {
        try {
            await api.createPanel(formData);
            cancelForm();
            await loadItems();
        } catch (e) {
            console.error(e);
            alert('Errore: ' + e.message);
        }
    }

    async function handleDelete(id) {
        try {
            await api.deletePanel(id);
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
        <h2>Pannelli PV</h2>
        <button class="btn btn-primary" on:click={() => {
            if (showForm && editingId === null) { cancelForm(); }
            else { cancelForm(); showForm = true; }
        }}>
            {showForm ? 'Annulla' : '+ Aggiungi'}
        </button>
    </div>

    {#if showForm}
        <div id="panel-form" class="card form-card">
            <h3>{editingId ? 'Modifica pannello' : 'Nuovo pannello'}</h3>
            {#if editingId}
                <p class="edit-hint">Il nome identifica univocamente il pannello.</p>
            {/if}
            <form on:submit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <div class="form-group">
                    <label class="label" for="pan-name">Nome *</label>
                    <input id="pan-name" class="input" bind:value={formData.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="pan-power">Potenza (W) *</label>
                    <input id="pan-power" class="input" type="number" step="1" min="0"
                           bind:value={formData.power_w} required />
                </div>
                <div class="form-group">
                    <label class="label" for="pan-manufacturer">Produttore</label>
                    <input id="pan-manufacturer" class="input" bind:value={formData.manufacturer} />
                </div>
                <div class="form-group">
                    <label class="label" for="pan-model">Numero modello</label>
                    <input id="pan-model" class="input" bind:value={formData.model_number} />
                </div>
                <div class="form-group">
                    <label class="label" for="pan-datasheet">URL scheda tecnica</label>
                    <input id="pan-datasheet" class="input" bind:value={formData.datasheet} />
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
        <p class="empty">Nessun pannello nel catalogo. Aggiungine uno.</p>
    {:else}
        <div class="grid">
            {#each items as item (item.id)}
                <div class="card item-card" class:editing={editingId === item.id}>
                    <div class="item-body">
                        <h4>{item.name}</h4>
                        <p class="specs">{item.power_w} W</p>
                        {#if item.manufacturer}
                            <p class="meta">{item.manufacturer}{item.model_number ? ' · ' + item.model_number : ''}</p>
                        {/if}
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmId === item.id}
                            <span class="confirm-label">Eliminare?</span>
                            <button class="btn btn-sm btn-danger"
                                    on:click={() => handleDelete(item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost"
                                    on:click={() => deleteConfirmId = null}>No</button>
                        {:else}
                            <button class="btn btn-sm btn-ghost" title="Modifica"
                                    on:click={() => startEdit(item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina"
                                    on:click={() => { deleteConfirmId = item.id; editingId = null; }}>🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
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
    .btn-sm { padding: 0.2rem 0.5rem; font-size: 0.82rem; }
    .btn-del:hover { color: var(--color-danger, #dc3545); }
    .specs { font-size: 1.2rem; font-weight: bold; color: var(--color-accent); }
    .meta { color: var(--color-text-secondary); font-size: 0.9rem; }
    .form-actions { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: flex-end; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
</style>
