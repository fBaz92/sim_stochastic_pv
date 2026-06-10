<script>
    import { onMount } from 'svelte';
    import { api } from '../../api';
    import InverterDetail from './InverterDetail.svelte';

    let inverters = [];
    let loading = false;
    let showForm = false;

    /** ID dell'inverter in editing; null = modalità creazione. */
    let editingId = null;

    /** ID dell'inverter con conferma eliminazione in attesa; null = nessuna. */
    let deleteConfirmId = null;
    // Product sheet toggled per item.
    let detailItem = null;

    /** Stato vuoto del form — usato anche per il reset. */
    const emptyForm = () => ({
        name: '',
        manufacturer: '',
        model_number: '',
        p_ac_max_kw: 0,
        datasheet: '',
        // Phase 16 — optional inverter datasheet fields. Gated behind
        // an accordion to keep the casual UI tidy.
        v_dc_min_v: null,
        v_dc_max_v: null,
        v_mppt_min_v: null,
        v_mppt_max_v: null,
        n_mppt_trackers: null,
        i_dc_max_per_mppt_a: null,
        specs: {},
    });

    let formData = emptyForm();
    let showElectrical = false;

    async function loadItems() {
        loading = true;
        try {
            inverters = await api.listInverters();
        } catch (e) {
            console.error(e);
            alert('Errore nel caricamento degli inverter');
        } finally {
            loading = false;
        }
    }

    /** Pre-popola il form con i dati di un inverter esistente per la modifica. */
    function startEdit(item) {
        editingId = item.id;
        const specs = item.specs ?? {};
        formData = {
            name: item.name ?? '',
            manufacturer: item.manufacturer ?? '',
            model_number: item.model_number ?? '',
            p_ac_max_kw: item.p_ac_max_kw ?? item.nominal_power_kw ?? 0,
            datasheet: item.datasheet ?? '',
            v_dc_min_v: item.v_dc_min_v ?? specs.v_dc_min_v ?? null,
            v_dc_max_v: item.v_dc_max_v ?? specs.v_dc_max_v ?? null,
            v_mppt_min_v: item.v_mppt_min_v ?? specs.v_mppt_min_v ?? null,
            v_mppt_max_v: item.v_mppt_max_v ?? specs.v_mppt_max_v ?? null,
            n_mppt_trackers: item.n_mppt_trackers ?? specs.n_mppt_trackers ?? null,
            i_dc_max_per_mppt_a: item.i_dc_max_per_mppt_a ?? specs.i_dc_max_per_mppt_a ?? null,
            specs: specs,
        };
        showForm = true;
        showElectrical = (
            formData.v_dc_max_v != null ||
            formData.v_mppt_min_v != null ||
            formData.v_mppt_max_v != null
        );
        deleteConfirmId = null;
        // Scroll verso il form
        document.getElementById('inverter-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function cancelForm() {
        showForm = false;
        editingId = null;
        formData = emptyForm();
        showElectrical = false;
    }

    async function handleSubmit() {
        try {
            // Phase 16 — mirror electrical datasheet fields into specs
            // so the simulator picks them up even when the API surface
            // only persists ``specs``.
            const payload = { ...formData };
            payload.specs = {
                ...(formData.specs ?? {}),
                ...(formData.v_dc_min_v != null ? { v_dc_min_v: Number(formData.v_dc_min_v) } : {}),
                ...(formData.v_dc_max_v != null ? { v_dc_max_v: Number(formData.v_dc_max_v) } : {}),
                ...(formData.v_mppt_min_v != null ? { v_mppt_min_v: Number(formData.v_mppt_min_v) } : {}),
                ...(formData.v_mppt_max_v != null ? { v_mppt_max_v: Number(formData.v_mppt_max_v) } : {}),
                ...(formData.n_mppt_trackers != null ? { n_mppt_trackers: Number(formData.n_mppt_trackers) } : {}),
                ...(formData.i_dc_max_per_mppt_a != null ? { i_dc_max_per_mppt_a: Number(formData.i_dc_max_per_mppt_a) } : {}),
            };
            if (editingId != null) {
                await api.updateInverter(editingId, payload);
            } else {
                await api.createInverter(payload);
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
            await api.deleteInverter(id);
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
        <h2>Inverter</h2>
        <button class="btn btn-primary" on:click={() => {
            if (showForm && editingId === null) { cancelForm(); }
            else { cancelForm(); showForm = true; }
        }}>
            {showForm ? 'Annulla' : '+ Aggiungi'}
        </button>
    </div>

    {#if showForm}
        <div id="inverter-form" class="card form-card">
            <h3>{editingId ? 'Modifica inverter' : 'Nuovo inverter'}</h3>
            {#if editingId}
                <p class="edit-hint">
                    Stai modificando un inverter esistente. Puoi anche rinominarlo.
                </p>
            {/if}
            <form on:submit={(e) => { e.preventDefault(); handleSubmit(); }}>
                <div class="form-group">
                    <label class="label" for="inv-name">Nome *</label>
                    <input id="inv-name" class="input" bind:value={formData.name} required />
                </div>
                <div class="form-group">
                    <label class="label" for="inv-power">Potenza AC max (kW) *</label>
                    <input id="inv-power" class="input" type="number" step="0.1" min="0"
                           bind:value={formData.p_ac_max_kw} required />
                </div>
                <div class="form-group">
                    <label class="label" for="inv-manufacturer">Produttore</label>
                    <input id="inv-manufacturer" class="input" bind:value={formData.manufacturer} />
                </div>
                <div class="form-group">
                    <label class="label" for="inv-model">Numero modello</label>
                    <input id="inv-model" class="input" bind:value={formData.model_number} />
                </div>
                <div class="form-group">
                    <label class="label" for="inv-datasheet">URL scheda tecnica</label>
                    <input id="inv-datasheet" class="input" bind:value={formData.datasheet} />
                </div>

                <!-- Phase 16 — optional electrical datasheet block. -->
                <div class="form-group">
                    <label class="toggle-row">
                        <input type="checkbox" bind:checked={showElectrical} />
                        <span class="toggle-label">Dati elettrici dettagliati (Phase 16 — finestra MPPT)</span>
                    </label>
                    <p class="hint">
                        Compila questi campi per usare lo scenario con
                        <strong>electrical.mode='mppt_window'</strong>: il simulatore controlla
                        finestra MPPT, derating termico e shutdown V_dc ora per ora.
                    </p>
                </div>
                {#if showElectrical}
                    <div class="electrical-grid">
                        <div class="form-group">
                            <label class="label" for="inv-vdcmin">V_dc min (V)</label>
                            <input id="inv-vdcmin" class="input" type="number" step="1"
                                   bind:value={formData.v_dc_min_v} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="inv-vdcmax">V_dc max (V)</label>
                            <input id="inv-vdcmax" class="input" type="number" step="1"
                                   bind:value={formData.v_dc_max_v} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="inv-vmpptmin">V_mppt min (V)</label>
                            <input id="inv-vmpptmin" class="input" type="number" step="1"
                                   bind:value={formData.v_mppt_min_v} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="inv-vmpptmax">V_mppt max (V)</label>
                            <input id="inv-vmpptmax" class="input" type="number" step="1"
                                   bind:value={formData.v_mppt_max_v} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="inv-nmppt">N. MPPT tracker</label>
                            <input id="inv-nmppt" class="input" type="number" step="1" min="1"
                                   bind:value={formData.n_mppt_trackers} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="inv-idcmax">I_dc max per MPPT (A)</label>
                            <input id="inv-idcmax" class="input" type="number" step="0.1"
                                   bind:value={formData.i_dc_max_per_mppt_a} />
                        </div>
                    </div>
                {/if}

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
    {:else if inverters.length === 0}
        <p class="empty">Nessun inverter nel catalogo. Aggiungine uno.</p>
    {:else}
        <div class="grid">
            {#each inverters as item (item.id)}
                <div class="card item-card" class:editing={editingId === item.id}>
                    <div class="item-body">
                        <h4>{item.name}</h4>
                        <p class="specs">{item.p_ac_max_kw ?? item.nominal_power_kw} kW</p>
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
                            <button class="btn btn-sm btn-ghost" title="Scheda con finestre di tensione"
                                    on:click={() => detailItem = detailItem?.id === item.id ? null : item}>📊</button>
                            <button class="btn btn-sm btn-ghost"
                                    title="Modifica"
                                    on:click={() => startEdit(item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del"
                                    title="Elimina"
                                    on:click={() => { deleteConfirmId = item.id; editingId = null; }}>🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
        {#if detailItem}
            {#key detailItem.id}
                <InverterDetail inverter={detailItem} />
            {/key}
        {/if}
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
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 1rem;
    }
    .form-card {
        margin-bottom: 2rem;
        background-color: var(--color-bg-tertiary);
    }
    .edit-hint {
        font-size: 0.82rem;
        color: var(--color-text-muted, #6c757d);
        margin: 0 0 1rem;
    }
    .item-card {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .item-card.editing {
        border-color: var(--color-accent, #0d6efd);
        outline: 2px solid var(--color-accent, #0d6efd);
        outline-offset: 2px;
    }
    .item-body { flex: 1; }
    .item-actions {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        justify-content: flex-end;
        border-top: 1px solid var(--color-border, #e2e8f0);
        padding-top: 0.5rem;
    }
    .confirm-label {
        font-size: 0.82rem;
        color: var(--color-danger, #dc3545);
        font-weight: 500;
    }
    .specs {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--color-accent);
    }
    .meta { color: var(--color-text-secondary); font-size: 0.9rem; }
    .form-actions {
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        justify-content: flex-end;
    }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .toggle-row { display: flex; align-items: center; gap: 0.5rem; cursor: pointer; }
    .toggle-label { font-weight: 500; }
    .hint { font-size: 0.82rem; color: var(--color-text-muted, #6c757d); margin: 0.3rem 0 0 1.5rem; }
    .electrical-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 0.5rem 1rem;
        padding: 0.75rem;
        background-color: var(--color-bg-secondary, #f8f9fb);
        border-radius: 6px;
        border: 1px solid var(--color-border, #e2e8f0);
        margin-bottom: 1rem;
    }
    .electrical-grid .form-group { margin-bottom: 0; }
</style>
