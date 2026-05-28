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
        // Phase 16 — optional electrical datasheet fields. Sent
        // top-level *and* injected into specs so older API consumers
        // and the simulator both see them.
        v_oc_stc_v: null,
        v_mpp_stc_v: null,
        i_sc_stc_a: null,
        i_mpp_stc_a: null,
        n_cells_series: null,
        beta_voc_pct_per_c: null,
        gamma_pmax_pct_per_c: null,
        noct_c: null,
        specs: {},
    });

    let formData = emptyForm();
    let showElectrical = false;

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
        const specs = item.specs ?? {};
        formData = {
            name: item.name ?? '',
            manufacturer: item.manufacturer ?? '',
            model_number: item.model_number ?? '',
            power_w: item.power_w ?? 0,
            datasheet: item.datasheet ?? '',
            v_oc_stc_v: item.v_oc_stc_v ?? specs.v_oc_stc_v ?? null,
            v_mpp_stc_v: item.v_mpp_stc_v ?? specs.v_mpp_stc_v ?? null,
            i_sc_stc_a: item.i_sc_stc_a ?? specs.i_sc_stc_a ?? null,
            i_mpp_stc_a: item.i_mpp_stc_a ?? specs.i_mpp_stc_a ?? null,
            n_cells_series: item.n_cells_series ?? specs.n_cells_series ?? null,
            beta_voc_pct_per_c: item.beta_voc_pct_per_c ?? specs.beta_voc_pct_per_c ?? null,
            gamma_pmax_pct_per_c: item.gamma_pmax_pct_per_c ?? specs.gamma_pmax_pct_per_c ?? null,
            noct_c: item.noct_c ?? specs.noct_c ?? null,
            specs: specs,
        };
        showForm = true;
        // Auto-expand the electrical accordion when any electrical
        // field is already populated so the user can see the values
        // they're editing.
        showElectrical = (
            formData.v_oc_stc_v != null ||
            formData.v_mpp_stc_v != null ||
            formData.n_cells_series != null
        );
        deleteConfirmId = null;
        document.getElementById('panel-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function cancelForm() {
        showForm = false;
        editingId = null;
        formData = emptyForm();
        showElectrical = false;
    }

    async function handleSubmit() {
        try {
            // Mirror electrical datasheet fields into specs so the
            // simulator can read them via the JSON blob even if the API
            // payload is later trimmed by a proxy.
            const payload = { ...formData };
            payload.specs = {
                ...(formData.specs ?? {}),
                ...(formData.v_oc_stc_v != null ? { v_oc_stc_v: Number(formData.v_oc_stc_v) } : {}),
                ...(formData.v_mpp_stc_v != null ? { v_mpp_stc_v: Number(formData.v_mpp_stc_v) } : {}),
                ...(formData.i_sc_stc_a != null ? { i_sc_stc_a: Number(formData.i_sc_stc_a) } : {}),
                ...(formData.i_mpp_stc_a != null ? { i_mpp_stc_a: Number(formData.i_mpp_stc_a) } : {}),
                ...(formData.n_cells_series != null ? { n_cells_series: Number(formData.n_cells_series) } : {}),
                ...(formData.beta_voc_pct_per_c != null ? { beta_voc_pct_per_c: Number(formData.beta_voc_pct_per_c) } : {}),
                ...(formData.gamma_pmax_pct_per_c != null ? { gamma_pmax_pct_per_c: Number(formData.gamma_pmax_pct_per_c) } : {}),
                ...(formData.noct_c != null ? { noct_c: Number(formData.noct_c) } : {}),
            };
            if (editingId != null) {
                await api.updatePanel(editingId, payload);
            } else {
                await api.createPanel(payload);
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
                <p class="edit-hint">Stai modificando un pannello esistente. Puoi anche rinominarlo.</p>
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

                <!-- Phase 16 — optional electrical datasheet block, gated behind a toggle. -->
                <div class="form-group">
                    <label class="toggle-row">
                        <input type="checkbox" bind:checked={showElectrical} />
                        <span class="toggle-label">Dati elettrici dettagliati (Phase 16 — modello MPPT)</span>
                    </label>
                    <p class="hint">
                        Compila questi campi se vuoi usare il modello elettrico dettagliato
                        in uno scenario (controllo finestra MPPT, derating termico, shutdown V_dc).
                        Tutti i valori dal datasheet del produttore.
                    </p>
                </div>
                {#if showElectrical}
                    <div class="electrical-grid">
                        <div class="form-group">
                            <label class="label" for="pan-voc">V_oc STC (V)</label>
                            <input id="pan-voc" class="input" type="number" step="0.1"
                                   bind:value={formData.v_oc_stc_v} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-vmpp">V_mpp STC (V)</label>
                            <input id="pan-vmpp" class="input" type="number" step="0.1"
                                   bind:value={formData.v_mpp_stc_v} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-isc">I_sc STC (A)</label>
                            <input id="pan-isc" class="input" type="number" step="0.01"
                                   bind:value={formData.i_sc_stc_a} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-impp">I_mpp STC (A)</label>
                            <input id="pan-impp" class="input" type="number" step="0.01"
                                   bind:value={formData.i_mpp_stc_a} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-ncells">Celle in serie</label>
                            <input id="pan-ncells" class="input" type="number" step="1"
                                   bind:value={formData.n_cells_series} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-noct">NOCT (°C)</label>
                            <input id="pan-noct" class="input" type="number" step="0.1"
                                   bind:value={formData.noct_c} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-beta">β V_oc (%/°C)</label>
                            <input id="pan-beta" class="input" type="number" step="0.001"
                                   bind:value={formData.beta_voc_pct_per_c} />
                        </div>
                        <div class="form-group">
                            <label class="label" for="pan-gamma">γ P_max (%/°C)</label>
                            <input id="pan-gamma" class="input" type="number" step="0.001"
                                   bind:value={formData.gamma_pmax_pct_per_c} />
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
    .specs { font-size: 1.2rem; font-weight: bold; color: var(--color-accent); }
    .meta { color: var(--color-text-secondary); font-size: 0.9rem; }
    .form-actions { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: flex-end; }
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
