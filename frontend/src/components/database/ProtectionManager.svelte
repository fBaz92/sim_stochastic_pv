<script>
    /**
     * ProtectionManager — Database tab for the DC protection catalogue
     * (gPV fuses, disconnectors, SPDs). Plain CRUD over /api/protections;
     * the designer reads this list when costing the protection bill of
     * materials.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";

    let items = [];
    let loading = false;
    let showForm = false;
    let editingId = null;
    let deleteConfirmId = null;

    const KINDS = [
        { value: "fuse", label: "Fusibile" },
        { value: "breaker", label: "Magnetotermico" },
        { value: "disconnector", label: "Sezionatore" },
        { value: "spd", label: "Scaricatore (SPD)" },
    ];

    const emptyForm = () => ({
        name: "",
        manufacturer: "",
        kind: "fuse",
        rated_current_a: null,
        rated_voltage_v: 1000,
        price_eur: null,
        notes: "",
    });
    let formData = emptyForm();

    function kindLabel(value) {
        return KINDS.find((k) => k.value === value)?.label ?? value;
    }

    async function loadItems() {
        loading = true;
        try {
            items = await api.listProtections();
        } catch (e) {
            alert("Errore nel caricamento delle protezioni: " + e.message);
        } finally {
            loading = false;
        }
    }

    function startEdit(item) {
        editingId = item.id;
        formData = { ...emptyForm(), ...item };
        showForm = true;
    }

    async function handleSubmit() {
        try {
            const payload = {
                ...formData,
                rated_current_a: formData.rated_current_a !== null && formData.rated_current_a !== ""
                    ? Number(formData.rated_current_a) : null,
                rated_voltage_v: formData.rated_voltage_v !== null && formData.rated_voltage_v !== ""
                    ? Number(formData.rated_voltage_v) : null,
                price_eur: formData.price_eur !== null && formData.price_eur !== ""
                    ? Number(formData.price_eur) : null,
            };
            if (editingId) await api.updateProtection(editingId, payload);
            else await api.createProtection(payload);
            showForm = false;
            editingId = null;
            formData = emptyForm();
            await loadItems();
        } catch (e) {
            alert("Errore nel salvataggio: " + e.message);
        }
    }

    async function handleDelete(id) {
        try {
            await api.deleteProtection(id);
            deleteConfirmId = null;
            await loadItems();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    onMount(loadItems);
</script>

<div>
    <div class="header-actions">
        <h2>Protezioni DC</h2>
        <button class="btn btn-primary" on:click={() => {
            showForm = !showForm;
            if (!showForm) { editingId = null; formData = emptyForm(); }
        }}>
            {showForm ? "Annulla" : "+ Aggiungi protezione"}
        </button>
    </div>
    <p class="hint">
        Fusibili gPV, sezionatori e scaricatori. Il designer suggerisce la
        taglia del fusibile di stringa secondo CEI EN 62548 e la mappa su
        questo catalogo.
    </p>

    {#if showForm}
        <div class="card form-card">
            <form on:submit|preventDefault={handleSubmit}>
                <div class="form-grid">
                    <label>Nome *<input class="input" bind:value={formData.name} required /></label>
                    <label>Marca<input class="input" bind:value={formData.manufacturer} /></label>
                    <label>Tipo
                        <select class="input" bind:value={formData.kind}>
                            {#each KINDS as k}<option value={k.value}>{k.label}</option>{/each}
                        </select>
                    </label>
                    <label>Corrente In (A)<input class="input" type="number" step="1" min="1" bind:value={formData.rated_current_a} /></label>
                    <label>Tensione (V)<input class="input" type="number" step="50" min="50" bind:value={formData.rated_voltage_v} /></label>
                    <label>Prezzo (€)<input class="input" type="number" step="0.5" min="0" bind:value={formData.price_eur} /></label>
                </div>
                <div class="form-actions">
                    <button type="submit" class="btn btn-primary">
                        {editingId ? "Salva modifiche" : "Aggiungi"}
                    </button>
                </div>
            </form>
        </div>
    {/if}

    {#if loading}
        <p>Caricamento…</p>
    {:else if items.length === 0}
        <p class="empty">Nessuna protezione nel catalogo.</p>
    {:else}
        <div class="list">
            {#each items as item (item.id)}
                <div class="card item-card">
                    <div class="item-body">
                        <h4>{item.name}</h4>
                        <p class="meta">
                            {kindLabel(item.kind)}
                            {#if item.rated_current_a != null} · {item.rated_current_a} A{/if}
                            {#if item.rated_voltage_v != null} · {item.rated_voltage_v} V{/if}
                            {#if item.price_eur != null} · {item.price_eur} €{/if}
                        </p>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmId === item.id}
                            <span class="confirm-label">Eliminare?</span>
                            <button class="btn btn-sm btn-danger" on:click={() => handleDelete(item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost" on:click={() => (deleteConfirmId = null)}>No</button>
                        {:else}
                            <button class="btn btn-sm btn-ghost" title="Modifica" on:click={() => startEdit(item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina" on:click={() => (deleteConfirmId = item.id)}>🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .header-actions { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .hint { color: var(--color-text-secondary); font-size: 0.9rem; margin: 0 0 1.5rem; max-width: 65ch; }
    .form-card { padding: 1.25rem; margin-bottom: 1.5rem; background-color: var(--color-bg-tertiary); }
    .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem 1rem; }
    .form-grid label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.82rem; color: var(--color-text-secondary); }
    .form-actions { margin-top: 1rem; display: flex; justify-content: flex-end; }
    .list { display: flex; flex-direction: column; gap: 0.75rem; }
    .item-card {
        display: flex; flex-direction: row; align-items: center;
        justify-content: space-between; gap: 1rem; padding: 1rem; height: auto;
    }
    .item-body h4 { margin: 0; }
    .meta { color: var(--color-text-secondary); font-size: 0.85rem; margin: 0.25rem 0 0; }
    .item-actions { display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0; }
    .confirm-label { font-size: 0.82rem; color: var(--color-danger, #dc3545); font-weight: 500; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
</style>
