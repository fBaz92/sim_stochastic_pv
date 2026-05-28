<script>
    /**
     * SavedCampaignsManager — Database tab for saved campaign (Design)
     * configurations.
     *
     * Mirrors :file:`SavedScenariosManager.svelte`: list rows from
     * ``saved_configurations`` where ``config_type = 'campaign'`` (a.k.a.
     * "Design" in the UI vocabulary), with rename, delete, and
     * "Modifica nel wizard" deep-link into :file:`CampaignBuilder.svelte`.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";
    import { pendingConfigurationId } from "../../lib/stores";

    let items = [];
    let loading = false;
    let renamingId = null;
    let renameDraft = "";
    let deleteConfirmId = null;
    let message = "";

    async function load() {
        loading = true;
        message = "";
        try {
            items = await api.listConfigurations("campaign");
        } catch (e) {
            console.error(e);
            message = "Errore nel caricamento dei design salvati: " + e.message;
        } finally {
            loading = false;
        }
    }

    function startRename(item) {
        renamingId = item.id;
        renameDraft = item.name;
        deleteConfirmId = null;
    }

    function cancelRename() {
        renamingId = null;
        renameDraft = "";
    }

    async function commitRename(item) {
        const name = renameDraft.trim();
        if (!name || name === item.name) {
            cancelRename();
            return;
        }
        try {
            await api.updateConfiguration(item.id, {
                name,
                config_type: item.config_type,
                data: item.data,
            });
            await load();
            cancelRename();
        } catch (e) {
            alert("Errore nella rinomina: " + e.message);
        }
    }

    async function handleDelete(id) {
        try {
            await api.deleteConfiguration(id);
            deleteConfirmId = null;
            await load();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    function openInBuilder(item) {
        pendingConfigurationId.set(item.id);
        window.location.hash = "/design";
    }

    /**
     * Lightweight one-liner showing how big the design's search space is.
     */
    function describeCampaign(item) {
        const d = item.data ?? {};
        const sel = d.hardware_selections ?? d.selections ?? {};
        const invN = (sel.inverter_ids || sel.inverters || []).length;
        const panN = (sel.panel_ids || sel.panels || []).length;
        const batN = (sel.battery_ids || sel.batteries || []).length;
        const panCounts = (d.optimization?.panel_count_options ?? []).length;
        const batCounts = (d.optimization?.battery_count_options ?? []).length;
        const parts = [];
        if (invN) parts.push(`${invN} inverter`);
        if (panN) parts.push(`${panN} pannelli`);
        if (batN) parts.push(`${batN} batterie`);
        if (panCounts) parts.push(`${panCounts} taglie pannelli`);
        if (batCounts) parts.push(`${batCounts} taglie batterie`);
        return parts.length ? parts.join(" · ") : "design salvato";
    }

    onMount(load);
</script>

<div>
    <div class="header-actions">
        <h2>Design salvati</h2>
        <p class="hint">
            Esplorazioni hardware multi-configurazione (Design). Apri un
            design nel builder per modificarne lo spazio di ricerca,
            oppure rinominalo / eliminalo direttamente da qui.
        </p>
    </div>

    {#if message}
        <p class="error">{message}</p>
    {/if}

    {#if loading}
        <p>Caricamento…</p>
    {:else if items.length === 0}
        <p class="empty">
            Nessun design salvato. Creane uno dalla
            pagina <a href="#/design">Design</a>.
        </p>
    {:else}
        <div class="list">
            {#each items as item (item.id)}
                <div class="card item-card">
                    <div class="item-body">
                        {#if renamingId === item.id}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRename(item)}
                            >
                                <input
                                    class="input"
                                    bind:value={renameDraft}
                                    autofocus
                                    required
                                />
                                <button type="submit" class="btn btn-sm btn-primary">
                                    Salva
                                </button>
                                <button
                                    type="button"
                                    class="btn btn-sm btn-ghost"
                                    on:click={cancelRename}
                                >
                                    Annulla
                                </button>
                            </form>
                        {:else}
                            <h4>{item.name}</h4>
                        {/if}
                        <p class="meta">{describeCampaign(item)}</p>
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmId === item.id}
                            <span class="confirm-label">Eliminare?</span>
                            <button
                                class="btn btn-sm btn-danger"
                                on:click={() => handleDelete(item.id)}
                            >Sì</button>
                            <button
                                class="btn btn-sm btn-ghost"
                                on:click={() => (deleteConfirmId = null)}
                            >No</button>
                        {:else if renamingId !== item.id}
                            <button
                                class="btn btn-sm btn-primary"
                                title="Apri nel builder per modificare i parametri"
                                on:click={() => openInBuilder(item)}
                            >Modifica nel builder</button>
                            <button
                                class="btn btn-sm btn-ghost"
                                title="Rinomina"
                                on:click={() => startRename(item)}
                            >✏️</button>
                            <button
                                class="btn btn-sm btn-ghost btn-del"
                                title="Elimina"
                                on:click={() => {
                                    deleteConfirmId = item.id;
                                }}
                            >🗑️</button>
                        {/if}
                    </div>
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .header-actions { margin-bottom: 1.5rem; }
    .hint {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
        margin: 0.25rem 0 0;
        max-width: 60ch;
    }
    .list { display: flex; flex-direction: column; gap: 0.75rem; }
    .item-card {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem;
        height: auto;
    }
    .item-body { flex: 1; min-width: 0; }
    .item-body h4 { margin: 0; }
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.88rem;
        margin: 0.25rem 0 0;
    }
    .item-actions { display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0; }
    .rename-form { display: flex; gap: 0.4rem; align-items: center; }
    .rename-form .input { min-width: 18rem; }
    .confirm-label {
        font-size: 0.82rem;
        color: var(--color-danger, #dc3545);
        font-weight: 500;
    }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .error { color: var(--color-danger, #dc3545); }
</style>
