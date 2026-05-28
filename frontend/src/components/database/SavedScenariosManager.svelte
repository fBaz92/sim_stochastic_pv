<script>
    /**
     * SavedScenariosManager — Database tab for saved scenario configurations.
     *
     * Each row represents a record in ``saved_configurations`` with
     * ``config_type = 'scenario'``. The manager supports:
     *
     * - Inline rename of any scenario (PUT ``/configurations/{id}``).
     * - Delete with confirmation.
     * - "Modifica nel wizard" — sets ``pendingConfigurationId`` and routes
     *   to ``#/scenario``, where the wizard auto-loads the configuration.
     *
     * The full scenario payload (energy / load / price / economic / etc.)
     * is intentionally edited via the wizard, not here: this manager only
     * surfaces the *list-level* operations (rename, delete, deep-link
     * into the builder) because the wizard already validates every field
     * and renders helpful previews.
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
            items = await api.listConfigurations("scenario");
        } catch (e) {
            console.error(e);
            message = "Errore nel caricamento degli scenari salvati: " + e.message;
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

    function openInWizard(item) {
        pendingConfigurationId.set(item.id);
        // The router listens to hashchange; setting location.hash is the
        // canonical "navigate" gesture in this app.
        window.location.hash = "/scenario";
    }

    /**
     * Lightweight one-liner describing an entry so the user does not have
     * to open it to remember what it contains. Pulls the most discriminating
     * fields off the saved payload.
     */
    function describeScenario(item) {
        const d = item.data?.scenario ?? item.data ?? {};
        const parts = [];
        const kwp = d.solar?.pv_kwp ?? d.energy?.pv_kwp;
        if (kwp != null) parts.push(`${kwp} kWp`);
        const nBat = d.energy?.n_batteries;
        if (nBat != null) parts.push(`${nBat}× batteria`);
        const inv = d.energy?.inverter_p_ac_max_kw;
        if (inv != null) parts.push(`inverter ${inv} kW`);
        const years = d.economic?.years ?? d.energy?.n_years;
        if (years != null) parts.push(`${years} anni`);
        return parts.length ? parts.join(" · ") : "scenario salvato";
    }

    onMount(load);
</script>

<div>
    <div class="header-actions">
        <h2>Scenari salvati</h2>
        <p class="hint">
            Configurazioni complete pronte da rieseguire. Apri uno
            scenario nel wizard per modificarne i parametri, oppure
            rinominalo / eliminalo direttamente da qui.
        </p>
    </div>

    {#if message}
        <p class="error">{message}</p>
    {/if}

    {#if loading}
        <p>Caricamento…</p>
    {:else if items.length === 0}
        <p class="empty">
            Nessuno scenario salvato. Costruiscine uno dalla
            pagina <a href="#/scenario">Scenario</a>.
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
                        <p class="meta">{describeScenario(item)}</p>
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
                                title="Apri nel wizard per modificare i parametri"
                                on:click={() => openInWizard(item)}
                            >Modifica nel wizard</button>
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
    .header-actions {
        margin-bottom: 1.5rem;
    }
    .hint {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
        margin: 0.25rem 0 0;
        max-width: 60ch;
    }
    .list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
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
    .item-actions {
        display: flex;
        gap: 0.4rem;
        align-items: center;
        flex-shrink: 0;
    }
    .rename-form {
        display: flex;
        gap: 0.4rem;
        align-items: center;
    }
    .rename-form .input { min-width: 18rem; }
    .confirm-label {
        font-size: 0.82rem;
        color: var(--color-danger, #dc3545);
        font-weight: 500;
    }
    .empty {
        color: var(--color-text-muted, #6c757d);
        font-style: italic;
    }
    .error { color: var(--color-danger, #dc3545); }
</style>
