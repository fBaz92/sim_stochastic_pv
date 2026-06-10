<script>
    /**
     * PlantDesignManager — Database tab for plant designs ("Impianti").
     *
     * Lists the saved designs (received offers at the `essential` level,
     * detailed designs once the electrical designer lands), with rename
     * and delete plus a shortcut that opens the offer page pre-loaded
     * with the selected design. Designs are created from the
     * "Analizza un'offerta" page, not from here.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";

    let designs = [];
    let locations = [];
    let loading = false;
    let message = "";

    let renamingId = null;
    let renameDraft = "";
    let deleteConfirmId = null;

    async function loadAll() {
        loading = true;
        message = "";
        try {
            const [des, locs] = await Promise.all([
                api.listDesigns(),
                api.listLocations(),
            ]);
            designs = des;
            locations = locs;
        } catch (e) {
            message = "Errore nel caricamento degli impianti: " + e.message;
        } finally {
            loading = false;
        }
    }

    function locationName(id) {
        return locations.find((l) => l.id === id)?.name ?? null;
    }

    function levelLabel(level) {
        return level === "essential" ? "offerta" : "dettagliato";
    }

    function startRename(item) {
        renamingId = item.id;
        renameDraft = item.name;
        deleteConfirmId = null;
    }

    async function commitRename(item) {
        const name = renameDraft.trim();
        if (!name || name === item.name) {
            renamingId = null;
            return;
        }
        try {
            await api.updateDesign(item.id, { name });
            renamingId = null;
            await loadAll();
        } catch (e) {
            alert("Errore nella rinomina: " + e.message);
        }
    }

    async function deleteDesign(id) {
        try {
            await api.deleteDesign(id);
            deleteConfirmId = null;
            await loadAll();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    onMount(loadAll);
</script>

<div>
    <div class="header-actions">
        <h2>Impianti</h2>
        <a class="btn btn-primary" href="#/offerta">+ Analizza un'offerta</a>
    </div>
    <p class="hint">
        Ogni impianto descrive un sistema specifico: un'offerta ricevuta
        (dati di targa, costo, incentivo) o — in futuro — un progetto
        elettrico di dettaglio. Gli scenari lo referenziano e ne ereditano
        potenze, accumulo e investimento.
    </p>

    {#if message}<p class="error">{message}</p>{/if}

    {#if loading}
        <p>Caricamento…</p>
    {:else if designs.length === 0}
        <p class="empty">
            Nessun impianto salvato. Creane uno dalla pagina
            <a href="#/offerta">Analizza un'offerta</a>.
        </p>
    {:else}
        <div class="list">
            {#each designs as item (item.id)}
                <div class="card item-card">
                    <div class="item-body">
                        {#if renamingId === item.id}
                            <form
                                class="rename-form"
                                on:submit|preventDefault={() => commitRename(item)}
                            >
                                <input class="input" bind:value={renameDraft} autofocus required />
                                <button type="submit" class="btn btn-sm btn-primary">Salva</button>
                                <button type="button" class="btn btn-sm btn-ghost"
                                    on:click={() => (renamingId = null)}>Annulla</button>
                            </form>
                        {:else}
                            <h4>
                                {item.name}
                                <span class="badge">{levelLabel(item.design_level)}</span>
                            </h4>
                        {/if}
                        <p class="meta">
                            {item.data.p_ac_kw} kW AC
                            · {item.data.p_dc_kwp ?? item.data.p_ac_kw} kWp DC
                            {#if item.data.storage_kwh}
                                · accumulo {item.data.storage_kwh} kWh
                            {/if}
                            · {Number(item.data.total_cost_eur).toLocaleString("it-IT")} €
                            {#if item.data.tax_bonus?.enabled}
                                · detrazione {Math.round(item.data.tax_bonus.fraction_of_investment * 100)}%
                                in {item.data.tax_bonus.duration_years} anni
                            {/if}
                            {#if item.location_id != null && locationName(item.location_id)}
                                · 📍 {locationName(item.location_id)}
                            {/if}
                        </p>
                        {#if item.description}
                            <p class="meta">{item.description}</p>
                        {/if}
                    </div>
                    <div class="item-actions">
                        {#if deleteConfirmId === item.id}
                            <span class="confirm-label">Eliminare?</span>
                            <button class="btn btn-sm btn-danger"
                                on:click={() => deleteDesign(item.id)}>Sì</button>
                            <button class="btn btn-sm btn-ghost"
                                on:click={() => (deleteConfirmId = null)}>No</button>
                        {:else if renamingId !== item.id}
                            <button class="btn btn-sm btn-ghost" title="Rinomina"
                                on:click={() => startRename(item)}>✏️</button>
                            <button class="btn btn-sm btn-ghost btn-del" title="Elimina"
                                on:click={() => (deleteConfirmId = item.id)}>🗑️</button>
                        {/if}
                    </div>
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
        margin-bottom: 0.5rem;
    }
    .hint {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
        margin: 0 0 1.5rem;
        max-width: 65ch;
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
    .item-body h4 { margin: 0; display: flex; align-items: center; gap: 0.5rem; }
    .badge {
        font-size: 0.72rem;
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        border: 1px solid var(--color-border, #ddd);
        color: var(--color-text-secondary);
        font-weight: 400;
    }
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.85rem;
        margin: 0.25rem 0 0;
    }
    .item-actions { display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0; }
    .rename-form { display: flex; gap: 0.4rem; align-items: center; }
    .rename-form .input { min-width: 16rem; }
    .confirm-label {
        font-size: 0.82rem;
        color: var(--color-danger, #dc3545);
        font-weight: 500;
    }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .error { color: var(--color-danger, #dc3545); }
</style>
