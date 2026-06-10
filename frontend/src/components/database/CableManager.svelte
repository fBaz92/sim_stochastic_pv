<script>
    /**
     * CableManager — Database tab for the DC cable catalogue.
     *
     * CRUD over /api/cables plus a per-cable mini product sheet: ohmic
     * loss vs current at three run lengths (same ρ(70 °C) convention as
     * the designer engine), with the Iz thermal limit marked.
     */
    import { onMount } from "svelte";
    import { api } from "../../api";
    import ResultsChart from "../ResultsChart.svelte";

    let items = [];
    let loading = false;
    let showForm = false;
    let editingId = null;
    let deleteConfirmId = null;
    let detailId = null;

    const emptyForm = () => ({
        name: "",
        manufacturer: "",
        section_mm2: 6,
        material: "copper",
        price_eur_per_m: null,
        iz_a: null,
        notes: "",
    });
    let formData = emptyForm();

    async function loadItems() {
        loading = true;
        try {
            items = await api.listCables();
        } catch (e) {
            alert("Errore nel caricamento dei cavi: " + e.message);
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
                section_mm2: Number(formData.section_mm2),
                price_eur_per_m: formData.price_eur_per_m !== null && formData.price_eur_per_m !== ""
                    ? Number(formData.price_eur_per_m) : null,
                iz_a: formData.iz_a !== null && formData.iz_a !== ""
                    ? Number(formData.iz_a) : null,
            };
            if (editingId) await api.updateCable(editingId, payload);
            else await api.createCable(payload);
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
            await api.deleteCable(id);
            deleteConfirmId = null;
            await loadItems();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    // Mirrors the designer engine convention (cables.py): copper at the
    // typical 70 °C operating temperature of sun-exposed solar cables.
    const RHO_70C = 0.0175 * (1 + 0.00393 * (70 - 20));

    function lossConfig(item) {
        const iMax = item.iz_a ?? 25;
        const currents = Array.from({ length: 26 }, (_, k) => (iMax * k) / 25);
        const lengths = [10, 30, 50];
        const colors = ["#34d399", "#f59e0b", "#ef4444"];
        return {
            type: "line",
            data: {
                labels: currents.map((i) => i.toFixed(0)),
                datasets: lengths.map((L, k) => ({
                    label: `${L} m (andata)`,
                    data: currents.map(
                        (i) => (2 * L * RHO_70C / item.section_mm2) * i * i,
                    ),
                    borderColor: colors[k],
                    pointRadius: 0,
                    fill: false,
                })),
            },
            options: {
                plugins: { legend: { display: true } },
                scales: {
                    x: { title: { display: true, text: "Corrente (A)" } },
                    y: { title: { display: true, text: "Perdita per stringa (W)" } },
                },
            },
        };
    }

    onMount(loadItems);
</script>

<div>
    <div class="header-actions">
        <h2>Cavi DC</h2>
        <button class="btn btn-primary" on:click={() => {
            showForm = !showForm;
            if (!showForm) { editingId = null; formData = emptyForm(); }
        }}>
            {showForm ? "Annulla" : "+ Aggiungi cavo"}
        </button>
    </div>
    <p class="hint">
        Listino usato dal designer per costo del rame e verifica di
        portata Iz nella tabella di confronto sezioni.
    </p>

    {#if showForm}
        <div class="card form-card">
            <form on:submit|preventDefault={handleSubmit}>
                <div class="form-grid">
                    <label>Nome *<input class="input" bind:value={formData.name} required /></label>
                    <label>Marca<input class="input" bind:value={formData.manufacturer} /></label>
                    <label>Sezione (mm²) *<input class="input" type="number" step="0.5" min="0.5" bind:value={formData.section_mm2} required /></label>
                    <label>Prezzo (€/m)<input class="input" type="number" step="0.05" min="0" bind:value={formData.price_eur_per_m} /></label>
                    <label>Portata Iz (A)<input class="input" type="number" step="1" min="1" bind:value={formData.iz_a} /></label>
                    <label>Note<input class="input" bind:value={formData.notes} /></label>
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
        <p class="empty">Nessun cavo nel catalogo.</p>
    {:else}
        <div class="list">
            {#each items as item (item.id)}
                <div class="card item-card">
                    <div class="row">
                        <div class="item-body">
                            <h4>{item.name}</h4>
                            <p class="meta">
                                {item.section_mm2} mm² · {item.material}
                                {#if item.price_eur_per_m != null} · {item.price_eur_per_m} €/m{/if}
                                {#if item.iz_a != null} · Iz {item.iz_a} A{/if}
                            </p>
                        </div>
                        <div class="item-actions">
                            {#if deleteConfirmId === item.id}
                                <span class="confirm-label">Eliminare?</span>
                                <button class="btn btn-sm btn-danger" on:click={() => handleDelete(item.id)}>Sì</button>
                                <button class="btn btn-sm btn-ghost" on:click={() => (deleteConfirmId = null)}>No</button>
                            {:else}
                                <button class="btn btn-sm btn-ghost"
                                    on:click={() => (detailId = detailId === item.id ? null : item.id)}
                                >{detailId === item.id ? "Chiudi" : "📈 Perdite"}</button>
                                <button class="btn btn-sm btn-ghost" title="Modifica" on:click={() => startEdit(item)}>✏️</button>
                                <button class="btn btn-sm btn-ghost btn-del" title="Elimina" on:click={() => (deleteConfirmId = item.id)}>🗑️</button>
                            {/if}
                        </div>
                    </div>
                    {#if detailId === item.id}
                        {@const cfg = lossConfig(item)}
                        <div class="chart">
                            <ResultsChart type={cfg.type} data={cfg.data} options={cfg.options} downloadFilename="perdite_cavo" />
                        </div>
                        <p class="hint small">
                            Perdita ohmica andata+ritorno a 70 °C
                            (ρ = {RHO_70C.toFixed(4)} Ω·mm²/m), fino alla portata Iz.
                        </p>
                    {/if}
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .header-actions { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .hint { color: var(--color-text-secondary); font-size: 0.9rem; margin: 0 0 1.5rem; max-width: 65ch; }
    .hint.small { font-size: 0.8rem; margin: 0.4rem 0 0; }
    .form-card { padding: 1.25rem; margin-bottom: 1.5rem; background-color: var(--color-bg-tertiary); }
    .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem 1rem; }
    .form-grid label { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.82rem; color: var(--color-text-secondary); }
    .form-actions { margin-top: 1rem; display: flex; justify-content: flex-end; }
    .list { display: flex; flex-direction: column; gap: 0.75rem; }
    .item-card { padding: 1rem; height: auto; }
    .row { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
    .item-body h4 { margin: 0; }
    .meta { color: var(--color-text-secondary); font-size: 0.85rem; margin: 0.25rem 0 0; }
    .item-actions { display: flex; gap: 0.4rem; align-items: center; flex-shrink: 0; }
    .confirm-label { font-size: 0.82rem; color: var(--color-danger, #dc3545); font-weight: 500; }
    .chart { height: 240px; position: relative; margin-top: 0.75rem; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
</style>
