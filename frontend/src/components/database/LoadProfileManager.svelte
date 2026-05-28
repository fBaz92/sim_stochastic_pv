<script>
    import { onMount } from "svelte";
    import { api } from "../../api";
    import MonthInput from "../forms/MonthInput.svelte";
    import MonthlyProfileEditor from "../forms/MonthlyProfileEditor.svelte";
    import WeeklyPatternEditor from "../forms/WeeklyPatternEditor.svelte";

    /**
     * Load Profile Manager — Phase 8 edition.
     *
     * A load profile in the DB now represents a *complete consumption
     * personality* of a household: how the user consumes when at home and
     * when away. The scenario only specifies how many days per month the
     * user spends at home — i.e. the load profile is a property of the
     * *site/user*, the days-distribution is a property of the *scenario*.
     *
     * Profile types supported:
     *   - "arera"        — Italian ARERA standard (no data needed)
     *   - "custom"       — single monthly average (W), legacy
     *   - "custom_24h"   — single 12×24 hourly pattern, legacy
     *   - "home_away"    — NEW: two sub-profiles (home + away), each one
     *                     can be ARERA or one of the custom shapes.
     *
     * The legacy types stay editable for backward compat.
     */

    let items = [];
    let showForm = false;

    /** ID del profilo in editing; null = modalità creazione. */
    let editingId = null;

    /** ID del profilo con conferma eliminazione in attesa. */
    let deleteConfirmId = null;

    const emptyFormData = () => ({
        name: "",
        profile_type: "home_away",
        data: {
            // legacy
            monthly_w: Array(12).fill(100),
            monthly_24h_w: Array.from({ length: 12 }, () => Array(24).fill(100)),
            // home_away
            home_type: "custom_24h",
            home_monthly_w: Array(12).fill(300),
            home_monthly_24h_w: Array.from({ length: 12 }, () => Array(24).fill(300)),
            home_weekly_pattern_w: Array.from({ length: 7 }, () => Array(24).fill(100)),
            away_type: "custom_24h",
            away_monthly_w: Array(12).fill(100),
            away_monthly_24h_w: Array.from({ length: 12 }, () => Array(24).fill(100)),
            away_weekly_pattern_w: Array.from({ length: 7 }, () => Array(24).fill(100)),
        },
    });

    // Default state of the form. The "data" object carries all the variants
    // — we serialise only the relevant sub-tree based on `profile_type`.
    let newItem = emptyFormData();

    // Active side ("home" or "away") inside the home_away editor.
    let activeSide = "home";

    async function load() {
        items = await api.listLoadProfiles();
    }

    /**
     * Pre-popola il form con i dati di un profilo esistente.
     * Ricostruisce lo stato interno del form dalla struttura JSON salvata.
     */
    function startEdit(item) {
        editingId = item.id;
        deleteConfirmId = null;
        const d = item.data || {};
        const f = emptyFormData();
        f.name = item.name;
        f.profile_type = item.profile_type;

        if (item.profile_type === "custom") {
            f.data.monthly_w = d.monthly_w ?? Array(12).fill(100);
        } else if (item.profile_type === "custom_24h") {
            f.data.monthly_24h_w = d.monthly_24h_w ?? Array.from({ length: 12 }, () => Array(24).fill(100));
        } else if (item.profile_type === "home_away") {
            const home = d.home || {};
            const away = d.away || {};
            // Home side
            if (home.type === "arera") {
                f.data.home_type = "arera";
            } else if (home.type === "weekly" || home.weekly_pattern_w) {
                f.data.home_type = "weekly";
                if (home.monthly_24h_w) f.data.home_monthly_24h_w = home.monthly_24h_w;
                else if (home.monthly_w) f.data.home_monthly_w = home.monthly_w;
                if (home.weekly_pattern_w) f.data.home_weekly_pattern_w = home.weekly_pattern_w;
            } else if (home.monthly_24h_w) {
                f.data.home_type = "custom_24h";
                f.data.home_monthly_24h_w = home.monthly_24h_w;
            } else if (home.monthly_w) {
                f.data.home_type = "custom";
                f.data.home_monthly_w = home.monthly_w;
            }
            // Away side
            if (away.type === "arera") {
                f.data.away_type = "arera";
            } else if (away.type === "weekly" || away.weekly_pattern_w) {
                f.data.away_type = "weekly";
                if (away.monthly_24h_w) f.data.away_monthly_24h_w = away.monthly_24h_w;
                else if (away.monthly_w) f.data.away_monthly_w = away.monthly_w;
                if (away.weekly_pattern_w) f.data.away_weekly_pattern_w = away.weekly_pattern_w;
            } else if (away.monthly_24h_w) {
                f.data.away_type = "custom_24h";
                f.data.away_monthly_24h_w = away.monthly_24h_w;
            } else if (away.monthly_w) {
                f.data.away_type = "custom";
                f.data.away_monthly_w = away.monthly_w;
            }
        }

        newItem = f;
        showForm = true;
        document.getElementById("load-profile-form")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function cancelForm() {
        showForm = false;
        editingId = null;
        newItem = emptyFormData();
    }

    async function handleDelete(id) {
        try {
            await api.deleteLoadProfile(id);
            deleteConfirmId = null;
            await load();
        } catch (e) {
            alert("Errore nella cancellazione: " + e.message);
        }
    }

    /**
     * Convert the form state into the JSON payload expected by the backend.
     *
     * For home_away profiles the structure is:
     *
     *   {
     *     kind: "home_away",
     *     home: { type: "arera" } | { monthly_w: [...] } | { monthly_24h_w: [[...]] },
     *     away: { ... same options ... }
     *   }
     */
    /**
     * Serialize a single "side" (home or away) of a home_away profile.
     *
     * For the new "weekly" type the payload includes both the monthly baseline
     * (as monthly_24h_w or monthly_w) and the 7×24 weekly_pattern_w, plus
     * the discriminator `type: "weekly"` so the backend dispatcher can
     * recognise it in `_build_single_load_profile_factory`.
     */
    function buildSidePayload(sideType, monthlyW, monthly24hW, weeklyPatternW) {
        if (sideType === "arera") return { type: "arera" };
        if (sideType === "custom") return { monthly_w: monthlyW };
        if (sideType === "weekly") {
            return {
                type: "weekly",
                monthly_24h_w: monthly24hW,
                weekly_pattern_w: weeklyPatternW,
            };
        }
        return { monthly_24h_w: monthly24hW };
    }

    async function handleSubmit() {
        const payloadData = {};
        if (newItem.profile_type === "custom") {
            payloadData.monthly_w = newItem.data.monthly_w;
        } else if (newItem.profile_type === "custom_24h") {
            payloadData.monthly_24h_w = newItem.data.monthly_24h_w;
        } else if (newItem.profile_type === "home_away") {
            payloadData.kind = "home_away";
            payloadData.home = buildSidePayload(
                newItem.data.home_type,
                newItem.data.home_monthly_w,
                newItem.data.home_monthly_24h_w,
                newItem.data.home_weekly_pattern_w,
            );
            payloadData.away = buildSidePayload(
                newItem.data.away_type,
                newItem.data.away_monthly_w,
                newItem.data.away_monthly_24h_w,
                newItem.data.away_weekly_pattern_w,
            );
        }

        const payload = {
            name: newItem.name,
            profile_type: newItem.profile_type,
            data: payloadData,
        };
        try {
            if (editingId != null) {
                await api.updateLoadProfile(editingId, payload);
            } else {
                await api.createLoadProfile(payload);
            }
        } catch (e) {
            alert("Errore nel salvataggio: " + e.message);
            return;
        }
        cancelForm();
        load();
    }

    onMount(load);

    /**
     * Render a short, human-readable summary for the list of saved profiles.
     * Avoids dumping raw JSON on the user.
     */
    function describeSide(side) {
        if (!side) return "?";
        if (side.type === "arera") return "ARERA";
        if (side.type === "weekly" || side.weekly_pattern_w) return "settimanale";
        if (side.monthly_24h_w) return "24h";
        return "mensile";
    }

    function describeItem(item) {
        if (item.profile_type === "home_away") {
            const homeKind = describeSide(item.data?.home);
            const awayKind = describeSide(item.data?.away);
            return `casa/via (${homeKind} / ${awayKind})`;
        }
        return item.profile_type;
    }
</script>

<div class="manager">
    <div class="toolbar">
        <h2>Profili di carico</h2>
        <button class="btn btn-primary" on:click={() => {
            if (showForm && editingId === null) { cancelForm(); }
            else { cancelForm(); showForm = true; }
        }}>
            {showForm ? "Annulla" : "+ Aggiungi"}
        </button>
    </div>

    {#if showForm}
        <div id="load-profile-form" class="card form-card">
            <h3>{editingId ? "Modifica profilo" : "Nuovo profilo"}</h3>
            {#if editingId}
                <p class="edit-hint">Stai modificando un profilo esistente. Puoi anche rinominarlo.</p>
            {/if}
            <form
                on:submit={(e) => {
                    e.preventDefault();
                    handleSubmit();
                }}
            >
                <div class="form-group">
                    <label class="label" for="load-profile-name">Name</label>
                    <input
                        id="load-profile-name"
                        class="input"
                        bind:value={newItem.name}
                        required
                    />
                </div>
                <div class="form-group">
                    <label class="label" for="load-profile-type">Tipologia</label>
                    <select
                        id="load-profile-type"
                        class="select"
                        bind:value={newItem.profile_type}
                    >
                        <option value="home_away">
                            Casa + via (consigliato)
                        </option>
                        <option value="custom">Solo media mensile (W)</option>
                        <option value="custom_24h">Solo 12×24 (W)</option>
                    </select>
                    <p class="hint">
                        "Casa + via" descrive come consumi nei due regimi:
                        quanti giorni sei a casa lo decidi poi nello scenario.
                    </p>
                </div>

                {#if newItem.profile_type === "custom"}
                    <MonthInput
                        label="Average Watts"
                        bind:values={newItem.data.monthly_w}
                    />
                {:else if newItem.profile_type === "custom_24h"}
                    <MonthlyProfileEditor
                        label="24h Pattern"
                        bind:values={newItem.data.monthly_24h_w}
                    />
                {:else if newItem.profile_type === "home_away"}
                    <div class="side-tabs">
                        <button
                            type="button"
                            class="tab-btn"
                            class:active={activeSide === "home"}
                            on:click={() => (activeSide = "home")}
                            >Quando sono a casa</button
                        >
                        <button
                            type="button"
                            class="tab-btn"
                            class:active={activeSide === "away"}
                            on:click={() => (activeSide = "away")}
                            >Quando sono via</button
                        >
                    </div>

                    {#if activeSide === "home"}
                        <div class="form-group">
                            <label class="label" for="home-side-type"
                                >Forma del profilo "a casa"</label
                            >
                            <select
                                id="home-side-type"
                                class="select"
                                bind:value={newItem.data.home_type}
                            >
                                <option value="custom_24h">12×24 W (mensile)</option>
                                <option value="weekly">Pattern settimanale 7×24 W ✨</option>
                                <option value="custom">Media mensile (W)</option>
                                <option value="arera">ARERA</option>
                            </select>
                        </div>
                        {#if newItem.data.home_type === "weekly"}
                            <MonthlyProfileEditor
                                label="A casa — baseline mensile 12×24 (W)"
                                bind:values={newItem.data.home_monthly_24h_w}
                            />
                            <WeeklyPatternEditor
                                bind:values={newItem.data.home_weekly_pattern_w}
                            />
                        {:else if newItem.data.home_type === "custom_24h"}
                            <MonthlyProfileEditor
                                label="A casa — 24h × 12 mesi (W)"
                                bind:values={newItem.data.home_monthly_24h_w}
                            />
                        {:else if newItem.data.home_type === "custom"}
                            <MonthInput
                                label="A casa — media mensile (W)"
                                bind:values={newItem.data.home_monthly_w}
                            />
                        {:else}
                            <p class="hint">
                                Profilo ARERA: non servono parametri.
                            </p>
                        {/if}
                    {:else}
                        <div class="form-group">
                            <label class="label" for="away-side-type"
                                >Forma del profilo "via"</label
                            >
                            <select
                                id="away-side-type"
                                class="select"
                                bind:value={newItem.data.away_type}
                            >
                                <option value="custom_24h">12×24 W (mensile)</option>
                                <option value="weekly">Pattern settimanale 7×24 W ✨</option>
                                <option value="custom">Media mensile (W)</option>
                                <option value="arera">ARERA</option>
                            </select>
                        </div>
                        {#if newItem.data.away_type === "weekly"}
                            <MonthlyProfileEditor
                                label="Via — baseline mensile 12×24 (W)"
                                bind:values={newItem.data.away_monthly_24h_w}
                            />
                            <WeeklyPatternEditor
                                bind:values={newItem.data.away_weekly_pattern_w}
                            />
                        {:else if newItem.data.away_type === "custom_24h"}
                            <MonthlyProfileEditor
                                label="Via — 24h × 12 mesi (W)"
                                bind:values={newItem.data.away_monthly_24h_w}
                            />
                        {:else if newItem.data.away_type === "custom"}
                            <MonthInput
                                label="Via — media mensile (W)"
                                bind:values={newItem.data.away_monthly_w}
                            />
                        {:else}
                            <p class="hint">
                                Profilo ARERA: non servono parametri.
                            </p>
                        {/if}
                    {/if}
                {/if}

                <div class="form-actions">
                    <button type="button" class="btn btn-ghost" on:click={cancelForm}>Annulla</button>
                    <button class="btn btn-primary" type="submit">
                        {editingId ? "Aggiorna" : "Salva profilo"}
                    </button>
                </div>
            </form>
        </div>
    {/if}

    {#if items.length === 0}
        <p class="empty">Nessun profilo salvato. Aggiungine uno.</p>
    {:else}
        <div class="list">
            {#each items as item (item.id)}
                <div class="card item-card" class:editing={editingId === item.id}>
                    <div class="item-body">
                        <h3>{item.name}</h3>
                        <p class="meta">{describeItem(item)}</p>
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
    .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
    .item-card { padding: 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
    .item-card.editing { border-color: var(--color-accent, #0d6efd); outline: 2px solid var(--color-accent, #0d6efd); outline-offset: 2px; }
    .item-body { flex: 1; }
    .item-actions { display: flex; align-items: center; gap: 0.4rem; flex-shrink: 0; }
    .confirm-label { font-size: 0.82rem; color: var(--color-danger, #dc3545); font-weight: 500; }
    .form-card { padding: 1.5rem; margin-bottom: 2rem; }
    .edit-hint { font-size: 0.82rem; color: var(--color-text-muted, #6c757d); margin: 0 0 1rem; }
    .form-actions { margin-top: 1rem; display: flex; gap: 0.5rem; justify-content: flex-end; }
    .meta { color: var(--color-text-secondary); font-size: 0.9rem; }
    .hint { color: var(--color-text-secondary); font-size: 0.85rem; margin-top: 0.25rem; }
    .empty { color: var(--color-text-muted, #6c757d); font-style: italic; }
    .side-tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; border-bottom: 1px solid var(--color-border, #e2e8f0); }
    .tab-btn { background: none; border: none; padding: 0.5rem 1rem; cursor: pointer; color: var(--color-text-secondary); border-bottom: 2px solid transparent; font-weight: 500; }
    .tab-btn.active { color: var(--color-primary, #0d6efd); border-bottom-color: var(--color-primary, #0d6efd); }
</style>
