<script>
    import { onMount } from "svelte";
    import { api } from "../../api";
    import MonthInput from "../forms/MonthInput.svelte";
    import MonthlyProfileEditor from "../forms/MonthlyProfileEditor.svelte";

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

    // Default state of the form. The "data" object carries all the variants
    // — we serialise only the relevant sub-tree based on `profile_type`.
    let newItem = {
        name: "",
        profile_type: "home_away",
        data: {
            // legacy
            monthly_w: Array(12).fill(100),
            monthly_24h_w: Array.from({ length: 12 }, () => Array(24).fill(100)),
            // home_away
            home_type: "custom_24h",
            home_monthly_w: Array(12).fill(300),
            home_monthly_24h_w: Array.from({ length: 12 }, () =>
                Array(24).fill(300),
            ),
            away_type: "custom_24h",
            away_monthly_w: Array(12).fill(100),
            away_monthly_24h_w: Array.from({ length: 12 }, () =>
                Array(24).fill(100),
            ),
        },
    };

    // Active side ("home" or "away") inside the home_away editor.
    let activeSide = "home";

    async function load() {
        items = await api.listLoadProfiles();
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
    function buildSidePayload(sideType, monthlyW, monthly24hW) {
        if (sideType === "arera") return { type: "arera" };
        if (sideType === "custom") return { monthly_w: monthlyW };
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
            );
            payloadData.away = buildSidePayload(
                newItem.data.away_type,
                newItem.data.away_monthly_w,
                newItem.data.away_monthly_24h_w,
            );
        }

        await api.createLoadProfile({
            name: newItem.name,
            profile_type: newItem.profile_type,
            data: payloadData,
        });
        showForm = false;
        load();
    }

    onMount(load);

    /**
     * Render a short, human-readable summary for the list of saved profiles.
     * Avoids dumping raw JSON on the user.
     */
    function describeItem(item) {
        if (item.profile_type === "home_away") {
            const homeKind = item.data?.home?.type
                ? "arera"
                : item.data?.home?.monthly_24h_w
                  ? "24h"
                  : "monthly";
            const awayKind = item.data?.away?.type
                ? "arera"
                : item.data?.away?.monthly_24h_w
                  ? "24h"
                  : "monthly";
            return `home/away (${homeKind} / ${awayKind})`;
        }
        return item.profile_type;
    }
</script>

<div class="manager">
    <div class="toolbar">
        <h2>Load Profiles</h2>
        <button class="btn btn-primary" on:click={() => (showForm = !showForm)}>
            {showForm ? "Cancel" : "Add Profile"}
        </button>
    </div>

    {#if showForm}
        <div class="card form-card">
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
                                <option value="custom_24h">12×24 W</option>
                                <option value="custom">Media mensile (W)</option>
                                <option value="arera">ARERA</option>
                            </select>
                        </div>
                        {#if newItem.data.home_type === "custom_24h"}
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
                                <option value="custom_24h">12×24 W</option>
                                <option value="custom">Media mensile (W)</option>
                                <option value="arera">ARERA</option>
                            </select>
                        </div>
                        {#if newItem.data.away_type === "custom_24h"}
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

                <button class="btn btn-primary" type="submit"
                    >Save Profile</button
                >
            </form>
        </div>
    {/if}

    <div class="list">
        {#each items as item}
            <div class="card item-card">
                <h3>{item.name}</h3>
                <p class="meta">{describeItem(item)}</p>
            </div>
        {/each}
    </div>
</div>

<style>
    .toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    .item-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .form-card {
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .meta {
        color: var(--color-text-secondary);
        font-size: 0.9rem;
    }
    .hint {
        color: var(--color-text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .side-tabs {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--color-border, #e2e8f0);
    }
    .tab-btn {
        background: none;
        border: none;
        padding: 0.5rem 1rem;
        cursor: pointer;
        color: var(--color-text-secondary);
        border-bottom: 2px solid transparent;
        font-weight: 500;
    }
    .tab-btn.active {
        color: var(--color-primary, #0d6efd);
        border-bottom-color: var(--color-primary, #0d6efd);
    }
</style>
