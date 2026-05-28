<!--
    LocationSearch — Phase 14.

    Debounced typeahead that forwards a free-text query to the backend
    geocoding wrapper (``POST /api/external/geocode``) and shows up to N
    candidate locations as a clickable list. The parent listens to the
    ``select`` event to update the map / lat / lon.

    Props:
        placeholder   — input placeholder, default Italian wizard copy.
        debounceMs    — debounce window in ms. Default 500.
        maxResults    — max candidates returned by Nominatim. Default 5.

    Events:
        select        — fired when the user clicks one of the candidates,
                        with detail = the full GeocodeResultResponse object
                        (display_name, latitude, longitude, …).

    Notes:
        - The backend handles the Nominatim ``User-Agent`` requirement and
          can later add caching/rate-limiting in one place.
        - Empty / whitespace queries clear the results list without
          hitting the backend.
-->
<script>
    import { createEventDispatcher } from "svelte";
    import { api } from "../api.js";

    export let placeholder = "Cerca una città, indirizzo o località…";
    export let debounceMs = 500;
    export let maxResults = 5;

    const dispatch = createEventDispatcher();

    let query = "";
    let results = [];
    let loading = false;
    let error = null;
    let debounceTimer = null;

    function onInput() {
        clearTimeout(debounceTimer);
        if (!query || query.trim().length < 3) {
            results = [];
            error = null;
            return;
        }
        debounceTimer = setTimeout(runSearch, debounceMs);
    }

    async function runSearch() {
        loading = true;
        error = null;
        try {
            results = await api.geocode(query, { limit: maxResults });
        } catch (err) {
            error = err.message || "Errore di rete";
            results = [];
        } finally {
            loading = false;
        }
    }

    function pick(result) {
        results = [];
        query = result.display_name;
        dispatch("select", result);
    }
</script>

<div class="search-wrapper">
    <input
        class="search-input"
        type="text"
        bind:value={query}
        on:input={onInput}
        {placeholder}
        autocomplete="off"
    />
    {#if loading}
        <p class="hint">Cerco…</p>
    {/if}
    {#if error}
        <p class="hint error">{error}</p>
    {/if}
    {#if results.length > 0}
        <ul class="result-list">
            {#each results as r}
                <li>
                    <button type="button" class="result-btn" on:click={() => pick(r)}>
                        <span class="result-name">{r.display_name}</span>
                        <span class="result-meta">
                            {r.latitude.toFixed(3)}°, {r.longitude.toFixed(3)}°
                            {#if r.place_type}· {r.place_type}{/if}
                        </span>
                    </button>
                </li>
            {/each}
        </ul>
    {/if}
</div>

<style>
    .search-wrapper {
        position: relative;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
    }
    .search-input {
        width: 100%;
        padding: 0.55rem 0.75rem;
        border: 1px solid var(--border, #d1d5db);
        border-radius: 6px;
        font-size: 0.95rem;
    }
    .hint {
        font-size: 0.85rem;
        color: var(--text-muted, #6b7280);
        margin: 0;
    }
    .hint.error {
        color: var(--danger, #dc2626);
    }
    .result-list {
        list-style: none;
        margin: 0;
        padding: 0;
        border: 1px solid var(--border, #d1d5db);
        border-radius: 6px;
        max-height: 260px;
        overflow-y: auto;
        background: var(--surface, #fff);
    }
    .result-list li + li {
        border-top: 1px solid var(--border, #e5e7eb);
    }
    .result-btn {
        width: 100%;
        text-align: left;
        background: transparent;
        border: 0;
        padding: 0.55rem 0.75rem;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }
    .result-btn:hover {
        background: var(--bg-soft, #f3f4f6);
    }
    .result-name {
        font-weight: 500;
        font-size: 0.92rem;
    }
    .result-meta {
        font-size: 0.78rem;
        color: var(--text-muted, #6b7280);
    }
</style>
