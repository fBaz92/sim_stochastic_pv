<script>
    /**
     * PresenceCalendarEditor — the occupancy backbone of a home/away profile.
     *
     * Twelve month rows describe *when the building is used*: weekends on/off,
     * how many full weeks, a few extra weekdays, and a visit probability that
     * widens the band. A live "~X giorni" badge per month and an annual total
     * make the consequence visible immediately. The maths mirror the backend
     * (see ../../lib/presence.js) so nothing needs the server to update.
     *
     * Props:
     *   - calendar: { months: [12 × {weekends, full_weeks, extra_weekdays,
     *     visit_probability}] }. Re-initialised when the parent swaps the
     *     object identity (profile load / level transition).
     *   - compact: hide the per-month extras (extra weekdays + visit slider)
     *     for the minimal "Bolletta" view.
     * Dispatches `change` with the edited calendar on any edit / preset.
     */
    import { createEventDispatcher } from "svelte";
    import {
        normalizeCalendar,
        monthMinMax,
        annualPresenceFraction,
        MONTH_NAMES,
        PRESENCE_PRESETS,
    } from "../../lib/presence.js";

    export let calendar = null;
    export let compact = false;

    const dispatch = createEventDispatcher();

    let months = normalizeCalendar(calendar).months;

    // Re-sync when the parent hands us a different calendar object (a profile
    // was loaded, a preset applied upstream, or a level transition occurred).
    let lastExternal = calendar;
    $: if (calendar !== lastExternal) {
        lastExternal = calendar;
        months = normalizeCalendar(calendar).months;
    }

    function emit() {
        months = months; // poke Svelte reactivity for the in-place edits
        dispatch("change", { months: months.map((m) => ({ ...m })) });
    }

    function applyPreset(key) {
        months = normalizeCalendar(PRESENCE_PRESETS[key].calendar).months;
        emit();
    }

    $: bands = months.map((m, i) => monthMinMax(m, i));
    $: totalDays = Math.round(
        months.reduce((acc, _m, i) => acc + bands[i].expected, 0),
    );
    $: fraction = annualPresenceFraction({ months });

    function badge(i) {
        const b = bands[i];
        const lo = Math.round(b.min);
        const hi = Math.round(b.max);
        return hi > lo ? `${lo}–${hi} gg` : `${lo} gg`;
    }
</script>

<div class="presence" class:compact>
    <div class="presets">
        <span class="presets-label">Preset:</span>
        {#each Object.entries(PRESENCE_PRESETS) as [key, p]}
            <button type="button" class="chip" on:click={() => applyPreset(key)}>{p.label}</button>
        {/each}
    </div>

    <div class="table" role="table">
        <div class="head" role="row">
            <span class="c-month">Mese</span>
            <span class="c-we" title="I weekend sono giorni in casa">Weekend</span>
            <span class="c-weeks" title="Settimane intere passate in casa (0–5)">Settimane</span>
            {#if !compact}
                <span class="c-extra" title="Giorni feriali extra in casa (0–7)">Extra gg</span>
                <span class="c-visit" title="Probabilità di una visita nei giorni altrimenti via">Prob. visita</span>
            {/if}
            <span class="c-badge">In casa</span>
        </div>

        {#each months as m, i}
            <div class="rowm" role="row">
                <span class="c-month">{MONTH_NAMES[i]}</span>
                <span class="c-we">
                    <input type="checkbox" bind:checked={m.weekends} on:change={emit} aria-label="Weekend a casa {MONTH_NAMES[i]}" />
                </span>
                <span class="c-weeks">
                    <input class="num" type="number" min="0" max="5" step="1" bind:value={m.full_weeks} on:input={emit} />
                </span>
                {#if !compact}
                    <span class="c-extra">
                        <input class="num" type="number" min="0" max="7" step="1" bind:value={m.extra_weekdays} on:input={emit} />
                    </span>
                    <span class="c-visit">
                        <input type="range" min="0" max="1" step="0.05" bind:value={m.visit_probability} on:input={emit} />
                        <span class="visit-pct">{Math.round(m.visit_probability * 100)}%</span>
                    </span>
                {/if}
                <span class="c-badge"><span class="day-badge">{badge(i)}</span></span>
            </div>
        {/each}
    </div>

    <div class="summary">
        <span><strong>{totalDays}</strong> giorni/anno in casa</span>
        <span class="frac">({(fraction * 100).toFixed(0)}% dell'anno)</span>
    </div>
</div>

<style>
    .presence { font-size: 0.85rem; }
    .presets { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin-bottom: 0.75rem; }
    .presets-label { color: var(--color-text-secondary, #6b7280); font-size: 0.8rem; }
    .chip { border: 1px solid var(--color-border, #d1d5db); background: var(--color-bg-secondary, #f9fafb); border-radius: 999px; padding: 0.2rem 0.6rem; font-size: 0.78rem; cursor: pointer; }
    .chip:hover { border-color: var(--color-accent, #3b82f6); }

    .table { display: flex; flex-direction: column; gap: 2px; }
    .head, .rowm {
        display: grid;
        grid-template-columns: 5.5rem 4rem 5rem 4.5rem 1fr 4.5rem;
        align-items: center;
        gap: 0.4rem;
    }
    .compact .head, .compact .rowm {
        grid-template-columns: 6rem 4rem 5rem 5rem;
    }
    .head { font-size: 0.68rem; color: var(--color-text-secondary, #6b7280); text-transform: uppercase; letter-spacing: 0.02em; padding-bottom: 2px; border-bottom: 1px solid var(--color-border, #e5e7eb); }
    .rowm { padding: 1px 0; }
    .rowm:nth-child(even) { background: var(--color-bg-secondary, #f9fafb); }
    .c-month { font-weight: 500; }
    .c-we, .c-weeks, .c-extra, .c-badge { text-align: center; }
    .c-visit { display: flex; align-items: center; gap: 0.35rem; }
    .c-visit input[type="range"] { flex: 1; min-width: 40px; }
    .visit-pct { font-size: 0.72rem; color: var(--color-text-secondary, #6b7280); width: 2.4rem; text-align: right; }
    .num { width: 3rem; padding: 2px 4px; text-align: center; }
    .day-badge { display: inline-block; background: var(--color-accent, #1d4ed8); color: #fff; border-radius: 4px; padding: 1px 6px; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
    .summary { margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid var(--color-border, #e5e7eb); display: flex; align-items: baseline; gap: 0.5rem; }
    .summary strong { color: var(--color-accent, #1d4ed8); font-size: 1.05rem; }
    .frac { color: var(--color-text-secondary, #6b7280); }
</style>
