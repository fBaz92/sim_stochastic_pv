<script>
    // Lightweight CSS-grid heatmap — no charting library (CLAUDE.md §3.5).
    //
    // Two modes:
    //  - "continuous": each cell is a number; colour interpolates a viridis-like
    //    ramp between `min`..`max` (auto-ranged when not provided). A legend
    //    shows the value range.
    //  - "categorical": each cell is an integer index into `categories`
    //    (`-1` = empty/no value); colour comes from `categoryColors`. A legend
    //    lists the categories.
    export let matrix = [];          // number[rows][cols]
    export let rowLabels = [];       // string[rows]
    export let colLabels = [];       // string[cols]
    export let mode = "continuous";  // "continuous" | "categorical"
    export let unit = "";            // tooltip suffix for continuous mode
    export let valueDigits = 3;      // decimals shown in continuous tooltips
    export let categories = [];      // string[] for categorical mode
    export let categoryColors = [];  // string[] aligned with `categories`
    export let min = null;           // override continuous range
    export let max = null;
    export let colLabelEvery = 2;    // show every N-th column label to avoid clutter

    // Viridis-ish 5-stop ramp (low → high).
    const RAMP = [
        [68, 1, 84],     // dark purple
        [59, 82, 139],   // blue
        [33, 145, 140],  // teal
        [94, 201, 98],   // green
        [253, 231, 37],  // yellow
    ];

    function lerp(a, b, t) {
        return Math.round(a + (b - a) * t);
    }

    function rampColor(t) {
        const clamped = Math.max(0, Math.min(1, t));
        const seg = clamped * (RAMP.length - 1);
        const i = Math.min(RAMP.length - 2, Math.floor(seg));
        const f = seg - i;
        const c0 = RAMP[i];
        const c1 = RAMP[i + 1];
        return `rgb(${lerp(c0[0], c1[0], f)}, ${lerp(c0[1], c1[1], f)}, ${lerp(c0[2], c1[2], f)})`;
    }

    // Auto range for continuous mode.
    $: flat = matrix.flat().filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
    $: lo = min !== null ? min : (flat.length ? Math.min(...flat) : 0);
    $: hi = max !== null ? max : (flat.length ? Math.max(...flat) : 1);
    $: span = hi - lo || 1;

    function cellColor(value) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return "var(--color-bg-secondary, #f3f4f6)";
        }
        if (mode === "categorical") {
            if (value < 0 || value >= categoryColors.length) return "#e5e7eb";
            return categoryColors[value];
        }
        return rampColor((value - lo) / span);
    }

    function cellTitle(value, r, c) {
        const rl = rowLabels[r] ?? r;
        const cl = colLabels[c] ?? c;
        if (value === null || value === undefined || Number.isNaN(value)) {
            return `${rl} · ${cl}: —`;
        }
        if (mode === "categorical") {
            const name = value >= 0 && value < categories.length ? categories[value] : "—";
            return `${rl} · ${cl}: ${name}`;
        }
        return `${rl} · ${cl}: ${Number(value).toFixed(valueDigits)} ${unit}`.trim();
    }

    $: nCols = colLabels.length || (matrix[0] ? matrix[0].length : 0);
</script>

<div class="heatmap">
    <div
        class="grid"
        style="grid-template-columns: auto repeat({nCols}, 1fr);"
    >
        <!-- header row: empty corner + column labels -->
        <div class="corner"></div>
        {#each colLabels as cl, c}
            <div class="col-label">{c % colLabelEvery === 0 ? cl : ""}</div>
        {/each}

        {#each matrix as row, r}
            <div class="row-label">{rowLabels[r] ?? r}</div>
            {#each row as value, c}
                <div
                    class="cell"
                    style="background:{cellColor(value)};"
                    title={cellTitle(value, r, c)}
                ></div>
            {/each}
        {/each}
    </div>

    {#if mode === "continuous"}
        <div class="legend">
            <span class="legend-min">{lo.toFixed(valueDigits)}</span>
            <span class="legend-ramp"></span>
            <span class="legend-max">{hi.toFixed(valueDigits)} {unit}</span>
        </div>
    {:else}
        <div class="legend cats">
            {#each categories as cat, i}
                <span class="cat">
                    <span class="swatch" style="background:{categoryColors[i]};"></span>
                    {cat}
                </span>
            {/each}
        </div>
    {/if}
</div>

<style>
    .heatmap {
        width: 100%;
    }
    .grid {
        display: grid;
        gap: 1px;
        width: 100%;
    }
    .cell {
        aspect-ratio: 1 / 1;
        min-height: 10px;
        border-radius: 1px;
        cursor: help;
    }
    .corner {
        min-width: 28px;
    }
    .col-label {
        font-size: 0.6rem;
        color: var(--color-text-secondary, #6b7280);
        text-align: center;
        align-self: end;
        white-space: nowrap;
    }
    .row-label {
        font-size: 0.65rem;
        color: var(--color-text-secondary, #6b7280);
        padding-right: 6px;
        text-align: right;
        align-self: center;
        white-space: nowrap;
    }
    .legend {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.6rem;
        font-size: 0.7rem;
        color: var(--color-text-secondary, #6b7280);
    }
    .legend-ramp {
        flex: 1;
        height: 10px;
        border-radius: 3px;
        background: linear-gradient(
            to right,
            rgb(68, 1, 84),
            rgb(59, 82, 139),
            rgb(33, 145, 140),
            rgb(94, 201, 98),
            rgb(253, 231, 37)
        );
    }
    .legend.cats {
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .cat {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .swatch {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        display: inline-block;
    }
</style>
