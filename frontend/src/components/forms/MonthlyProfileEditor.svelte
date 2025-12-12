<script>
    import HourlyInput from "./HourlyInput.svelte";

    export let values = Array.from({ length: 12 }, () => Array(24).fill(0));
    export let label = "Monthly 24h Profile";

    let selectedMonth = 0;

    const MONTHS = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ];

    $: currentMonthValues = values[selectedMonth]?.slice() ?? Array(24).fill(0);

    function updateMonthValues(updatedValues) {
        const cloned = values.map((monthValues, index) =>
            index === selectedMonth ? [...updatedValues] : [...monthValues],
        );
        values = cloned;
    }

    function handleHourlyChange(event) {
        updateMonthValues(event.detail);
    }

    function copyToAll() {
        const template = currentMonthValues.slice();
        values = values.map(() => template.slice());
    }
</script>

<div class="card monthly-editor">
    <div class="editor-header">
        <h3>{label}</h3>
        <button class="btn btn-outline btn-sm" on:click={copyToAll}
            >Copy {MONTHS[selectedMonth]} to All</button
        >
    </div>

    <div class="month-selector">
        {#each MONTHS as m, i}
            <button
                class="selector-btn"
                class:active={selectedMonth === i}
                on:click={() => (selectedMonth = i)}
            >
                {m}
            </button>
        {/each}
    </div>

    <div class="editor-content">
        <!-- Force remount or ensure binding updates -->
        <HourlyInput
            label={`${MONTHS[selectedMonth]} Profile (Avg Watts)`}
            values={currentMonthValues}
            on:change={handleHourlyChange}
        />
    </div>
</div>

<style>
    .monthly-editor {
        padding: 1rem;
        background: var(--color-bg-secondary);
        border: 1px solid var(--color-border);
    }
    .editor-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .editor-header h3 {
        font-size: 1rem;
        margin: 0;
    }
    .month-selector {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--color-border);
        padding-bottom: 0.5rem;
    }
    .selector-btn {
        background: none;
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        padding: 0.25rem 0.5rem;
        font-size: 0.85rem;
        cursor: pointer;
        color: var(--color-text-secondary);
        transition: all 0.2s;
    }
    .selector-btn:hover {
        background: var(--color-bg-tertiary);
        color: var(--color-text-primary);
        border-color: var(--color-border-hover);
    }
    .selector-btn.active {
        background: var(--color-accent);
        color: white;
        font-weight: 600;
        border-color: var(--color-accent);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
