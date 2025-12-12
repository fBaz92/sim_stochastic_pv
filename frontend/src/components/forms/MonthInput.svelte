<script context="module">
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
</script>

<script>
    export let values = Array(12).fill(0);
    export let label = "Monthly Values";

    function handleInput(event, index) {
        const updated = [...values];
        const numericValue = event.currentTarget.valueAsNumber;
        updated[index] = Number.isNaN(numericValue) ? 0 : numericValue;
        values = updated;
    }
</script>

<div class="month-input-group">
    <span class="label">{label}</span>
    <div class="grid">
        {#each values as val, i}
            <div class="month-field">
                <span class="month-label">{MONTHS[i]}</span>
                <input
                    type="number"
                    step="0.1"
                    class="input month-input"
                    value={val}
                    on:input={(event) => handleInput(event, i)}
                />
            </div>
        {/each}
    </div>
</div>

<style>
    .month-input-group {
        margin-bottom: 1rem;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
    }
    .month-field {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .month-label {
        font-size: 0.75rem;
        color: var(--color-text-secondary);
        margin-bottom: 0.2rem;
    }
    .month-input {
        padding: 0.25rem;
        text-align: center;
        font-size: 0.9rem;
    }
</style>
