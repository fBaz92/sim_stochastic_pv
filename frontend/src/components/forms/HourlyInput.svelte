<script>
    import { createEventDispatcher } from "svelte";

    export let values = Array(24).fill(0);
    export let label = "24h Profile (Watts)";

    const dispatch = createEventDispatcher();

    function handleInput(event, index) {
        const updated = [...values];
        const numericValue = event.currentTarget.valueAsNumber;
        updated[index] = Number.isNaN(numericValue) ? 0 : numericValue;
        values = updated;
        dispatch("change", updated);
    }
</script>

<div class="hourly-input-group">
    <span class="label">{label}</span>
    <div class="grid-24">
        {#each values as val, i}
            <div class="field">
                <span class="sub-label">{i}h</span>
                <input
                    type="number"
                    step="0.1"
                    class="input small-input"
                    value={val}
                    on:input={(event) => handleInput(event, i)}
                />
            </div>
        {/each}
    </div>
</div>

<style>
    .hourly-input-group {
        margin-bottom: 1rem;
    }
    .grid-24 {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(2.5rem, 1fr));
        gap: 0.5rem;
    }
    .field {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .sub-label {
        font-size: 0.7rem;
        color: var(--color-text-secondary);
    }
    .small-input {
        padding: 0.2rem;
        text-align: center;
        font-size: 0.85rem;
    }
</style>
