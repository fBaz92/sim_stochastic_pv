<script>
    import InverterManager from "../components/database/InverterManager.svelte";
    import PanelManager from "../components/database/PanelManager.svelte";
    import BatteryManager from "../components/database/BatteryManager.svelte";
    import LoadProfileManager from "../components/database/LoadProfileManager.svelte";
    import PriceProfileManager from "../components/database/PriceProfileManager.svelte";

    let activeTab = "inverters";
</script>

<div class="container">
    <h1 class="page-title">Database Management</h1>

    <div class="tabs">
        <button
            class="tab-btn"
            class:active={activeTab === "inverters"}
            on:click={() => (activeTab = "inverters")}
        >
            Inverters
        </button>
        <button
            class="tab-btn"
            class:active={activeTab === "panels"}
            on:click={() => (activeTab = "panels")}
        >
            Panels
        </button>
        <button
            class="tab-btn"
            class:active={activeTab === "batteries"}
            on:click={() => (activeTab = "batteries")}
        >
            Batteries
        </button>
        <button
            class="tab-btn"
            class:active={activeTab === "loads"}
            on:click={() => (activeTab = "loads")}>Load Profiles</button
        >
        <button
            class="tab-btn"
            class:active={activeTab === "prices"}
            on:click={() => (activeTab = "prices")}>Price Profiles</button
        >
    </div>

    <div class="tab-content">
        {#if activeTab === "inverters"}
            <InverterManager />
        {:else if activeTab === "panels"}
            <PanelManager />
        {:else if activeTab === "batteries"}
            <BatteryManager />
        {:else if activeTab === "loads"}
            <LoadProfileManager />
        {:else if activeTab === "prices"}
            <PriceProfileManager />
        {/if}
    </div>
</div>
