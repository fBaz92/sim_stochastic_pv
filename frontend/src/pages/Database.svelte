<script>
    import InverterManager from "../components/database/InverterManager.svelte";
    import PanelManager from "../components/database/PanelManager.svelte";
    import BatteryManager from "../components/database/BatteryManager.svelte";
    import LoadProfileManager from "../components/database/LoadProfileManager.svelte";
    import PriceProfileManager from "../components/database/PriceProfileManager.svelte";
    import SavedScenariosManager from "../components/database/SavedScenariosManager.svelte";
    import SavedCampaignsManager from "../components/database/SavedCampaignsManager.svelte";
    import LocationsManager from "../components/database/LocationsManager.svelte";
    import MarketProfileManager from "../components/database/MarketProfileManager.svelte";
    import PlantDesignManager from "../components/database/PlantDesignManager.svelte";

    const TABS = [
        { id: "scenarios", label: "Scenari" },
        { id: "campaigns", label: "Design" },
        { id: "plants", label: "Impianti" },
        { id: "locations", label: "Posizioni" },
        { id: "inverters", label: "Inverter" },
        { id: "panels", label: "Pannelli" },
        { id: "batteries", label: "Batterie" },
        { id: "loads", label: "Profili di carico" },
        { id: "prices", label: "Profili di prezzo" },
        { id: "markets", label: "Profili di mercato" },
    ];

    let activeTab = "scenarios";
</script>

<div class="container">
    <h1 class="page-title">Database</h1>
    <p class="page-subtitle">
        Catalogo riusabile di scenari, design, posizioni geografiche e
        hardware. Ogni voce è modificabile da qui — i cambiamenti si
        propagano automaticamente agli scenari e ai design che la
        referenziano per ID.
    </p>

    <div class="tabs">
        {#each TABS as tab}
            <button
                class="tab-btn"
                class:active={activeTab === tab.id}
                on:click={() => (activeTab = tab.id)}
            >
                {tab.label}
            </button>
        {/each}
    </div>

    <div class="tab-content">
        {#if activeTab === "scenarios"}
            <SavedScenariosManager />
        {:else if activeTab === "campaigns"}
            <SavedCampaignsManager />
        {:else if activeTab === "plants"}
            <PlantDesignManager />
        {:else if activeTab === "locations"}
            <LocationsManager />
        {:else if activeTab === "inverters"}
            <InverterManager />
        {:else if activeTab === "panels"}
            <PanelManager />
        {:else if activeTab === "batteries"}
            <BatteryManager />
        {:else if activeTab === "loads"}
            <LoadProfileManager />
        {:else if activeTab === "prices"}
            <PriceProfileManager />
        {:else if activeTab === "markets"}
            <MarketProfileManager />
        {/if}
    </div>
</div>

<style>
    .page-subtitle {
        color: var(--color-text-secondary);
        max-width: 70ch;
        margin-bottom: 1.5rem;
    }
    .tabs {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        border-bottom: 1px solid var(--color-border, #e2e8f0);
        margin-bottom: 1.5rem;
    }
</style>
