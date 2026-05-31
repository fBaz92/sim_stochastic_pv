import Dashboard from './pages/Dashboard.svelte';
import ScenarioBuilder from './pages/ScenarioBuilder.svelte';
import CampaignBuilder from './pages/CampaignBuilder.svelte';
import Database from './pages/Database.svelte';
import ThermalLab from './pages/ThermalLab.svelte';
import ElectricityMarket from './pages/ElectricityMarket.svelte';

// Phase 11 — "Campagna" is now branded "Design" in the UI. The route
// path moved from /campaign to /design, but the imported component
// keeps its historical filename so the API client and DB type stay
// stable (see CLAUDE.md §Glossario rapido).
export default {
    '/': Dashboard,
    '/scenario': ScenarioBuilder,
    '/design': CampaignBuilder,
    // Legacy alias so existing bookmarks / saved URLs keep working.
    '/campaign': CampaignBuilder,
    // Phase 19 — thermal laboratory: compare insulation levels + HVAC sizing.
    '/thermal-lab': ThermalLab,
    // Electricity-market lab: design the wholesale market + ritiro dedicato.
    '/market': ElectricityMarket,
    '/database': Database
}
