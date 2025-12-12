import Dashboard from './pages/Dashboard.svelte';
import ScenarioBuilder from './pages/ScenarioBuilder.svelte';
import CampaignBuilder from './pages/CampaignBuilder.svelte';
import Database from './pages/Database.svelte';

export default {
    '/': Dashboard,
    '/scenario': ScenarioBuilder,
    '/campaign': CampaignBuilder,
    '/database': Database
}
