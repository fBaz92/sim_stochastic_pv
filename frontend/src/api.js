const API_BASE = 'http://localhost:8000/api';

async function request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers,
    };

    const response = await fetch(url, { ...options, headers });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'API Request Failed');
    }
    // 204 No Content: return null instead of calling .json() (which would throw)
    if (response.status === 204) return null;
    return response.json();
}

export const api = {
    // ── Hardware ──────────────────────────────────────────────────────────
    listInverters: () => request('/inverters'),
    createInverter: (data) => request('/inverters', { method: 'POST', body: JSON.stringify(data) }),
    deleteInverter: (id) => request(`/inverters/${id}`, { method: 'DELETE' }),

    listPanels: () => request('/panels'),
    createPanel: (data) => request('/panels', { method: 'POST', body: JSON.stringify(data) }),
    deletePanel: (id) => request(`/panels/${id}`, { method: 'DELETE' }),

    listBatteries: () => request('/batteries'),
    createBattery: (data) => request('/batteries', { method: 'POST', body: JSON.stringify(data) }),
    deleteBattery: (id) => request(`/batteries/${id}`, { method: 'DELETE' }),

    // ── Profiles ──────────────────────────────────────────────────────────

    // Phase 6: solar profiles feed the Wizard "Luogo di installazione" step.
    async listSolarProfiles() { return request('/profiles/solar'); },

    async listLoadProfiles() { return request('/profiles/load'); },
    async createLoadProfile(data) {
        return request('/profiles/load', { method: 'POST', body: JSON.stringify(data) });
    },
    deleteLoadProfile: (id) => request(`/profiles/load/${id}`, { method: 'DELETE' }),

    async listPriceProfiles() { return request('/profiles/price'); },
    async createPriceProfile(data) {
        return request('/profiles/price', { method: 'POST', body: JSON.stringify(data) });
    },
    deletePriceProfile: (id) => request(`/profiles/price/${id}`, { method: 'DELETE' }),

    // Phase 10: Monte Carlo preview of a price profile (fan chart in DB UI)
    async previewPriceProfileById(id, { n_paths = 200, n_years = 20, seed = 42 } = {}) {
        const q = new URLSearchParams({ n_paths, n_years, seed }).toString();
        return request(`/profiles/price/${id}/preview?${q}`);
    },
    async previewPriceParameters(payload, { n_paths = 200, n_years = 20, seed = 42 } = {}) {
        const q = new URLSearchParams({ n_paths, n_years, seed }).toString();
        return request(`/profiles/price/preview?${q}`, {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },

    // ── Configurations ────────────────────────────────────────────────────
    async listConfigurations(type) {
        const url = type ? `/configurations?type=${type}` : '/configurations';
        return request(url);
    },
    async createConfiguration(data) {
        return request('/configurations', { method: 'POST', body: JSON.stringify(data) });
    },
    deleteConfiguration: (id) => request(`/configurations/${id}`, { method: 'DELETE' }),

    // ── Scenarios (Execution History) ─────────────────────────────────────
    async listScenarios() {
        return request('/scenarios');
    },

    // ── Simulation ────────────────────────────────────────────────────────
    async triggerAnalysis(payload) {
        return request('/analysis', { method: 'POST', body: JSON.stringify(payload) });
    },

    async triggerOptimization(payload) {
        return request('/optimization', { method: 'POST', body: JSON.stringify(payload) });
    },

    // Run saved configurations (DB-driven workflow)
    async runSavedScenario(scenarioId, params = {}) {
        const { seed, n_mc } = params;
        const queryParams = new URLSearchParams();
        if (seed !== undefined) queryParams.append('seed', seed);
        if (n_mc !== undefined) queryParams.append('n_mc', n_mc);
        const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
        return request(`/scenarios/${scenarioId}/run${query}`, { method: 'POST' });
    },

    async runSavedCampaign(campaignId, params = {}) {
        const { seed, n_mc } = params;
        const queryParams = new URLSearchParams();
        if (seed !== undefined) queryParams.append('seed', seed);
        if (n_mc !== undefined) queryParams.append('n_mc', n_mc);
        const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
        // Phase 7: backend exposes both `/campaigns/{id}/run` (preferred) and
        // the legacy `/optimizations/{id}/run` as aliases. We hit the new
        // path for terminology consistency with the rest of the UI.
        return request(`/campaigns/${campaignId}/run${query}`, { method: 'POST' });
    },

    async listRuns() {
        return request('/runs');
    },
};
