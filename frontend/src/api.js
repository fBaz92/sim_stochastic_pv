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
    return response.json();
}

export const api = {
    // Hardware
    listInverters: () => request('/inverters'),
    createInverter: (data) => request('/inverters', { method: 'POST', body: JSON.stringify(data) }),

    listPanels: () => request('/panels'),
    createPanel: (data) => request('/panels', { method: 'POST', body: JSON.stringify(data) }),

    listBatteries: () => request('/batteries'),
    async createBattery(data) {
        return request('/batteries', { method: 'POST', body: JSON.stringify(data) });
    },

    // Profiles
    async listLoadProfiles() { return request('/profiles/load'); },
    async createLoadProfile(data) {
        return request('/profiles/load', { method: 'POST', body: JSON.stringify(data) });
    },
    async listPriceProfiles() { return request('/profiles/price'); },
    async createPriceProfile(data) {
        return request('/profiles/price', { method: 'POST', body: JSON.stringify(data) });
    },

    // Configurations
    async listConfigurations(type) {
        const url = type ? `/configurations?type=${type}` : '/configurations';
        return request(url);
    },
    async createConfiguration(data) {
        return request('/configurations', { method: 'POST', body: JSON.stringify(data) });
    },

    // Scenarios (Execution History)
    async listScenarios() {
        return request('/scenarios');
    },

    // Simulation
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
        return request(`/campaigns/${campaignId}/run${query}`, { method: 'POST' });
    },

    async listRuns() {
        return request('/runs');
    },
};

