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

/**
 * POST a JSON payload and trigger a browser download of the binary response.
 * Used by the thermal-lab Excel/PDF exports, which need a POST body (the
 * full comparison config) rather than a simple GET URL.
 */
async function downloadPost(endpoint, payload, filename) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || 'Export failed');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

export const api = {
    // ── Hardware ──────────────────────────────────────────────────────────
    listInverters: () => request('/inverters'),
    createInverter: (data) => request('/inverters', { method: 'POST', body: JSON.stringify(data) }),
    updateInverter: (id, data) => request(`/inverters/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    deleteInverter: (id) => request(`/inverters/${id}`, { method: 'DELETE' }),

    listPanels: () => request('/panels'),
    createPanel: (data) => request('/panels', { method: 'POST', body: JSON.stringify(data) }),
    updatePanel: (id, data) => request(`/panels/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    deletePanel: (id) => request(`/panels/${id}`, { method: 'DELETE' }),

    listBatteries: () => request('/batteries'),
    createBattery: (data) => request('/batteries', { method: 'POST', body: JSON.stringify(data) }),
    updateBattery: (id, data) => request(`/batteries/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    deleteBattery: (id) => request(`/batteries/${id}`, { method: 'DELETE' }),

    // ── Profiles ──────────────────────────────────────────────────────────

    // Solar profiles back the "Luogo di installazione" step in the wizard
    // and the "Posizioni" tab in the Database UI.
    async listSolarProfiles() { return request('/profiles/solar'); },
    updateSolarProfile: (id, data) =>
        request(`/profiles/solar/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    deleteSolarProfile: (id) => request(`/profiles/solar/${id}`, { method: 'DELETE' }),

    // Phase 14 — external geolocation, climate normals, and one-shot import
    // of a solar profile from a (lat, lon, tilt, azimuth) combo via
    // PVGIS + Open-Meteo on the backend.
    async geocode(query, { limit = 5, accept_language = 'it,en' } = {}) {
        return request('/external/geocode', {
            method: 'POST',
            body: JSON.stringify({ query, limit, accept_language }),
        });
    },
    async getClimateNormals(lat, lon, { lookback_years = 10 } = {}) {
        const q = new URLSearchParams({ lat, lon, lookback_years }).toString();
        return request(`/external/climate-normals?${q}`);
    },
    async createSolarProfileFromLocation(payload) {
        return request('/profiles/solar/from_location', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },

    // Phase 15 — stochastic thermal profile (ClimateProfileModel).
    // The from_location endpoint fits a ThermalModel from 10 years of
    // Open-Meteo daily archive (seasonal harmonic + AR(1) residuals +
    // GPD tails for heatwaves/coldsnaps).
    async listClimateProfiles() {
        return request('/profiles/climate');
    },
    async createClimateProfileFromLocation(payload) {
        return request('/profiles/climate/from_location', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
    async previewClimateProfileById(id, { n_paths = 50, n_years = 1, seed = 42 } = {}) {
        const q = new URLSearchParams({ n_paths, n_years, seed }).toString();
        return request(`/profiles/climate/${id}/preview?${q}`);
    },
    updateClimateProfile: (id, data) =>
        request(`/profiles/climate/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    deleteClimateProfile: (id) => request(`/profiles/climate/${id}`, { method: 'DELETE' }),

    async listLoadProfiles() { return request('/profiles/load'); },
    async createLoadProfile(data) {
        return request('/profiles/load', { method: 'POST', body: JSON.stringify(data) });
    },
    updateLoadProfile: (id, data) =>
        request(`/profiles/load/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    deleteLoadProfile: (id) => request(`/profiles/load/${id}`, { method: 'DELETE' }),

    async listPriceProfiles() { return request('/profiles/price'); },
    async createPriceProfile(data) {
        return request('/profiles/price', { method: 'POST', body: JSON.stringify(data) });
    },
    updatePriceProfile: (id, data) =>
        request(`/profiles/price/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
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

    // ── Thermal lab (Phase 19) ────────────────────────────────────────────
    // Compare insulation levels (house variants) against a saved climate
    // profile, and preview the hourly indoor-temperature trajectory of a
    // single configuration. Both run the HVAC/RC engine server-side.
    async compareThermalLab(payload) {
        return request('/thermal-lab/compare', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
    async previewThermalTimeseries(payload) {
        return request('/thermal-lab/timeseries', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
    async exportThermalLabXlsx(payload) {
        return downloadPost('/thermal-lab/compare/export.xlsx', payload, 'laboratorio_termico.xlsx');
    },
    async exportThermalLabPdf(payload) {
        return downloadPost('/thermal-lab/compare/export.pdf', payload, 'laboratorio_termico.pdf');
    },

    // ── Electricity market lab ────────────────────────────────────────────
    // Design a generation mix + capacity trends + fuel/CO2 scenarios and read
    // back the wholesale price views; persist a designed market as a reusable
    // market profile referenced by scenarios.
    async runMarketLab(payload) {
        return request('/market/run', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
    async exportMarketXlsx(payload) {
        return downloadPost('/market/run/export.xlsx', payload, 'mercato_elettrico.xlsx');
    },
    async exportMarketPdf(payload) {
        return downloadPost('/market/run/export.pdf', payload, 'mercato_elettrico.pdf');
    },
    async listMarketProfiles() {
        return request('/market/profiles');
    },
    async saveMarketProfile(payload) {
        return request('/market/profiles', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
    async deleteMarketProfile(id) {
        return request(`/market/profiles/${id}`, { method: 'DELETE' });
    },

    // ── Configurations ────────────────────────────────────────────────────
    async listConfigurations(type) {
        const url = type ? `/configurations?type=${type}` : '/configurations';
        return request(url);
    },
    async createConfiguration(data) {
        return request('/configurations', { method: 'POST', body: JSON.stringify(data) });
    },
    getConfiguration: (id) => request(`/configurations/${id}`),
    updateConfiguration: (id, data) =>
        request(`/configurations/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
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

    /**
     * List run results with optional filters and pagination.
     * @param {object} [opts]
     * @param {number} [opts.limit]  Max rows (default backend = 50).
     * @param {number} [opts.offset]  Pagination offset (default 0).
     * @param {string} [opts.scenarioName]  Substring filter on summary.scenario.
     * @param {string} [opts.location]  Exact match on summary.location_name.
     * @param {string} [opts.dateFrom]  ISO timestamp lower bound (inclusive).
     * @param {string} [opts.dateTo]    ISO timestamp upper bound (inclusive).
     * @param {boolean} [opts.includeArchived]  Include archived runs.
     */
    async listRuns(opts = {}) {
        const params = new URLSearchParams();
        if (opts.limit != null) params.set('limit', String(opts.limit));
        if (opts.offset != null) params.set('offset', String(opts.offset));
        if (opts.scenarioName) params.set('scenario_name', opts.scenarioName);
        if (opts.location) params.set('location', opts.location);
        if (opts.dateFrom) params.set('date_from', opts.dateFrom);
        if (opts.dateTo) params.set('date_to', opts.dateTo);
        if (opts.includeArchived) params.set('include_archived', 'true');
        const qs = params.toString();
        return request('/runs' + (qs ? `?${qs}` : ''));
    },
    async listRunLocations() {
        return request('/runs/locations');
    },
    async archiveRun(runId) {
        return request(`/runs/${runId}/archive`, { method: 'PATCH' });
    },
    async unarchiveRun(runId) {
        return request(`/runs/${runId}/unarchive`, { method: 'PATCH' });
    },
    async deleteRun(runId) {
        return request(`/runs/${runId}`, { method: 'DELETE' });
    },

    // Phase 11 — download helpers for the per-run Excel and PDF exports.
    // We expose URLs (rather than fetching the binary) so the Dashboard
    // can use a plain <a href download> link and the browser handles the
    // save-as dialog natively.
    runCashflowXlsxUrl(runId) {
        return `${API_BASE}/runs/${runId}/export/cashflow.xlsx`;
    },
    runReportPdfUrl(runId) {
        return `${API_BASE}/runs/${runId}/export/report.pdf`;
    },

    // Phase 12 — background job queue. The submit endpoints return a
    // job_id immediately; the client polls the status endpoint to drive
    // the floating progress bar.
    async submitAnalysisJob(payload) {
        return request('/jobs/analysis', { method: 'POST', body: JSON.stringify(payload) });
    },
    async submitOptimizationJob(payload) {
        return request('/jobs/optimization', { method: 'POST', body: JSON.stringify(payload) });
    },
    async getJob(jobId) {
        return request(`/jobs/${jobId}`);
    },

    // Phase 11+ — inline load profile templates and Excel parsing.
    loadProfileTemplateUrl(kind) {
        return `${API_BASE}/load-profiles/template/${kind}.xlsx`;
    },
    async parseLoadProfileXlsx(kind, file) {
        const fd = new FormData();
        fd.append("file", file);
        const response = await fetch(
            `${API_BASE}/load-profiles/parse-xlsx/${kind}`,
            { method: "POST", body: fd },
        );
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || "Upload failed");
        }
        return response.json();
    },
};
