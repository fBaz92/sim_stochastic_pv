<script>
    /**
     * MarketConfigEditor — shared editor for the electricity-market build
     * configuration (generation mix, capacity trends, fuel/CO₂ scenarios,
     * simulation parameters).
     *
     * This is the form half of the market designer, factored out so that both
     * the "Mercato elettrico" lab page and the Database "Profili di mercato"
     * manager drive the same inputs. It owns the config state internally and
     * exposes two imperative methods:
     *
     *   - ``getConfig()`` → the payload object accepted by ``POST /api/market/run``
     *     and ``POST /api/market/profiles`` (under the ``config`` key).
     *   - ``setConfig(c)`` → load a saved ``config`` object back into the form.
     *
     * PMG / retail / save UI is intentionally NOT here: those differ between the
     * lab (save-as-profile box) and the manager (CRUD), so each owns its own.
     *
     * Events:
     *   - ``displaychange`` — fired when the "Anno mostrato" dropdown changes, so
     *     a parent that re-runs per displayed year can react.
     */
    import { createEventDispatcher } from "svelte";
    import { TECHS } from "../lib/marketCharts.js";

    const dispatch = createEventDispatcher();

    // ── Config state (defaults = Italian base mix) ────────────────────────
    let techState = {
        gas: { cap: 45, growth: 0, stepYear: "", stepCap: "" },
        coal: { cap: 0, growth: 0, stepYear: "", stepCap: "" },
        nuclear: { cap: 0, growth: 0, stepYear: "", stepCap: "" },
        wind: { cap: 13, growth: 4, stepYear: "", stepCap: "" },
        solar: { cap: 30, growth: 6, stepYear: "", stepCap: "" },
        hydro_mustrun: { cap: 8, growth: 0, stepYear: "", stepCap: "" },
    };
    let gasScenario = "base";
    let co2Scenario = "base";
    let coalScenario = "";
    let gasDrift = 0.0;
    let co2Drift = 0.0;
    let nYears = 20;
    let nTrajectories = 8;
    let nRuns = 6;
    let displayYear = 0;
    let seed = 42;

    /** Build the config payload consumed by the market API. */
    export function getConfig() {
        const capacities_gw = {};
        const capacity_trends = {};
        for (const t of TECHS) {
            const s = techState[t.key];
            capacities_gw[t.key] = Number(s.cap);
            const trend = { annual_growth_pct: Number(s.growth) };
            if (s.stepYear !== "" && s.stepCap !== "") {
                trend.step_year = Number(s.stepYear);
                trend.step_capacity_gw = Number(s.stepCap);
            }
            capacity_trends[t.key] = trend;
        }
        return {
            capacities_gw,
            capacity_trends,
            gas_scenario: gasScenario,
            co2_scenario: co2Scenario || null,
            coal_scenario: coalScenario || null,
            gas_mu_drift_annual: Number(gasDrift),
            co2_mu_drift_annual: Number(co2Drift),
            n_years: Number(nYears),
            n_trajectories: Number(nTrajectories),
            n_runs: Number(nRuns),
            seed: Number(seed),
            display_year: Number(displayYear),
        };
    }

    /**
     * Load a saved ``config`` object into the editor.
     *
     * Reads only the keys it recognises, so a lighter seeded profile loads too
     * (the rest keeps the current defaults).
     */
    export function setConfig(c) {
        c = c || {};
        if (c.gas_scenario) gasScenario = c.gas_scenario;
        if ("co2_scenario" in c) co2Scenario = c.co2_scenario ?? "";
        if ("coal_scenario" in c) coalScenario = c.coal_scenario ?? "";
        if (c.gas_mu_drift_annual != null) gasDrift = c.gas_mu_drift_annual;
        if (c.co2_mu_drift_annual != null) co2Drift = c.co2_mu_drift_annual;
        if (c.n_years != null) nYears = c.n_years;
        if (c.n_trajectories != null) nTrajectories = c.n_trajectories;
        if (c.n_runs != null) nRuns = c.n_runs;
        if (c.seed != null) seed = c.seed;
        if (c.display_year != null) displayYear = c.display_year;
        for (const t of TECHS) {
            if (c.capacities_gw && c.capacities_gw[t.key] != null) {
                techState[t.key].cap = c.capacities_gw[t.key];
            }
            const tr = c.capacity_trends && c.capacity_trends[t.key];
            if (tr) {
                techState[t.key].growth = tr.annual_growth_pct ?? 0;
                techState[t.key].stepYear = tr.step_year ?? "";
                techState[t.key].stepCap = tr.step_capacity_gw ?? "";
            }
        }
        techState = { ...techState }; // trigger reactivity after nested edits
    }

    /** Current displayed-year (read-only accessor for parents that need it). */
    export function getDisplayYear() {
        return Number(displayYear);
    }

    // Year options for the "Anno mostrato" dropdown (0 .. n_years-1).
    $: yearOptions = Array.from({ length: Math.max(1, Number(nYears) || 1) }, (_, i) => i);
    // Keep the displayed-year dropdown valid when the horizon shrinks.
    $: if (displayYear > (Number(nYears) || 1) - 1) displayYear = Math.max(0, (Number(nYears) || 1) - 1);
</script>

<div class="section-title">Mix e trend di capacità</div>
<div class="mix-table-wrap">
    <table class="mix-table">
        <thead>
            <tr>
                <th>Tecnologia</th>
                <th title="Capacità installata all'anno 0 (GW)">GW</th>
                <th title="Crescita (positivo) o dismissione (negativo) annua composta, in % all'anno">%/anno</th>
                <th title="Anno (0-based) a partire dal quale si impone una nuova capacità (es. nuovo nucleare)">Anno step</th>
                <th title="Capacità imposta (GW) a partire dall'anno step">GW step</th>
            </tr>
        </thead>
        <tbody>
            {#each TECHS as t}
                <tr>
                    <td><span class="dot" style="background:{t.color}"></span>{t.label}</td>
                    <td><input class="input mini" type="number" min="0" step="0.5" bind:value={techState[t.key].cap} /></td>
                    <td><input class="input mini" type="number" step="0.5" bind:value={techState[t.key].growth} /></td>
                    <td><input class="input mini" type="number" min="0" bind:value={techState[t.key].stepYear} placeholder="—" /></td>
                    <td><input class="input mini" type="number" min="0" step="0.5" bind:value={techState[t.key].stepCap} placeholder="—" /></td>
                </tr>
            {/each}
        </tbody>
    </table>
</div>
<p class="hint">
    <strong>%/anno</strong>: crescita (valore positivo) o dismissione
    (negativo) annua composta della capacità — es. <em>Solare +6</em>,
    <em>Carbone −10</em>. <strong>Anno step</strong> + <strong>GW step</strong>:
    impongono una capacità a partire da un dato anno — es. nucleare
    <em>anno 9 → 4 GW</em> (la crescita riparte da lì). Lascia vuoti i
    campi step per non usarli.
</p>

<div class="divider"></div>
<div class="section-title">Scenari combustibili</div>
<div class="grid-mini">
    <div class="form-group">
        <label class="label" for="gas">Scenario gas</label>
        <select id="gas" class="select" bind:value={gasScenario}>
            <option value="base">Base</option>
            <option value="tension">Tensione</option>
            <option value="crisis">Crisi</option>
        </select>
    </div>
    <div class="form-group">
        <label class="label" for="co2">Scenario CO₂</label>
        <select id="co2" class="select" bind:value={co2Scenario}>
            <option value="">Default motore</option>
            <option value="base">Base</option>
            <option value="low">Basso</option>
            <option value="high">Alto</option>
        </select>
    </div>
    <div class="form-group">
        <label class="label" for="coal">Scenario carbone</label>
        <select id="coal" class="select" bind:value={coalScenario}>
            <option value="">Default motore</option>
            <option value="base">Base</option>
            <option value="tension">Tensione</option>
            <option value="crisis">Crisi</option>
        </select>
    </div>
    <div class="form-group">
        <label class="label" for="gasDrift">Drift gas %/anno</label>
        <input id="gasDrift" class="input" type="number" step="0.01" bind:value={gasDrift} />
    </div>
</div>

<div class="divider"></div>
<div class="section-title">Simulazione</div>
<div class="grid-mini">
    <div class="form-group">
        <label class="label" for="nYears">Anni orizzonte</label>
        <input id="nYears" class="input" type="number" min="1" max="30" bind:value={nYears} />
    </div>
    <div class="form-group">
        <label class="label" for="displayYear">Anno mostrato</label>
        <select id="displayYear" class="select" bind:value={displayYear} on:change={() => dispatch("displaychange", Number(displayYear))}>
            {#each yearOptions as y}
                <option value={y}>Anno {y}</option>
            {/each}
        </select>
    </div>
    <div class="form-group">
        <label class="label" for="nTraj" title="Numero di traiettorie di mercato indipendenti per la banda di incertezza del prezzo (fan chart / heatmap / durata)">Traiettorie ⓘ</label>
        <input id="nTraj" class="input" type="number" min="1" max="100" bind:value={nTrajectories} />
    </div>
    <div class="form-group">
        <label class="label" for="nRuns" title="Simulazioni Monte Carlo del mercato usate per la heatmap «chi fissa il prezzo»: quante volte si ridispatcha l'anno per stimare quale tecnologia è marginale. Più simulazioni = stima più stabile, ma più lento.">Simulazioni «chi fissa il prezzo» ⓘ</label>
        <input id="nRuns" class="input" type="number" min="1" max="100" bind:value={nRuns} />
    </div>
</div>
<p class="hint">
    «Anno mostrato» ricalcola heatmap, curva di durata e «chi fissa
    il prezzo» per l'anno scelto. «Traiettorie» allarga/restringe la
    banda d'incertezza del prezzo; «Simulazioni» raffina la stima di
    chi fissa il prezzo.
</p>

<style>
    .mix-table-wrap {
        overflow-x: auto;
    }
    .mix-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }
    .mix-table th {
        text-align: left;
        font-weight: 600;
        color: var(--color-text-secondary, #6b7280);
        padding: 2px 3px;
        font-size: 0.7rem;
        cursor: help;
    }
    .mix-table td {
        padding: 2px 3px;
        white-space: nowrap;
    }
    .input.mini {
        width: 52px;
        padding: 3px 4px;
    }
    .hint {
        font-size: 0.75rem;
        color: var(--color-text-secondary, #6b7280);
        line-height: 1.4;
        margin: 0.5rem 0 0 0;
    }
    .dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 5px;
    }
</style>
