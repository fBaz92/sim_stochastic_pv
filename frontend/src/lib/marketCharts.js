/**
 * marketCharts.js — shared display config + Chart.js builders for the
 * electricity-market views.
 *
 * Used by the "Mercato elettrico" lab page (ElectricityMarket.svelte),
 * the shared MarketConfigEditor component, and the Database manager
 * (MarketProfileManager.svelte). All builders take the response payload
 * of `POST /api/market/run` and return a Chart.js config object ready
 * for the ResultsChart component.
 */

// ── Technology display config (label + colour) ────────────────────────
export const TECHS = [
    { key: "gas", label: "Gas", color: "#ef4444" },
    { key: "coal", label: "Carbone", color: "#6b7280" },
    { key: "nuclear", label: "Nucleare", color: "#8b5cf6" },
    { key: "wind", label: "Eolico", color: "#10b981" },
    { key: "solar", label: "Solare", color: "#f59e0b" },
    { key: "hydro_mustrun", label: "Idro (must-run)", color: "#3b82f6" },
];

const TECH_COLOR = {
    gas: "#ef4444", coal: "#6b7280", nuclear: "#8b5cf6",
    wind: "#10b981", solar: "#f59e0b", hydro_mustrun: "#3b82f6",
    import: "#ec4899",
};
const FALLBACK = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#6b7280", "#ec4899"];

/** Stable colour for a technology key, with an index-based fallback. */
export const colorFor = (tech, i) => TECH_COLOR[tech] ?? FALLBACK[i % FALLBACK.length];

export const MONTHS = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"];
export const HOURS = Array.from({ length: 24 }, (_, h) => String(h));

/** Annual wholesale-price fan chart (mean + p05–p95 band) over the horizon. */
export function buildFanConfig(res) {
    return {
        type: "line",
        data: {
            labels: res.years,
            datasets: [
                {
                    label: "p05",
                    data: res.annual_price_p05_eur_per_kwh,
                    borderColor: "transparent",
                    pointRadius: 0,
                    fill: false,
                },
                {
                    label: "p05–p95",
                    data: res.annual_price_p95_eur_per_kwh,
                    borderColor: "transparent",
                    backgroundColor: "rgba(59,130,246,0.15)",
                    pointRadius: 0,
                    fill: "-1",
                },
                {
                    label: "media",
                    data: res.annual_price_mean_eur_per_kwh,
                    borderColor: "#1d4ed8",
                    backgroundColor: "#1d4ed8",
                    pointRadius: 2,
                    fill: false,
                },
            ],
        },
        options: {
            plugins: {
                legend: {
                    display: true,
                    // Hide the invisible lower-bound dataset from the legend.
                    labels: { filter: (item) => item.text !== "p05" },
                },
            },
            scales: {
                x: { title: { display: true, text: "Anno" } },
                y: { title: { display: true, text: "€/kWh" } },
            },
        },
    };
}

/** Fuel-price trajectories: gas (€/MWh, left axis) + CO₂ (€/t, right axis). */
export function buildFuelConfig(res) {
    return {
        type: "line",
        data: {
            labels: res.years,
            datasets: [
                {
                    label: "Gas (€/MWh)",
                    data: res.gas_price_by_year_eur_per_mwh,
                    borderColor: "#ef4444",
                    backgroundColor: "#ef4444",
                    pointRadius: 2,
                    fill: false,
                    yAxisID: "y",
                },
                {
                    label: "CO₂ (€/t)",
                    data: res.co2_price_by_year_eur_per_ton,
                    borderColor: "#6b7280",
                    backgroundColor: "#6b7280",
                    pointRadius: 2,
                    fill: false,
                    yAxisID: "y1",
                },
            ],
        },
        options: {
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: "Anno" } },
                y: {
                    position: "left",
                    title: { display: true, text: "Gas €/MWh" },
                },
                y1: {
                    position: "right",
                    title: { display: true, text: "CO₂ €/t" },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    };
}

/** Price-duration curve: price vs. % of hours of the year. */
export function buildDurationConfig(res) {
    return {
        type: "line",
        data: {
            labels: res.duration_curve_x.map((v) => (v * 100).toFixed(0)),
            datasets: [
                {
                    label: "Prezzo €/kWh",
                    data: res.duration_curve_price_eur_per_kwh,
                    borderColor: "#0d9488",
                    backgroundColor: "rgba(13,148,136,0.15)",
                    pointRadius: 0,
                    fill: true,
                },
            ],
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: "% delle ore dell'anno" } },
                y: { title: { display: true, text: "€/kWh" } },
            },
        },
    };
}

/** Stacked installed-capacity evolution (GW) by technology over the years. */
export function buildCapacityConfig(res) {
    return {
        type: "line",
        data: {
            labels: res.years,
            datasets: TECHS.filter((t) => res.capacity_by_year_gw[t.key]).map((t) => ({
                label: t.label,
                data: res.capacity_by_year_gw[t.key],
                borderColor: t.color,
                backgroundColor: t.color + "cc",
                pointRadius: 0,
                fill: true,
            })),
        },
        options: {
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: "Anno" } },
                y: { stacked: true, title: { display: true, text: "GW installati" } },
            },
        },
    };
}
