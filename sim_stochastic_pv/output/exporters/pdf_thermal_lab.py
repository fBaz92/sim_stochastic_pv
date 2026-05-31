"""
PDF report exporter for the Phase-19 thermal-lab comparison.

Renders a single-page-ish summary of a thermal-lab comparison via WeasyPrint
(HTML+CSS → PDF) with matplotlib figures embedded as base64 PNGs, mirroring
the Phase-11 ``pdf_report`` exporter.

Sections:
- a KPI comparison table (one row per house variant);
- a daily HVAC-energy chart per variant overlaid with the outdoor
  temperature, with the worst heating/cooling days marked;
- an annual-cost bar chart per variant;
- (dynamic mode only) the representative indoor-temperature band per variant.

The exporter consumes a plain mapping (the ``/thermal-lab/compare`` response
enriched with run-meta keys), never the simulation objects, so it is
testable without re-running the Monte Carlo.
"""

from __future__ import annotations

import base64
import io
from typing import Any, BinaryIO, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
from jinja2 import Template
from weasyprint import HTML


# Distinct colours per variant (kept in sync with the frontend palette).
_VARIANT_COLORS = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]
_MONTHS_SHORT = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                 "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
_MONTH_START_DOY = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _fig_to_base64(fig) -> str:
    """Serialise a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _color(idx: int) -> str:
    return _VARIANT_COLORS[idx % len(_VARIANT_COLORS)]


def _plot_daily(report: Mapping[str, Any]) -> str | None:
    """Daily HVAC energy per variant + outdoor temperature on a twin axis."""
    variants = report.get("variants") or []
    days = report.get("days") or []
    outdoor = report.get("daily_outdoor_mean_c") or []
    if not variants or not days:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for idx, v in enumerate(variants):
        ax.plot(days, v.get("daily_hvac_kwh") or [], color=_color(idx),
                lw=1.4, label=v.get("label"))
        wh = v.get("worst_heating_day_index")
        if wh is not None:
            ax.plot(wh, (v.get("daily_hvac_kwh") or [0])[wh], "o",
                    color=_color(idx), markeredgecolor="white", markersize=6)
    ax2 = ax.twinx()
    ax2.plot(days, outdoor, color="#6c757d", lw=1.2, ls="--", label="T esterna")
    ax2.set_ylabel("T esterna (°C)")
    ax.set_ylabel("Energia HVAC (kWh/giorno)")
    ax.set_xlabel("Mese")
    ax.set_xticks(_MONTH_START_DOY)
    ax.set_xticklabels(_MONTHS_SHORT, fontsize=7)
    ax.set_title("Consumi giornalieri (anno tipico)")
    ax.legend(loc="upper center", fontsize=7, ncol=min(3, len(variants)))
    ax.grid(alpha=0.2)
    return _fig_to_base64(fig)


def _plot_cost(report: Mapping[str, Any]) -> str | None:
    """Annual HVAC cost per variant as a bar chart."""
    variants = report.get("variants") or []
    if not variants:
        return None
    labels = [v.get("label") for v in variants]
    costs = [v.get("annual_cost_eur_mean", 0.0) for v in variants]
    colors = [_color(i) for i in range(len(variants))]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(labels, costs, color=colors)
    ax.set_ylabel("€/anno")
    ax.set_title("Costo annuo HVAC per configurazione")
    ax.grid(alpha=0.2, axis="y")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(7)
    return _fig_to_base64(fig)


def _plot_indoor(report: Mapping[str, Any]) -> str | None:
    """Representative indoor-temperature band per variant (dynamic only)."""
    variants = report.get("variants") or []
    days = report.get("days") or []
    if not days or not any(v.get("daily_indoor_min_c") for v in variants):
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for idx, v in enumerate(variants):
        vmin = v.get("daily_indoor_min_c")
        vmax = v.get("daily_indoor_max_c")
        if not vmin:
            continue
        ax.fill_between(days, vmin, vmax, color=_color(idx), alpha=0.18)
        ax.plot(days, vmin, color=_color(idx), lw=1.0, label=v.get("label"))
    ax.set_ylabel("T interna (°C)")
    ax.set_xlabel("Mese")
    ax.set_xticks(_MONTH_START_DOY)
    ax.set_xticklabels(_MONTHS_SHORT, fontsize=7)
    ax.set_title("Temperatura interna giornaliera (min–max, path rappresentativo)")
    ax.legend(loc="lower center", fontsize=7, ncol=min(3, len(variants)))
    ax.grid(alpha=0.2)
    return _fig_to_base64(fig)


def _variant_rows(report: Mapping[str, Any]) -> list[dict]:
    """Pre-format the KPI table rows so the Jinja template stays trivial."""
    dynamic = bool(report.get("dynamic"))
    rows = []
    for v in report.get("variants") or []:
        rows.append({
            "label": v.get("label"),
            "ua": f"{v.get('ua_kw_per_c', 0.0):.3f}",
            "kwh": f"{v.get('hvac_kwh_annual_mean', 0.0):.0f}",
            "kwh_band": f"{v.get('hvac_kwh_annual_p05', 0.0):.0f}–{v.get('hvac_kwh_annual_p95', 0.0):.0f}",
            "heat_cool": f"{v.get('heating_kwh_annual_mean', 0.0):.0f} / {v.get('cooling_kwh_annual_mean', 0.0):.0f}",
            "cost": f"{v.get('annual_cost_eur_mean', 0.0):.0f}",
            "breach": f"{v.get('comfort_breach_hours_per_year_mean', 0.0):.0f}",
            "peak": f"{v.get('p_elec_hvac_peak_kw_mean', 0.0):.2f}",
            "t_in": (f"{v['t_in_min_c']:.1f} / {v['t_in_max_c']:.1f}" if dynamic else "—"),
        })
    return rows


_HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Laboratorio termico — confronto</title>
<style>
@page { size: A4; margin: 1.5cm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2937; }
h1 { font-size: 19pt; margin: 0 0 0.2em; }
h2 { font-size: 13pt; margin: 1.2em 0 0.4em; color: #0d6efd; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.2em; }
.subtitle { color: #6c757d; font-size: 9.5pt; margin-bottom: 1em; }
img.chart { width: 100%; height: auto; page-break-inside: avoid; margin-top: 0.3em; }
table.kpi { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
table.kpi th, table.kpi td { border: 1px solid #e2e8f0; padding: 0.3em 0.5em; text-align: right; }
table.kpi th { background: #f8f9fa; }
table.kpi td:first-child, table.kpi th:first-child { text-align: left; }
.section { page-break-inside: avoid; }
.muted { color: #6c757d; font-size: 8.5pt; }
</style>
</head>
<body>
<h1>Laboratorio termico — confronto isolamenti</h1>
<p class="subtitle">
  {% if climate_name %}Clima: {{ climate_name }} · {% endif %}
  {{ n_paths }} path × {{ n_years }} anno/i · {{ price_label }} ·
  Modello {{ "dinamico RC" if dynamic else "steady-state" }}
</p>

<h2>Confronto KPI</h2>
<table class="kpi">
<thead><tr>
<th>Configurazione</th><th>UA (kW/°C)</th><th>kWh/anno</th><th>Banda (p05–p95)</th>
<th>Risc./Raffr. (kWh)</th><th>Costo (€/anno)</th><th>Breach (h/anno)</th>
<th>Picco (kW)</th><th>T int. min/max</th>
</tr></thead>
<tbody>
{% for r in rows %}
<tr>
<td>{{ r.label }}</td><td>{{ r.ua }}</td><td>{{ r.kwh }}</td><td>{{ r.kwh_band }}</td>
<td>{{ r.heat_cool }}</td><td>{{ r.cost }}</td><td>{{ r.breach }}</td>
<td>{{ r.peak }}</td><td>{{ r.t_in }}</td>
</tr>
{% endfor %}
</tbody>
</table>
<p class="muted">"Breach" = ore/anno in cui la pompa è satura e non tiene il setpoint.</p>

{% if daily_png %}
<div class="section">
<h2>Consumi giornalieri</h2>
<img class="chart" src="data:image/png;base64,{{ daily_png }}" alt="Consumi giornalieri"/>
<p class="muted">● = giorno più gravoso in riscaldamento per ciascuna configurazione.</p>
</div>
{% endif %}

{% if cost_png %}
<div class="section">
<h2>Costo annuo</h2>
<img class="chart" src="data:image/png;base64,{{ cost_png }}" alt="Costo annuo"/>
</div>
{% endif %}

{% if indoor_png %}
<div class="section">
<h2>Temperatura interna</h2>
<img class="chart" src="data:image/png;base64,{{ indoor_png }}" alt="Temperatura interna"/>
</div>
{% endif %}
</body>
</html>"""
)


def build_thermal_lab_pdf(report: Mapping[str, Any], buffer: BinaryIO) -> None:
    """
    Render a thermal-lab comparison into ``buffer`` as a PDF report.

    Args:
        report: The ``/thermal-lab/compare`` response mapping enriched with
            the run-meta keys ``climate_name``, ``dynamic`` and
            ``electricity_price_eur_per_kwh``.
        buffer: Binary stream the WeasyPrint output is written to.

    Raises:
        ValueError: If the report carries no variants.

    Notes:
        WeasyPrint needs ``libpango`` / ``libcairo`` / ``libgdk-pixbuf`` at
        runtime (already documented for Docker in the README).
    """
    variants: Sequence[Any] = report.get("variants") or []
    if not variants:
        raise ValueError("Thermal-lab report has no variants to export.")

    context = {
        "climate_name": report.get("climate_name"),
        "n_paths": report.get("n_paths", "?"),
        "n_years": report.get("n_years", "?"),
        "price_label": report.get("price_label")
        or f"{report.get('electricity_price_eur_per_kwh', '?')} €/kWh",
        "dynamic": bool(report.get("dynamic")),
        "rows": _variant_rows(report),
        "daily_png": _plot_daily(report),
        "cost_png": _plot_cost(report),
        "indoor_png": _plot_indoor(report),
    }
    html_str = _HTML_TEMPLATE.render(**context)
    HTML(string=html_str).write_pdf(target=buffer)
