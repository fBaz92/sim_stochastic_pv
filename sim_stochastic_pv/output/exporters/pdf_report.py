"""
PDF report exporter (Phase 11).

Renders a multi-page summary of a Monte Carlo run using WeasyPrint
(HTML+CSS → PDF) with matplotlib figures embedded as base64 PNGs.

The exporter reads only ``RunResultRecord.summary`` — never the on-disk
output_dir — so reports remain reproducible even after older artefacts
are cleaned up. Legacy runs (predating Phase 11) lack the cashflow_table
and inflation payloads; the renderer degrades gracefully and skips the
sections that have no data.
"""

from __future__ import annotations

import base64
import io
from typing import Any, BinaryIO, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from weasyprint import HTML


# ─────────────────────────────────────────────────────────────────────────
# Chart helpers — each returns a base64-encoded PNG ready for <img src=...>.
# Plots fail soft: when their input data is missing, the helper returns
# None and the template omits the corresponding section.
# ─────────────────────────────────────────────────────────────────────────


def _fig_to_base64(fig) -> str:
    """Serialise a matplotlib figure to a base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_profit_fan(profit_block: Mapping[str, Any] | None) -> str | None:
    if not profit_block:
        return None
    months = profit_block.get("months") or []
    mean = profit_block.get("mean_gain_eur") or []
    p05 = profit_block.get("p05_gain_eur") or []
    p95 = profit_block.get("p95_gain_eur") or []
    if not months:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.fill_between(months, p05, p95, color="#198754", alpha=0.18, label="p05–p95")
    ax.plot(months, mean, color="#198754", lw=2.2, label="Media")
    be_median = profit_block.get("break_even_month_median")
    if be_median is not None:
        ax.axvline(be_median, color="#dc3545", lw=1.5, ls="--", label="Break-even mediano")
    ax.axhline(0, color="#6c757d", lw=0.8, ls=":")
    ax.set_xlabel("Mese dall'inizio")
    ax.set_ylabel("Guadagno cumulato (€)")
    ax.set_title("Proiezione finanziaria (nominale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _plot_energy_bars(energy_block: Mapping[str, Any] | None) -> str | None:
    if not energy_block:
        return None
    months = energy_block.get("months") or []
    if not months:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.bar(months, energy_block.get("pv_prod_mean_kwh", []), color="#ffc107", label="PV")
    ax.bar(
        months,
        energy_block.get("grid_import_mean_kwh", []),
        color="#dc3545",
        label="Da rete",
        bottom=energy_block.get("solar_used_mean_kwh", []),
    )
    ax.bar(
        months,
        energy_block.get("solar_used_mean_kwh", []),
        color="#0d6efd",
        label="Autoconsumo",
    )
    ax.set_xlabel("Mese")
    ax.set_ylabel("Energia (kWh)")
    ax.set_title("Bilancio energetico mensile (medie)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    return _fig_to_base64(fig)


def _plot_price_fan(price_block: Mapping[str, Any] | None) -> str | None:
    if not price_block:
        return None
    months = price_block.get("months") or []
    if not months:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.3))
    p05 = price_block.get("p05_eur_per_kwh", [])
    p95 = price_block.get("p95_eur_per_kwh", [])
    mean = price_block.get("mean_eur_per_kwh", [])
    if p05 and p95:
        ax.fill_between(months, p05, p95, color="#0d6efd", alpha=0.15, label="p05–p95")
    ax.plot(months, mean, color="#0d6efd", lw=2.2, label="Media")
    for path in (price_block.get("sample_paths") or [])[:10]:
        ax.plot(months, path, color="#6c757d", lw=0.6, alpha=0.4)
    ax.set_xlabel("Mese dall'inizio")
    ax.set_ylabel("Prezzo (EUR/kWh)")
    ax.set_title("Traiettorie del prezzo dell'energia")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _plot_inflation_fan(inflation_block: Mapping[str, Any] | None) -> str | None:
    if not inflation_block:
        return None
    years = inflation_block.get("years") or []
    if not years:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.3))
    p05 = inflation_block.get("p05_factor", [])
    p95 = inflation_block.get("p95_factor", [])
    mean = inflation_block.get("mean_factor", [])
    if p05 and p95:
        ax.fill_between(years, p05, p95, color="#6f42c1", alpha=0.15, label="p05–p95")
    ax.plot(years, mean, color="#6f42c1", lw=2.2, label="Media")
    for path in (inflation_block.get("sample_paths") or [])[:10]:
        ax.plot(years, path, color="#6c757d", lw=0.6, alpha=0.4)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Fattore cumulativo")
    ax.set_title("Inflazione attesa")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────
# Cash-flow table — rendered as plain HTML (no matplotlib).
# ─────────────────────────────────────────────────────────────────────────


def _cashflow_rows(cf: Mapping[str, Any] | None, max_rows: int = 240) -> list[dict]:
    """Pre-format cash-flow rows so the Jinja template stays trivial."""
    if not cf:
        return []
    months = cf.get("months", [])
    n = min(len(months), max_rows)
    rows: list[dict] = []
    for i in range(n):
        rows.append(
            {
                "month": months[i],
                "year": months[i] // 12,
                "month_in_year": months[i] % 12,
                "savings": cf.get("mean_savings_eur", [])[i] if i < len(cf.get("mean_savings_eur", [])) else None,
                "savings_real": cf.get("mean_savings_real_eur", [])[i] if i < len(cf.get("mean_savings_real_eur", [])) else None,
                "bonus": cf.get("bonus_per_month_eur", [])[i] if i < len(cf.get("bonus_per_month_eur", [])) else None,
                "profit_cum": cf.get("mean_profit_cum_eur", [])[i] if i < len(cf.get("mean_profit_cum_eur", [])) else None,
                "profit_cum_real": cf.get("mean_profit_cum_real_eur", [])[i] if i < len(cf.get("mean_profit_cum_real_eur", [])) else None,
            }
        )
    return rows


def _fmt_pct(p: float | None) -> str:
    return "—" if p is None else f"{p * 100:.1f} %"


def _fmt_eur(v: float | None) -> str:
    return "—" if v is None else f"€ {v:,.0f}"


def _fmt_months(m: float | None) -> str:
    if m is None:
        return "—"
    m_int = int(round(m)) + 1
    y, r = divmod(m_int, 12)
    if y == 0:
        return f"{r} mesi"
    if r == 0:
        return f"{y} anni"
    return f"{y} anni e {r} mesi"


_HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Report Monte Carlo — {{ scenario }}</title>
<style>
@page { size: A4; margin: 1.5cm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2937; }
h1 { font-size: 20pt; margin: 0 0 0.2em; }
h2 { font-size: 14pt; margin: 1.4em 0 0.4em; color: #0d6efd; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.2em; }
.subtitle { color: #6c757d; font-size: 10pt; margin-bottom: 1.2em; }
.kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5em 1.2em; }
.kpi { border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.5em 0.8em; }
.kpi .label { font-size: 8pt; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 13pt; font-weight: 700; }
img.chart { width: 100%; height: auto; page-break-inside: avoid; }
table.cashflow { width: 100%; border-collapse: collapse; font-size: 8.5pt; margin-top: 0.5em; }
table.cashflow th, table.cashflow td { border: 1px solid #e2e8f0; padding: 0.2em 0.4em; text-align: right; }
table.cashflow th { background: #f8f9fa; }
table.cashflow td:first-child { text-align: left; }
.section { page-break-inside: avoid; }
.section.full-page { page-break-before: always; }
.muted { color: #6c757d; font-size: 9pt; }
</style>
</head>
<body>
<h1>{{ scenario }}</h1>
<p class="subtitle">Report Monte Carlo — Run #{{ run_id }}</p>

<h2>Decisione</h2>
<div class="kpi-grid">
    <div class="kpi"><div class="label">Probabilità di guadagno</div><div class="value">{{ prob_gain }}</div></div>
    <div class="kpi"><div class="label">Break-even atteso</div><div class="value">{{ break_even }}</div></div>
    <div class="kpi"><div class="label">IRR atteso</div><div class="value">{{ irr_mean }}</div></div>
    <div class="kpi"><div class="label">NPV mediano</div><div class="value">{{ npv }}</div></div>
    {% if tax_bonus_total %}
    <div class="kpi"><div class="label">Bonus fiscale totale</div><div class="value">{{ tax_bonus_total }}</div></div>
    {% endif %}
    <div class="kpi"><div class="label">Guadagno reale (fine orizzonte)</div><div class="value">{{ final_gain_real }}</div></div>
</div>

{% if profit_png %}
<div class="section full-page">
<h2>Proiezione finanziaria</h2>
<img class="chart" src="data:image/png;base64,{{ profit_png }}" alt="Profit fan chart"/>
</div>
{% endif %}

{% if energy_png %}
<div class="section">
<h2>Bilancio energetico</h2>
<img class="chart" src="data:image/png;base64,{{ energy_png }}" alt="Energy bars"/>
</div>
{% endif %}

{% if price_png %}
<div class="section">
<h2>Prezzo dell'energia</h2>
<img class="chart" src="data:image/png;base64,{{ price_png }}" alt="Price fan chart"/>
</div>
{% endif %}

{% if inflation_png %}
<div class="section">
<h2>Inflazione attesa</h2>
<img class="chart" src="data:image/png;base64,{{ inflation_png }}" alt="Inflation fan chart"/>
</div>
{% endif %}

{% if rows %}
<div class="section full-page">
<h2>Cash flow medio mensile</h2>
<p class="muted">Tutti i valori sono medie cross-path (Monte Carlo). Il bonus fiscale è
una colonna separata anche se è già incluso nei risparmi nominali.</p>
<table class="cashflow">
<thead><tr>
<th>Mese</th><th>Risp. nom. (€)</th><th>Risp. reale (€)</th>
<th>Bonus (€)</th><th>Prof. cum. nom. (€)</th><th>Prof. cum. reale (€)</th>
</tr></thead>
<tbody>
{% for r in rows %}
<tr>
<td>{{ r.year }}-{{ "%02d"|format(r.month_in_year + 1) }}</td>
<td>{{ r.savings|round(2) if r.savings is not none else "—" }}</td>
<td>{{ r.savings_real|round(2) if r.savings_real is not none else "—" }}</td>
<td>{{ r.bonus|round(2) if r.bonus else "" }}</td>
<td>{{ r.profit_cum|round(2) if r.profit_cum is not none else "—" }}</td>
<td>{{ r.profit_cum_real|round(2) if r.profit_cum_real is not none else "—" }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
{% endif %}
</body>
</html>"""
)


def build_pdf_report(
    summary: Mapping[str, Any], run_id: int, buffer: BinaryIO
) -> None:
    """
    Render the run summary as a PDF report into ``buffer``.

    Args:
        summary: The ``RunResultRecord.summary`` JSON. Older runs without
            ``plots_data.cashflow_table`` are still supported — the
            cash-flow section is simply omitted.
        run_id: Database id of the run, shown in the report subtitle.
        buffer: Binary stream the WeasyPrint output is written to.

    Notes:
        - Uses ``matplotlib`` (Agg backend) to render figures as PNG and
          embeds them inline via base64 data URIs — no temporary files.
        - WeasyPrint needs ``libpango``, ``libcairo`` and ``libgdk-pixbuf``
          at runtime (already available on macOS via brew; documented
          for Docker in the README + Dockerfile.backend).
    """
    plots_data = summary.get("plots_data") or {}

    context = {
        "scenario": summary.get("scenario", "Scenario"),
        "run_id": run_id,
        "prob_gain": _fmt_pct(summary.get("prob_gain")),
        "break_even": _fmt_months(summary.get("break_even_month_median")),
        "irr_mean": _fmt_pct(summary.get("irr_mean")),
        "npv": _fmt_eur(summary.get("npv_median_eur")),
        "final_gain_real": _fmt_eur(summary.get("final_gain_real_mean_eur")),
        "tax_bonus_total": (
            _fmt_eur(summary.get("tax_bonus_total_eur"))
            if summary.get("tax_bonus_total_eur")
            else None
        ),
        "profit_png": _plot_profit_fan(plots_data.get("profit")),
        "energy_png": _plot_energy_bars(plots_data.get("energy_monthly")),
        "price_png": _plot_price_fan(plots_data.get("price")),
        "inflation_png": _plot_inflation_fan(plots_data.get("inflation")),
        "rows": _cashflow_rows(plots_data.get("cashflow_table")),
    }

    html_str = _HTML_TEMPLATE.render(**context)
    HTML(string=html_str).write_pdf(target=buffer)
