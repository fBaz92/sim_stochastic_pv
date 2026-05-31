"""
PDF report exporter for the electricity-market lab.

Renders a market-lab run via WeasyPrint (HTML+CSS → PDF) with matplotlib
figures embedded as base64 PNGs, mirroring the thermal-lab PDF exporter.

Sections:
- a summary table (scenario, mean price, surface size);
- the display-year wholesale price heatmap (month × hour);
- the annual price fan chart (mean + p05/p95 band);
- the price duration curve;
- the installed-capacity stacked-area chart over the horizon;
- a "who sets the price" share-of-year bar chart.

The exporter consumes a plain mapping (the ``/market/run`` response enriched
with run-meta keys), never the simulation objects, so it is testable without
re-running the market engine.
"""

from __future__ import annotations

import base64
import io
from typing import Any, BinaryIO, Mapping

import matplotlib

matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from weasyprint import HTML

_MONTHS_SHORT = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                 "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
_TECH_COLORS = {
    "gas": "#ef4444",
    "coal": "#6b7280",
    "nuclear": "#8b5cf6",
    "wind": "#10b981",
    "solar": "#f59e0b",
    "hydro_mustrun": "#3b82f6",
    "import": "#ec4899",
}


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tech_color(tech: str, idx: int) -> str:
    """Stable colour for a technology (named palette, hashed fallback)."""
    fallback = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#6b7280", "#ec4899"]
    return _TECH_COLORS.get(tech, fallback[idx % len(fallback)])


def _heatmap_png(report: Mapping[str, Any]) -> str | None:
    """Month × hour wholesale-price heatmap."""
    grid = report.get("price_heatmap_eur_per_kwh")
    if not grid:
        return None
    arr = np.asarray(grid, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(arr, aspect="auto", origin="upper", cmap="viridis")
    ax.set_yticks(range(12))
    ax.set_yticklabels(_MONTHS_SHORT)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Ora del giorno")
    ax.set_title(f"Prezzo all'ingrosso €/kWh — anno {report.get('display_year', 0)}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _fig_to_base64(fig)


def _fan_png(report: Mapping[str, Any]) -> str | None:
    """Annual price fan chart (mean + p05/p95 band)."""
    years = report.get("years")
    mean = report.get("annual_price_mean_eur_per_kwh")
    if not years or not mean:
        return None
    p05 = report.get("annual_price_p05_eur_per_kwh") or mean
    p95 = report.get("annual_price_p95_eur_per_kwh") or mean
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.fill_between(years, p05, p95, color="#3b82f6", alpha=0.2, label="p05–p95")
    ax.plot(years, mean, color="#1d4ed8", marker="o", ms=3, label="media")
    ax.set_xlabel("Anno")
    ax.set_ylabel("€/kWh")
    ax.set_title("Prezzo medio annuale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _duration_png(report: Mapping[str, Any]) -> str | None:
    """Price duration curve."""
    x = report.get("duration_curve_x")
    y = report.get("duration_curve_price_eur_per_kwh")
    if not x or not y:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.plot(np.asarray(x) * 100.0, y, color="#0d9488")
    ax.set_xlabel("% delle ore dell'anno")
    ax.set_ylabel("€/kWh")
    ax.set_title("Curva di durata del prezzo")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _capacity_png(report: Mapping[str, Any]) -> str | None:
    """Installed-capacity stacked-area chart over the horizon."""
    techs = report.get("techs")
    years = report.get("years")
    cap = report.get("capacity_by_year_gw")
    if not techs or not years or not cap:
        return None
    series = [np.asarray(cap.get(t, [0] * len(years)), dtype=float) for t in techs]
    colors = [_tech_color(t, i) for i, t in enumerate(techs)]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.stackplot(years, *series, labels=techs, colors=colors, alpha=0.85)
    ax.set_xlabel("Anno")
    ax.set_ylabel("GW")
    ax.set_title("Capacità installata per tecnologia")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _setter_png(report: Mapping[str, Any]) -> str | None:
    """Who-sets-the-price share-of-year bar chart."""
    share = report.get("price_setter_share_year") or {}
    if not share:
        return None
    items = sorted(share.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [100.0 * float(v) for _, v in items]
    colors = [_tech_color(t, i) for i, t in enumerate(labels)]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("% delle ore dell'anno")
    ax.set_title("Chi fissa il prezzo (quota annuale)")
    ax.grid(True, axis="x", alpha=0.3)
    return _fig_to_base64(fig)


_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8" />
<style>
  @page { size: A4; margin: 1.6cm; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #1f2937; }
  h1 { font-size: 18px; margin: 0 0 2px 0; }
  .sub { color: #6b7280; font-size: 11px; margin-bottom: 14px; }
  table.meta { border-collapse: collapse; margin-bottom: 16px; font-size: 11px; }
  table.meta td { border: 1px solid #e5e7eb; padding: 4px 8px; }
  table.meta td.k { background: #f3f4f6; font-weight: 600; }
  .fig { margin: 10px 0 16px 0; }
  .fig img { width: 100%; }
</style>
</head>
<body>
  <h1>Mercato elettrico — report</h1>
  <div class="sub">{{ subtitle }}</div>

  <table class="meta">
    <tr><td class="k">Anno orizzonte</td><td>{{ display_year }}</td>
        <td class="k">Prezzo medio</td><td>{{ mean_price }} €/kWh</td></tr>
    <tr><td class="k">Scenario gas</td><td>{{ gas }}</td>
        <td class="k">Scenario CO₂</td><td>{{ co2 }}</td></tr>
    <tr><td class="k">Traiettorie</td><td>{{ n_traj }}</td>
        <td class="k">Run (chi fissa prezzo)</td><td>{{ n_runs }}</td></tr>
  </table>

  {% for img in figures %}
  <div class="fig"><img src="data:image/png;base64,{{ img }}" /></div>
  {% endfor %}
</body>
</html>
"""
)


def build_market_pdf(report: Mapping[str, Any], buffer: BinaryIO) -> None:
    """
    Render a market-lab run into ``buffer`` as a PDF report.

    Args:
        report: A mapping with the ``/market/run`` response keys, optionally
            enriched with ``gas_scenario`` / ``co2_scenario`` / ``coal_scenario``.
        buffer: Binary stream the WeasyPrint output is written to.

    Raises:
        ValueError: If ``report`` carries no price heatmap — nothing to export.
    """
    if not report.get("price_heatmap_eur_per_kwh"):
        raise ValueError("Market report has no price heatmap to export.")

    figures = [
        fig
        for fig in (
            _heatmap_png(report),
            _fan_png(report),
            _duration_png(report),
            _capacity_png(report),
            _setter_png(report),
        )
        if fig is not None
    ]

    n_techs = len(report.get("techs") or [])
    n_years = len(report.get("years") or [])
    html_str = _TEMPLATE.render(
        subtitle=f"{n_techs} tecnologie · orizzonte {n_years} anni",
        display_year=report.get("display_year", 0),
        mean_price=f"{report.get('mean_price_eur_per_kwh', 0.0):.4f}",
        gas=report.get("gas_scenario") or "—",
        co2=report.get("co2_scenario") or "—",
        n_traj=report.get("n_trajectories", "—"),
        n_runs=report.get("n_runs", "—"),
        figures=figures,
    )
    HTML(string=html_str).write_pdf(target=buffer)
