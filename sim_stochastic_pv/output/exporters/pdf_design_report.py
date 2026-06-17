"""
PDF technical report ("relazione tecnica") for a detailed plant design.

Renders the full electrical design — components, temperature-corrected
values, string sizing with margins, current checks, temperature margins,
string protections and the DC-cable comparison — via WeasyPrint
(HTML+CSS → PDF), in the structure an Italian residential PV design
dossier expects.

Normative note baked into the document: the array design references
CEI EN 62548 (PV array design requirements), CEI 64-8 (low-voltage
electrical installations) and the CEI 82-25 guide; CEI 0-21 governs the
*connection* to the LV grid and is referenced in that role only.

The exporter consumes the plant-design record plus a freshly computed
:class:`~sim_stochastic_pv.simulation.electrical_design.DesignEvaluation`
(the caller re-runs the engine from the saved designer inputs, so the
report always reflects the engine's current formulas), and the optional
production-preview result.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any, BinaryIO, Mapping

from jinja2 import Template
# WeasyPrint is imported lazily inside the render function so the API and the
# test suite can boot without its native libraries (PDF export is an optional
# feature; the import only needs to succeed when a PDF is actually requested).

from ...simulation.electrical_design import DesignEvaluation
from ...simulation.electrical_design.production import ProductionPreviewResult


_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<style>
    @page { size: A4; margin: 2cm 1.8cm; }
    body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; color: #1a202c; }
    h1 { font-size: 16pt; margin: 0 0 2pt; }
    h2 { font-size: 11.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #cbd5e0; padding-bottom: 2pt; }
    .subtitle { color: #4a5568; margin: 0 0 10pt; font-size: 9pt; }
    table { width: 100%; border-collapse: collapse; margin: 4pt 0; }
    th, td { border: 0.5pt solid #cbd5e0; padding: 2.5pt 5pt; text-align: left; font-size: 8.5pt; }
    th { background: #edf2f7; }
    td.num, th.num { text-align: right; }
    .ok { color: #276749; font-weight: bold; }
    .ko { color: #c53030; font-weight: bold; }
    .note { font-size: 8pt; color: #4a5568; }
    ul { margin: 3pt 0; padding-left: 14pt; }
    li { margin-bottom: 1.5pt; }
    .recommended { background: #f0fff4; }
</style>
</head>
<body>
    <h1>Relazione tecnica di dimensionamento — {{ name }}</h1>
    <p class="subtitle">
        Generata il {{ today }} · livello di progetto: dettagliato
        {% if location_name %}· sito: {{ location_name }}{% endif %}
    </p>

    <h2>1. Riferimenti normativi</h2>
    <ul>
        <li><strong>CEI EN 62548</strong> — Requisiti di progettazione per schiere fotovoltaiche (dimensionamento stringhe e protezioni).</li>
        <li><strong>CEI 64-8</strong> — Impianti elettrici utilizzatori a tensione nominale ≤ 1000 V (cablaggi e protezioni).</li>
        <li><strong>Guida CEI 82-25</strong> — Guida alla realizzazione di sistemi di generazione fotovoltaica.</li>
        <li><strong>CEI 0-21</strong> — Regola tecnica di riferimento per la connessione alla rete BT (lato connessione, non oggetto del presente dimensionamento).</li>
    </ul>

    <h2>2. Componenti selezionati</h2>
    <table>
        <tr><th></th><th>Modulo fotovoltaico</th><th>Inverter</th></tr>
        <tr><td>Potenza nominale</td>
            <td class="num">{{ panel.power_w }} W</td>
            <td class="num">{{ inverter.p_ac_nom_kw }} kW AC</td></tr>
        <tr><td>Tensioni caratteristiche (STC)</td>
            <td>Voc {{ panel.v_oc_stc_v }} V · Vmp {{ panel.v_mpp_stc_v }} V</td>
            <td>Vdc max {{ inverter.v_dc_max_v }} V · MPPT pieno carico
                {{ inverter.v_mppt_full_load_min_v or inverter.v_mppt_min_v }}–{{ inverter.v_mppt_full_load_max_v or inverter.v_mppt_max_v }} V</td></tr>
        <tr><td>Correnti caratteristiche</td>
            <td>Isc {{ panel.i_sc_stc_a }} A · Imp {{ panel.i_mpp_stc_a }} A</td>
            <td>I max op/MPPT {{ inverter.i_dc_max_per_mppt_a }} A · Isc max/MPPT {{ inverter.i_sc_max_per_mppt_a }} A</td></tr>
        <tr><td>Coefficienti / architettura</td>
            <td>α {{ panel.alpha_isc_pct_per_c }} · β {{ panel.beta_voc_pct_per_c }} · γ {{ panel.gamma_pmax_pct_per_c }} %/°C</td>
            <td>{{ inverter.n_mppt_trackers }} MPPT · max {{ inverter.max_strings_per_mppt }} stringa/e per MPPT</td></tr>
        <tr><td>Limiti di sistema</td>
            <td>V sistema {{ panel.v_system_max_v }} V · max fusibile {{ panel.max_series_fuse_a or "n.d." }} A</td>
            <td>Rendimento max {{ "%.1f" | format((inverter.efficiency_max or 0) * 100) }} %</td></tr>
    </table>

    <h2>3. Parametri di progetto</h2>
    <table>
        <tr>
            <td>Temperatura minima del sito</td><td class="num">{{ site.t_min_c }} °C</td>
            <td>Potenza AC richiesta</td><td class="num">{{ requirements.p_ac_required_kw }} kW</td>
        </tr>
        <tr>
            <td>Temperatura massima del sito</td><td class="num">{{ site.t_max_c }} °C</td>
            <td>Rapporto DC/AC obiettivo</td><td class="num">{{ requirements.target_dc_ac_ratio }}</td>
        </tr>
        <tr>
            <td>Sovratemperatura di cella (ΔT)</td><td class="num">{{ site.delta_t_cell_c }} °C</td>
            <td>Moduli per stringa (scelta)</td><td class="num">{{ requirements.n_panels_per_string }}</td>
        </tr>
        <tr>
            <td>Fattore di sicurezza su Isc</td><td class="num">{{ requirements.safety_factor_isc }}</td>
            <td>Perdita massima ammessa nei cavi DC</td><td class="num">{{ "%.1f" | format(requirements.max_cable_loss_fraction * 100) }} %</td>
        </tr>
    </table>

    <h2>4. Valori del modulo corretti in temperatura</h2>
    <p class="note">Modello lineare IEC: X(T) = X<sub>STC</sub> · (1 + coeff/100 · (T − 25)). Caso freddo: cella alla T minima ambiente; caso caldo: cella a T max + ΔT. V<sub>mp</sub> corretta con γ (prassi conservativa).</p>
    <table>
        <tr><th>Grandezza</th><th class="num">Valore</th><th>Condizione</th></tr>
        <tr><td>Voc a cella fredda</td><td class="num">{{ "%.2f" | format(ev.corrected.v_oc_cold_v) }} V</td><td>T cella {{ "%.0f" | format(ev.corrected.t_cell_cold_c) }} °C — massima tensione possibile</td></tr>
        <tr><td>Vmp a cella fredda</td><td class="num">{{ "%.2f" | format(ev.corrected.v_mp_cold_v) }} V</td><td>verifica inseguimento MPPT invernale</td></tr>
        <tr><td>Vmp a cella calda</td><td class="num">{{ "%.2f" | format(ev.corrected.v_mp_hot_v) }} V</td><td>T cella {{ "%.0f" | format(ev.corrected.t_cell_hot_c) }} °C — minima tensione operativa</td></tr>
        <tr><td>Isc a cella calda</td><td class="num">{{ "%.2f" | format(ev.corrected.i_sc_hot_a) }} A</td><td>massima corrente operativa</td></tr>
        <tr><td>Isc di progetto (cavi/protezioni)</td><td class="num">{{ "%.2f" | format(ev.corrected.i_sc_design_a) }} A</td><td>Isc calda × {{ requirements.safety_factor_isc }}</td></tr>
    </table>

    <h2>5. Dimensionamento delle stringhe</h2>
    <table>
        <tr><td>Tensione limite di stringa (min fra Vdc,max inverter e V sistema modulo)</td><td class="num">{{ "%.0f" | format(ev.bounds.v_limit_v) }} V</td></tr>
        <tr><td>Range ammissibile moduli per stringa</td><td class="num">{{ ev.bounds.n_min }} – {{ ev.bounds.n_max }}</td></tr>
        <tr><td>Moduli per stringa scelti</td><td class="num">{{ requirements.n_panels_per_string }}
            {% if ev.voltages.n_in_range %}<span class="ok">(nel range)</span>{% else %}<span class="ko">(FUORI RANGE)</span>{% endif %}</td></tr>
    </table>
    <table>
        <tr><th>Verifica di tensione</th><th class="num">Valore</th><th class="num">Margine</th><th>Esito</th></tr>
        <tr><td>Voc stringa a freddo ≤ {{ "%.0f" | format(ev.bounds.v_limit_v) }} V</td>
            <td class="num">{{ "%.1f" | format(ev.voltages.v_oc_string_cold_v) }} V</td>
            <td class="num">{{ "%.1f" | format(ev.voltages.v_oc_margin_v) }} V</td>
            <td>{% if ev.voltages.v_oc_margin_v >= 0 %}<span class="ok">OK</span>{% else %}<span class="ko">NON SUPERATA</span>{% endif %}</td></tr>
        <tr><td>Vmp stringa a caldo ≥ V MPPT min</td>
            <td class="num">{{ "%.1f" | format(ev.voltages.v_mp_string_hot_v) }} V</td>
            <td class="num">{{ "%.1f" | format(ev.voltages.v_mp_hot_margin_v) }} V</td>
            <td>{% if ev.voltages.v_mp_hot_margin_v >= 0 %}<span class="ok">OK</span>{% else %}<span class="ko">NON SUPERATA</span>{% endif %}</td></tr>
        <tr><td>Vmp stringa a freddo ≤ V MPPT max (inseguimento)</td>
            <td class="num">{{ "%.1f" | format(ev.voltages.v_mp_string_cold_v) }} V</td>
            <td class="num">{{ "%.1f" | format(ev.voltages.v_mp_cold_margin_v) }} V</td>
            <td>{% if ev.voltages.v_mp_cold_margin_v >= 0 %}<span class="ok">OK</span>{% else %}<span class="ko">NON SUPERATA</span>{% endif %}</td></tr>
    </table>

    <h2>6. Taglia dell'impianto</h2>
    <table>
        <tr><td>Potenza DC obiettivo</td><td class="num">{{ "%.2f" | format(ev.plant.p_dc_target_kwp) }} kWp</td>
            <td>Numero di stringhe (identiche)</td><td class="num">{{ ev.plant.n_strings }}</td></tr>
        <tr><td>Potenza di una stringa</td><td class="num">{{ "%.2f" | format(ev.plant.string_power_kwp) }} kWp</td>
            <td>Moduli totali</td><td class="num">{{ ev.plant.total_panels }}</td></tr>
        <tr><td><strong>Potenza DC installata</strong></td><td class="num"><strong>{{ "%.2f" | format(ev.plant.p_dc_installed_kwp) }} kWp</strong></td>
            <td>Rapporto DC/AC effettivo</td><td class="num">{{ "%.2f" | format(ev.plant.dc_ac_ratio) }}</td></tr>
    </table>

    <h2>7. Verifiche di corrente per MPPT (caso peggiore)</h2>
    <table>
        <tr><th>Verifica</th><th class="num">Valore</th><th class="num">Limite</th><th>Esito</th></tr>
        <tr><td>Ingressi fisici ({{ ev.currents.strings_per_mppt }} stringa/e per MPPT)</td>
            <td class="num">{{ ev.currents.strings_per_mppt }}</td>
            <td class="num">{{ inverter.max_strings_per_mppt }}</td>
            <td>{% if ev.currents.inputs_ok %}<span class="ok">OK</span>{% else %}<span class="ko">NON SUPERATA</span>{% endif %}</td></tr>
        <tr><td>Corrente operativa (Imp a cella calda)</td>
            <td class="num">{{ "%.2f" | format(ev.currents.i_operating_a) }} A</td>
            <td class="num">{{ inverter.i_dc_max_per_mppt_a }} A</td>
            <td>{% if ev.currents.i_operating_margin_a >= 0 %}<span class="ok">OK</span>{% else %}<span class="ko">NON SUPERATA ({{ "%.2f" | format(ev.currents.i_operating_margin_a) }} A)</span>{% endif %}</td></tr>
        <tr><td>Corrente di cortocircuito (Isc a cella calda)</td>
            <td class="num">{{ "%.2f" | format(ev.currents.i_sc_a) }} A</td>
            <td class="num">{{ inverter.i_sc_max_per_mppt_a }} A</td>
            <td>{% if ev.currents.i_sc_margin_a >= 0 %}<span class="ok">OK</span>{% else %}<span class="ko">NON SUPERATA ({{ "%.2f" | format(ev.currents.i_sc_margin_a) }} A)</span>{% endif %}</td></tr>
    </table>

    <h2>8. Margini di temperatura del design</h2>
    <table>
        <tr><td>Temperatura di cella minima ammissibile (vincolo Voc)</td>
            <td class="num">{{ "%.1f" | format(ev.margins.t_min_admissible_c) }} °C</td>
            <td>margine sul caso freddo: <strong>{{ "%.1f" | format(ev.margins.margin_cold_c) }} °C</strong></td></tr>
        <tr><td>Temperatura ambiente massima ammissibile (vincolo Vmp ≥ MPPT min)</td>
            <td class="num">{{ "%.1f" | format(ev.margins.t_amb_max_admissible_c) }} °C</td>
            <td>margine sul caso caldo: <strong>{{ "%.1f" | format(ev.margins.margin_hot_c) }} °C</strong></td></tr>
        <tr><td>Esito robustezza termica</td>
            <td colspan="2">{% if ev.margins.robust %}<span class="ok">Design robusto sulle temperature del sito</span>{% else %}<span class="ko">Margine negativo: rivedere il numero di moduli per stringa</span>{% endif %}</td></tr>
    </table>

    <h2>9. Protezioni di stringa (CEI EN 62548)</h2>
    <table>
        <tr><td>Protezione richiesta</td>
            <td>{{ "SÌ (≥ 3 stringhe in parallelo)" if ev.protection.protection_required else "NO (< 3 stringhe in parallelo)" }}</td></tr>
        <tr><td>Finestra normativa del fusibile (1,5–2,4 × Isc STC)</td>
            <td class="num">{{ "%.1f" | format(ev.protection.i_fuse_min_a) }} – {{ "%.1f" | format(ev.protection.i_fuse_max_norm_a) }} A</td></tr>
        <tr><td>Taglia standard gPV consigliata</td>
            <td class="num">{{ ev.protection.recommended_fuse_a or "—" }} A
                {% if ev.protection.fuse_within_module_limit == false %}<span class="ko">(supera il max del modulo!)</span>{% endif %}</td></tr>
    </table>

    <h2>10. Cavi DC — confronto sezioni</h2>
    <p class="note">Perdita ohmica andata+ritorno su {{ cable_length_m }} m a Imp STC, resistività rame a {{ cable_temp_c }} °C
        (ρ = {{ "%.5f" | format(ev.cables.resistivity_ohm_mm2_per_m) }} Ω·mm²/m). Sezione scelta: <strong>{{ chosen_section or "—" }} mm²</strong>.</p>
    <table>
        <tr><th>Sezione</th><th class="num">ΔV</th><th class="num">Caduta</th><th class="num">Perdita totale</th>
            <th class="num">% su DC</th><th class="num">Iz</th><th>Esito</th></tr>
        {% for row in ev.cables.rows %}
        <tr {% if row.section_mm2 == ev.cables.recommended_section_mm2 %}class="recommended"{% endif %}>
            <td>{{ row.section_mm2 }} mm²{% if row.section_mm2 == ev.cables.recommended_section_mm2 %} ★{% endif %}</td>
            <td class="num">{{ "%.2f" | format(row.voltage_drop_v) }} V</td>
            <td class="num">{{ "%.2f" | format(row.voltage_drop_fraction * 100) }} %</td>
            <td class="num">{{ "%.0f" | format(row.loss_total_kw * 1000) }} W</td>
            <td class="num">{{ "%.2f" | format(row.loss_fraction_of_dc * 100) }} %</td>
            <td class="num">{{ row.iz_a or "—" }}</td>
            <td>{% if row.loss_ok and row.iz_ok != false %}<span class="ok">OK</span>{% elif row.iz_ok == false %}<span class="ko">Iz insufficiente</span>{% else %}Oltre soglia{% endif %}</td>
        </tr>
        {% endfor %}
    </table>

    {% if production %}
    <h2>11. Produzione attesa (Monte Carlo orario, {{ production.n_paths }} scenari meteo)</h2>
    <table>
        <tr><td><strong>Energia AC attesa</strong></td>
            <td class="num"><strong>{{ "%.0f" | format(production.annual_ac_kwh_mean) }} kWh/anno</strong>
                [{{ "%.0f" | format(production.annual_ac_kwh_p05) }} – {{ "%.0f" | format(production.annual_ac_kwh_p95) }}]</td></tr>
        <tr><td>Energia DC lorda</td><td class="num">{{ "%.0f" | format(production.annual_dc_kwh_mean) }} kWh/anno</td></tr>
        <tr><td>Clipping inverter (DC/AC {{ "%.2f" | format(ev.plant.dc_ac_ratio) }})</td>
            <td class="num">{{ "%.0f" | format(production.clipping_kwh_mean) }} kWh ({{ "%.1f" | format(production.clipping_fraction * 100) }} %)</td></tr>
        <tr><td>Perdite cavi DC</td>
            <td class="num">{{ "%.0f" | format(production.cable_loss_kwh_mean) }} kWh ({{ "%.2f" | format(production.cable_loss_fraction * 100) }} %)</td></tr>
        {% if production.electrical_derating_kwh_mean > 0 %}
        <tr><td>Derating elettrico (finestra MPPT + temperatura)</td>
            <td class="num">{{ "%.0f" | format(production.electrical_derating_kwh_mean) }} kWh</td></tr>
        {% endif %}
    </table>
    {% endif %}

    <h2>{{ "12" if production else "11" }}. Ipotesi e limiti del dimensionamento</h2>
    <ul>
        <li>Caso freddo: temperatura di cella pari alla minima ambiente (alba fredda e soleggiata, modulo non riscaldato) — condizione di Voc massima.</li>
        <li>Caso caldo: temperatura di cella pari alla massima ambiente + ΔT da NOCT in pieno sole.</li>
        <li>La correzione di Vmp usa il coefficiente γ(Pmax): prassi conservativa quando il datasheet non pubblica il coefficiente specifico di Vmp.</li>
        <li>Tutte le stringhe sono identiche; con stringhe non multiple degli MPPT le verifiche usano il tracker più caricato.</li>
        <li>Il fattore {{ requirements.safety_factor_isc }} su Isc è riservato a cavi e protezioni (irraggiamenti superiori a 1000 W/m²: riflessione neve, alta quota).</li>
        <li>Le perdite cavo della tabella sono calcolate a Imp STC; la sezione "Produzione attesa" (quando presente) le integra ora per ora sul meteo stocastico del sito (∝ I²).</li>
        <li>Esclusi dal presente dimensionamento: lato AC (cavo inverter–POC), ombreggiamenti e mismatch, moduli bifacciali, derating dell'inverter per temperatura ambiente e altitudine.</li>
        <li>Verificare il coordinamento delle protezioni e la posa dei cavi secondo CEI 64-8 / CEI-UNEL applicabili.</li>
    </ul>
</body>
</html>
"""
)


def build_design_report_pdf(
    *,
    name: str,
    location_name: str | None,
    panel: Mapping[str, Any],
    inverter: Mapping[str, Any],
    site: Mapping[str, Any],
    requirements: Mapping[str, Any],
    evaluation: DesignEvaluation,
    chosen_section_mm2: float | None,
    cable_length_m: float,
    cable_temp_c: float,
    production: ProductionPreviewResult | None = None,
    output: BinaryIO | None = None,
) -> BinaryIO:
    """
    Render the technical report PDF for a detailed plant design.

    Args:
        name: Design name (report title).
        location_name: Installation-site name, when known.
        panel: Designer panel datasheet dict (as stored in
            ``design.data.designer.panel``).
        inverter: Designer inverter datasheet dict.
        site: Site corners dict (``t_min_c``, ``t_max_c``,
            ``delta_t_cell_c``).
        requirements: Requirements dict (``p_ac_required_kw``,
            ``target_dc_ac_ratio``, ``n_panels_per_string``,
            ``safety_factor_isc``, ``max_cable_loss_fraction``).
        evaluation: Freshly computed engine evaluation.
        chosen_section_mm2: Cable section picked by the designer.
        cable_length_m: One-way string-run length used by the table (m).
        cable_temp_c: Cable operating temperature (°C).
        production: Optional hourly-MC production preview to include.
        output: Optional writable binary stream; a fresh ``BytesIO`` is
            created when omitted.

    Returns:
        The stream containing the rendered PDF, positioned at 0.
    """
    html = _TEMPLATE.render(
        name=name,
        today=date.today().strftime("%d/%m/%Y"),
        location_name=location_name,
        panel=panel,
        inverter=inverter,
        site=site,
        requirements=_RequirementsView(requirements),
        ev=evaluation,
        chosen_section=chosen_section_mm2,
        cable_length_m=cable_length_m,
        cable_temp_c=cable_temp_c,
        production=production,
    )
    stream = output or io.BytesIO()
    from weasyprint import HTML  # lazy: native libs only needed to render
    HTML(string=html).write_pdf(stream)
    stream.seek(0)
    return stream


class _RequirementsView:
    """Attribute view over the requirements mapping with safe defaults."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.p_ac_required_kw = data.get("p_ac_required_kw")
        self.target_dc_ac_ratio = data.get("target_dc_ac_ratio", 1.2)
        self.n_panels_per_string = data.get("n_panels_per_string")
        self.safety_factor_isc = data.get("safety_factor_isc", 1.25)
        self.max_cable_loss_fraction = data.get("max_cable_loss_fraction", 0.01)
