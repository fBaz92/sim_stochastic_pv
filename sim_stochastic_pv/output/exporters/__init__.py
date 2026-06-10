"""
File exporters for run results (Phase 11).

Two formats are supported:

- ``xlsx_cashflow``: a 2-sheet Excel workbook with the mean monthly
  cash-flow series and the run KPIs.
- ``pdf_report``: a multi-page PDF rendered via WeasyPrint that mirrors
  the Dashboard (Decisione cards + all the charts + cash-flow table).

Both readers consume the ``RunResultRecord.summary`` JSON only, so they
work even when the original ``output_dir`` on disk has been deleted.
"""

from .pdf_design_report import build_design_report_pdf
from .pdf_market import build_market_pdf
from .pdf_report import build_pdf_report
from .pdf_thermal_lab import build_thermal_lab_pdf
from .xlsx_cashflow import build_cashflow_xlsx
from .xlsx_load_profile import build_template_xlsx, parse_load_profile_xlsx
from .xlsx_market import build_market_xlsx
from .xlsx_thermal_lab import build_thermal_lab_xlsx

__all__ = [
    "build_cashflow_xlsx",
    "build_design_report_pdf",
    "build_market_pdf",
    "build_market_xlsx",
    "build_pdf_report",
    "build_template_xlsx",
    "build_thermal_lab_pdf",
    "build_thermal_lab_xlsx",
    "parse_load_profile_xlsx",
]
