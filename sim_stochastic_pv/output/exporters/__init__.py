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

from .pdf_report import build_pdf_report
from .xlsx_cashflow import build_cashflow_xlsx
from .xlsx_load_profile import build_template_xlsx, parse_load_profile_xlsx

__all__ = [
    "build_cashflow_xlsx",
    "build_pdf_report",
    "build_template_xlsx",
    "parse_load_profile_xlsx",
]
