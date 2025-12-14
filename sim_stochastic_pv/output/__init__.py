"""
Output generation and reporting for PV system analysis.
"""

from .result_builder import ResultBuilder
from .report import generate_report

__all__ = [
    "ResultBuilder",
    "generate_report",
]
