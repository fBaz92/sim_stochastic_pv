from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Payload for triggering an analysis via API."""
    n_mc: Optional[int] = Field(default=None, ge=1, description="Numero di percorsi Monte Carlo")
    seed: Optional[int] = Field(default=None, description="Seed RNG per l'analisi")


class AnalysisResponse(BaseModel):
    """Response body with aggregated analysis metrics."""
    scenario: str
    final_gain_mean_eur: float
    final_gain_real_mean_eur: float
    prob_gain: float
    output_dir: Optional[str] = None


class OptimizationRequest(BaseModel):
    """Payload for optimization requests."""
    seed: Optional[int] = Field(default=None, description="Seed RNG per l'ottimizzazione")


class OptimizationResponse(BaseModel):
    """Summary of optimization execution."""
    evaluations: int
    output_dir: Optional[str] = None


class RunResult(BaseModel):
    """Serialized representation of stored run results."""
    id: int
    result_type: str
    summary: Dict[str, Any]
    scenario_id: Optional[int]
    optimization_id: Optional[int]
    created_at: datetime

    class Config:
        orm_mode = True
