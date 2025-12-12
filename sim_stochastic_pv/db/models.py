from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from .session import Base


class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class InverterModel(Base, TimestampMixin):
    __tablename__ = "inverters"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model_number = Column(String(255), nullable=True)
    nominal_power_kw = Column(Float, nullable=True)
    datasheet = Column(JSON, nullable=True)
    specs = Column(JSON, nullable=True)

    scenarios = relationship("ScenarioRecord", back_populates="inverter")


class PanelModel(Base, TimestampMixin):
    __tablename__ = "panels"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model_number = Column(String(255), nullable=True)
    power_w = Column(Float, nullable=True)
    datasheet = Column(JSON, nullable=True)
    specs = Column(JSON, nullable=True)

    scenarios = relationship("ScenarioRecord", back_populates="panel")


class BatteryModel(Base, TimestampMixin):
    __tablename__ = "batteries"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model_number = Column(String(255), nullable=True)
    capacity_kwh = Column(Float, nullable=True)
    datasheet = Column(JSON, nullable=True)
    specs = Column(JSON, nullable=True)

    scenarios = relationship("ScenarioRecord", back_populates="battery")


class ScenarioRecord(Base, TimestampMixin):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)

    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=True)
    panel_id = Column(Integer, ForeignKey("panels.id"), nullable=True)
    battery_id = Column(Integer, ForeignKey("batteries.id"), nullable=True)

    inverter = relationship("InverterModel", back_populates="scenarios")
    panel = relationship("PanelModel", back_populates="scenarios")
    battery = relationship("BatteryModel", back_populates="scenarios")
    runs = relationship("RunResultRecord", back_populates="scenario")


class OptimizationRecord(Base, TimestampMixin):
    __tablename__ = "optimizations"

    id = Column(Integer, primary_key=True)
    label = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="completed")
    request = Column(JSON, nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)

    runs = relationship("RunResultRecord", back_populates="optimization")


class RunResultRecord(Base, TimestampMixin):
    __tablename__ = "run_results"

    id = Column(Integer, primary_key=True)
    result_type = Column(String(50), nullable=False)
    summary = Column(JSON, nullable=False)
    output_dir = Column(Text, nullable=True)

    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=True)
    optimization_id = Column(Integer, ForeignKey("optimizations.id"), nullable=True)

    scenario = relationship("ScenarioRecord", back_populates="runs")
    optimization = relationship("OptimizationRecord", back_populates="runs")
