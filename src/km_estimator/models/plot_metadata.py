from typing import Literal

from pydantic import BaseModel


class AxisConfig(BaseModel):
    label: str | None
    start: float
    end: float
    tick_interval: float | None
    tick_values: list[float]
    scale: Literal["linear", "log"] = "linear"


class CurveInfo(BaseModel):
    name: str
    color_description: str | None
    line_style: str | None


class RiskGroup(BaseModel):
    name: str
    counts: list[int]


class RiskTable(BaseModel):
    time_points: list[float]
    groups: list[RiskGroup]


class RawOCRTokens(BaseModel):
    x_tick_labels: list[str]
    y_tick_labels: list[str]
    axis_labels: list[str]
    legend_labels: list[str]
    risk_table_text: list[list[str]] | None
    title: str | None
    annotations: list[str] = []


class PlotMetadata(BaseModel):
    x_axis: AxisConfig
    y_axis: AxisConfig
    curves: list[CurveInfo]
    risk_table: RiskTable | None
    title: str | None = None
    annotations: list[str] = []


class ValidationResult(BaseModel):
    valid: bool
    axes_present: bool
    curves_present: bool
    ticks_readable: bool
    legend_present: bool
    risk_table_present: bool
    feedback: str
