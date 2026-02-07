from typing import Literal

from pydantic import BaseModel, field_validator


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
    extraction_confidence: float | None = None


class PlotMetadata(BaseModel):
    x_axis: AxisConfig
    y_axis: AxisConfig
    curves: list[CurveInfo]
    risk_table: RiskTable | None
    curve_direction: Literal["downward", "upward"] = "downward"
    title: str | None = None
    annotations: list[str] = []

    @field_validator("curve_direction", mode="before")
    @classmethod
    def _normalize_curve_direction(cls, value: object) -> str:
        """Normalize permissive LLM direction strings into the supported enum."""
        if value is None:
            return "downward"
        text = str(value).strip().lower().replace("-", " ").replace("_", " ")
        if text in {"upward", "up", "increasing", "increase", "ascending"}:
            return "upward"
        if text in {"downward", "down", "decreasing", "decrease", "descending"}:
            return "downward"
        if "incidence" in text or "increas" in text or "event probability" in text:
            return "upward"
        if "survival" in text or "decreas" in text or "kaplan" in text:
            return "downward"
        return "downward"


class ValidationResult(BaseModel):
    valid: bool
    axes_present: bool
    curves_present: bool
    ticks_readable: bool
    legend_present: bool
    risk_table_present: bool
    feedback: str
