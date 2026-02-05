from enum import Enum
from typing import Any

from pydantic import BaseModel

from .plot_metadata import PlotMetadata


class ProcessingStage(str, Enum):
    INPUT = "input"
    PREPROCESS = "preprocess"
    INPUT_GUARD = "input_guard"
    MMPU = "mmpu"
    DIGITIZE = "digitize"
    RECONSTRUCT = "reconstruct"
    VALIDATE = "validate"


class ProcessingError(BaseModel):
    stage: ProcessingStage
    error_type: str
    recoverable: bool
    message: str
    details: dict[str, Any] = {}


class ReconstructionMode(str, Enum):
    FULL = "full"
    ESTIMATED = "estimated"


class PatientRecord(BaseModel):
    time: float
    event: bool


class CurveIPD(BaseModel):
    group_name: str
    patients: list[PatientRecord]
    censoring_times: list[float] = []
    validation_mae: float | None = None
    validation_dtw: float | None = None
    validation_rmse: float | None = None
    validation_max_error: float | None = None
    validation_frechet: float | None = None
    digitization_confidence: float = 1.0


class IPDOutput(BaseModel):
    metadata: PlotMetadata
    curves: list[CurveIPD]
    reconstruction_mode: ReconstructionMode
    warnings: list[str] = []
    confidence_score: float = 1.0
    errors: list[ProcessingError] = []
    partial_results: bool = False
