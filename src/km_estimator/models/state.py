from pydantic import BaseModel

from .ipd_output import IPDOutput, ProcessingError
from .plot_metadata import PlotMetadata, ValidationResult


class PipelineConfig(BaseModel):
    use_pro: bool = True
    use_flash: bool = True
    single_model_mode: bool = False

    max_input_guard_retries: int = 3
    max_mmpu_retries: int = 3
    max_validation_retries: int = 3

    convergence_threshold: float = 0.9
    validation_mae_threshold: float = 0.02

    target_resolution: int = 2000
    min_resolution: int = 200
    max_resolution: int = 4000

    api_timeout_seconds: int = 30
    api_max_retries: int = 3

    min_image_variance: float = 100.0
    min_confidence_output: float = 0.3


class PipelineState(BaseModel):
    image_path: str
    config: PipelineConfig = PipelineConfig()

    preprocessed_image_path: str | None = None
    quality_score: int | None = None

    validation_result: ValidationResult | None = None
    input_guard_retries: int = 0

    plot_metadata: PlotMetadata | None = None
    mmpu_retries: int = 0

    digitized_curves: dict[str, list[tuple[float, float]]] | None = None
    censoring_marks: dict[str, list[float]] | None = None

    output: IPDOutput | None = None
    validation_retries: int = 0

    errors: list[ProcessingError] = []
