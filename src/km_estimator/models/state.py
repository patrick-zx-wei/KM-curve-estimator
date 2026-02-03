from pydantic import BaseModel

from km_estimator import config

from .ipd_output import IPDOutput, ProcessingError
from .plot_metadata import PlotMetadata, RawOCRTokens, ValidationResult


class PipelineConfig(BaseModel):
    use_pro: bool = True
    use_flash: bool = True
    single_model_mode: bool = False

    max_input_guard_retries: int = config.MAX_INPUT_GUARD_RETRIES
    max_mmpu_retries: int = config.MAX_MMPU_RETRIES
    max_validation_retries: int = config.MAX_VALIDATION_RETRIES

    convergence_threshold: float = config.CONVERGENCE_THRESHOLD
    validation_mae_threshold: float = config.VALIDATION_MAE_THRESHOLD

    target_resolution: int = config.TARGET_RESOLUTION
    min_resolution: int = config.MIN_RESOLUTION
    max_resolution: int = config.MAX_RESOLUTION

    api_timeout_seconds: int = config.API_TIMEOUT_SECONDS
    api_max_retries: int = config.API_MAX_RETRIES

    min_image_variance: float = config.MIN_IMAGE_VARIANCE
    min_confidence_output: float = config.MIN_CONFIDENCE_OUTPUT


class PipelineState(BaseModel):
    image_path: str
    config: PipelineConfig = PipelineConfig()

    preprocessed_image_path: str | None = None
    quality_score: int | None = None

    validation_result: ValidationResult | None = None
    input_guard_retries: int = 0

    ocr_tokens: RawOCRTokens | None = None
    plot_metadata: PlotMetadata | None = None
    mmpu_retries: int = 0

    digitized_curves: dict[str, list[tuple[float, float]]] | None = None
    censoring_marks: dict[str, list[float]] | None = None

    output: IPDOutput | None = None
    validation_retries: int = 0

    errors: list[ProcessingError] = []
