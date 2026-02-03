"""Gemini API client for OCR and metadata extraction."""

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Generic, TypeVar

from google.api_core.exceptions import GoogleAPIError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from km_estimator import config
from km_estimator.models import (
    AxisConfig,
    PlotMetadata,
    ProcessingError,
    ProcessingStage,
    RawOCRTokens,
    ValidationResult,
)

T = TypeVar("T", bound=BaseModel)

MODELS = {"pro": config.GEMINI_PRO_MODEL, "flash": config.GEMINI_FLASH_MODEL}
MIME_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}
FLOAT_TOLERANCE = 0.01

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini")


@dataclass
class DualExtractionResult(Generic[T]):
    """Result from dual-model extraction."""
    result: T | None
    error: ProcessingError | None
    converged: bool
    attempts: int
    warnings: list[str] = field(default_factory=list)


# --- Model Setup ---


@lru_cache(maxsize=8)
def _get_model(model: str, output_type: type, timeout: int, retries: int) -> Runnable:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    base = ChatGoogleGenerativeAI(model=MODELS[model], max_retries=retries, timeout=timeout)
    return base.with_structured_output(output_type).with_retry(
        stop_after_attempt=retries, wait_exponential_jitter=True
    )


def _read_image(path: str) -> tuple[bytes, str] | ProcessingError:
    p = Path(path)
    try:
        return p.read_bytes(), MIME_TYPES.get(p.suffix.lower(), "image/png")
    except FileNotFoundError:
        return ProcessingError(
            stage=ProcessingStage.MMPU, error_type="file_not_found",
            recoverable=False, message=f"Image not found: {path}"
        )
    except PermissionError:
        return ProcessingError(
            stage=ProcessingStage.MMPU, error_type="permission_denied",
            recoverable=False, message=f"Permission denied: {path}"
        )


def _invoke(
    path: str, prompt: str, output_type: type[T], stage: ProcessingStage,
    model: str = "pro", timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> T | ProcessingError:
    """Generic LLM invoke with image."""
    if model not in MODELS:
        return ProcessingError(stage=stage, error_type="invalid_model",
                               recoverable=False, message=f"Unknown model: {model}")
    img = _read_image(path)
    if isinstance(img, ProcessingError):
        return ProcessingError(stage=stage, error_type=img.error_type,
                               recoverable=img.recoverable, message=img.message,
                               details={"path": path})
    data, mime = img
    try:
        llm = _get_model(model, output_type, timeout, max_retries)
        b64 = base64.b64encode(data).decode()
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            {"type": "text", "text": prompt},
        ])
        return llm.invoke([msg])
    except ValidationError as e:
        return ProcessingError(stage=stage, error_type="validation_error",
                               recoverable=True, message=str(e),
                               details={"errors": e.errors()})
    except GoogleAPIError as e:
        return ProcessingError(stage=stage, error_type=type(e).__name__,
                               recoverable=getattr(e, "retryable", False), message=str(e),
                               details={"model": model, "path": path})


# --- Single-Model Extraction ---


def validate_image(path: str, **kw) -> ValidationResult | ProcessingError:
    return _invoke(path, config.INPUT_GUARD_PROMPT, ValidationResult,
                   ProcessingStage.INPUT_GUARD, **kw)


def extract_ocr(path: str, **kw) -> RawOCRTokens | ProcessingError:
    return _invoke(path, config.OCR_PROMPT, RawOCRTokens, ProcessingStage.MMPU, **kw)


def extract_metadata(path: str, ocr: RawOCRTokens, **kw) -> PlotMetadata | ProcessingError:
    prompt = config.ANALYSIS_PROMPT_TEMPLATE.format(ocr_json=ocr.model_dump_json(indent=2))
    return _invoke(path, prompt, PlotMetadata, ProcessingStage.MMPU, **kw)


# --- Convergence ---


def _ocr_converged(a: RawOCRTokens, b: RawOCRTokens) -> bool:
    return (a.axis_labels == b.axis_labels and a.x_tick_labels == b.x_tick_labels
            and a.y_tick_labels == b.y_tick_labels)


def _axis_match(a: AxisConfig, b: AxisConfig) -> bool:
    if len(a.tick_values) != len(b.tick_values):
        return False
    return (all(abs(x - y) < FLOAT_TOLERANCE for x, y in zip(a.tick_values, b.tick_values))
            and abs(a.start - b.start) < FLOAT_TOLERANCE
            and abs(a.end - b.end) < FLOAT_TOLERANCE)


def _metadata_converged(a: PlotMetadata, b: PlotMetadata) -> bool:
    return _axis_match(a.x_axis, b.x_axis) and _axis_match(a.y_axis, b.y_axis)


# --- Dual-Model Extraction ---


def _extract_dual(
    extractor: Callable[..., T | ProcessingError],
    converged: Callable[[T, T], bool],
    max_attempts: int,
    single_mode: bool,
    **kw,
) -> DualExtractionResult[T]:
    """Generic dual-model extraction with convergence loop."""
    if single_mode:
        r = extractor(model="pro", **kw)
        err = isinstance(r, ProcessingError)
        return DualExtractionResult(None if err else r, r if err else None, not err, 1)

    warnings: list[str] = []
    last_pro = last_flash = None

    for attempt in range(1, max_attempts + 1):
        pro_f = _executor.submit(extractor, model="pro", **kw)
        flash_f = _executor.submit(extractor, model="flash", **kw)
        pro, flash = pro_f.result(), flash_f.result()
        last_pro, last_flash = pro, flash

        pro_ok, flash_ok = not isinstance(pro, ProcessingError), not isinstance(flash, ProcessingError)

        if not pro_ok and not flash_ok:
            warnings.append(f"Attempt {attempt}: both failed")
            continue
        if not pro_ok:
            warnings.append(f"Pro failed: {pro.message}")
            return DualExtractionResult(flash, None, False, attempt, warnings)
        if not flash_ok:
            warnings.append(f"Flash failed: {flash.message}")
            return DualExtractionResult(pro, None, False, attempt, warnings)
        if converged(pro, flash):
            return DualExtractionResult(pro, None, True, attempt, warnings)
        warnings.append(f"Attempt {attempt}: diverged")

    # Fallback to Pro
    if last_pro and not isinstance(last_pro, ProcessingError):
        warnings.append("Max attempts, using Pro")
        return DualExtractionResult(last_pro, None, False, max_attempts, warnings)

    err = last_pro if isinstance(last_pro, ProcessingError) else ProcessingError(
        stage=ProcessingStage.MMPU, error_type="all_failed",
        recoverable=False, message="All attempts failed"
    )
    return DualExtractionResult(None, err, False, max_attempts, warnings)


def extract_ocr_dual(
    path: str,
    max_attempts: int = config.MAX_MMPU_RETRIES,
    timeout: int = config.API_TIMEOUT_SECONDS,
    single_model_mode: bool = False,
) -> DualExtractionResult[RawOCRTokens]:
    return _extract_dual(
        lambda **kw: extract_ocr(path, timeout=timeout, **kw),
        _ocr_converged, max_attempts, single_model_mode
    )


def extract_metadata_dual(
    path: str,
    ocr: RawOCRTokens,
    max_attempts: int = config.MAX_MMPU_RETRIES,
    timeout: int = config.API_TIMEOUT_SECONDS,
    single_model_mode: bool = False,
) -> DualExtractionResult[PlotMetadata]:
    return _extract_dual(
        lambda **kw: extract_metadata(path, ocr, timeout=timeout, **kw),
        _metadata_converged, max_attempts, single_model_mode
    )
