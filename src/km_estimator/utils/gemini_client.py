"""Gemini API client for verification extraction."""

import base64
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from google.api_core.exceptions import GoogleAPIError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from km_estimator import config
from km_estimator.models import (
    PlotMetadata,
    ProcessingError,
    ProcessingStage,
    RawOCRTokens,
    ValidationResult,
)

T = TypeVar("T", bound=BaseModel)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


# --- Model Setup ---


@lru_cache(maxsize=4)
def _get_model(model: str, output_type: type, timeout: int, retries: int) -> Runnable:
    """Get cached Gemini model instance."""
    base = ChatGoogleGenerativeAI(model=model, max_retries=retries, timeout=timeout)
    return base.with_structured_output(output_type).with_retry(
        stop_after_attempt=retries, wait_exponential_jitter=True
    )


def _read_image(path: str) -> tuple[bytes, str] | ProcessingError:
    """Read image file and determine MIME type."""
    p = Path(path)
    try:
        return p.read_bytes(), MIME_TYPES.get(p.suffix.lower(), "image/png")
    except FileNotFoundError:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="file_not_found",
            recoverable=False,
            message=f"Image not found: {path}",
        )
    except PermissionError:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="permission_denied",
            recoverable=False,
            message=f"Permission denied: {path}",
        )
    except Exception as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="io_error",
            recoverable=False,
            message=f"Failed reading image: {e}",
            details={"path": path, "error_type": type(e).__name__},
        )


def _invoke(
    path: str,
    prompt: str,
    output_type: type[T],
    stage: ProcessingStage,
    model: str = config.GEMINI_FLASH_MODEL,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> T | ProcessingError:
    """Generic LLM invoke with image."""
    img = _read_image(path)
    if isinstance(img, ProcessingError):
        return ProcessingError(
            stage=stage,
            error_type=img.error_type,
            recoverable=img.recoverable,
            message=img.message,
            details={"path": path},
        )
    data, mime = img
    try:
        llm = _get_model(model, output_type, timeout, max_retries)
        b64 = base64.b64encode(data).decode()
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
                {"type": "text", "text": prompt},
            ]
        )
        return llm.invoke([msg])
    except ValidationError as e:
        return ProcessingError(
            stage=stage,
            error_type="validation_error",
            recoverable=True,
            message=str(e),
            details={"errors": e.errors()},
        )
    except GoogleAPIError as e:
        return ProcessingError(
            stage=stage,
            error_type=type(e).__name__,
            recoverable=getattr(e, "retryable", False),
            message=str(e),
            details={"model": model, "path": path},
        )
    except Exception as e:
        return ProcessingError(
            stage=stage,
            error_type=type(e).__name__,
            recoverable=True,
            message=f"Gemini invocation failed: {e}",
            details={"model": model, "path": path},
        )


# --- Single-Model Extraction ---


def validate_image(path: str, **kw) -> ValidationResult | ProcessingError:
    """Validate image is a valid KM curve."""
    return _invoke(
        path, config.INPUT_GUARD_PROMPT, ValidationResult, ProcessingStage.INPUT_GUARD, **kw
    )


def extract_ocr(path: str, **kw) -> RawOCRTokens | ProcessingError:
    """Extract OCR tokens using Gemini with model-specific prompt."""
    return _invoke(path, config.OCR_PROMPT_GEMINI, RawOCRTokens, ProcessingStage.MMPU, **kw)


def extract_metadata(path: str, ocr: RawOCRTokens, **kw) -> PlotMetadata | ProcessingError:
    """Extract plot metadata using Gemini."""
    prompt = config.ANALYSIS_PROMPT_TEMPLATE.format(ocr_json=ocr.model_dump_json(indent=2))
    return _invoke(path, prompt, PlotMetadata, ProcessingStage.MMPU, **kw)


# --- Async Single-Model Extraction ---


async def _ainvoke(
    path: str,
    prompt: str,
    output_type: type[T],
    stage: ProcessingStage,
    model: str = config.GEMINI_FLASH_MODEL,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> T | ProcessingError:
    """Async generic LLM invoke with image."""
    img = _read_image(path)
    if isinstance(img, ProcessingError):
        return ProcessingError(
            stage=stage,
            error_type=img.error_type,
            recoverable=img.recoverable,
            message=img.message,
            details={"path": path},
        )
    data, mime = img
    try:
        llm = _get_model(model, output_type, timeout, max_retries)
        b64 = base64.b64encode(data).decode()
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
                {"type": "text", "text": prompt},
            ]
        )
        return await llm.ainvoke([msg])
    except ValidationError as e:
        return ProcessingError(
            stage=stage,
            error_type="validation_error",
            recoverable=True,
            message=str(e),
            details={"errors": e.errors()},
        )
    except GoogleAPIError as e:
        return ProcessingError(
            stage=stage,
            error_type=type(e).__name__,
            recoverable=getattr(e, "retryable", False),
            message=str(e),
            details={"model": model, "path": path},
        )
    except Exception as e:
        return ProcessingError(
            stage=stage,
            error_type=type(e).__name__,
            recoverable=True,
            message=f"Gemini invocation failed: {e}",
            details={"model": model, "path": path},
        )


async def validate_image_async(path: str, **kw) -> ValidationResult | ProcessingError:
    """Async: Validate image is a valid KM curve."""
    return await _ainvoke(
        path, config.INPUT_GUARD_PROMPT, ValidationResult, ProcessingStage.INPUT_GUARD, **kw
    )


async def extract_ocr_async(path: str, **kw) -> RawOCRTokens | ProcessingError:
    """Async: Extract OCR tokens using Gemini with model-specific prompt."""
    return await _ainvoke(path, config.OCR_PROMPT_GEMINI, RawOCRTokens, ProcessingStage.MMPU, **kw)


async def extract_metadata_async(
    path: str, ocr: RawOCRTokens, **kw
) -> PlotMetadata | ProcessingError:
    """Async: Extract plot metadata using Gemini."""
    prompt = config.ANALYSIS_PROMPT_TEMPLATE.format(ocr_json=ocr.model_dump_json(indent=2))
    return await _ainvoke(path, prompt, PlotMetadata, ProcessingStage.MMPU, **kw)
