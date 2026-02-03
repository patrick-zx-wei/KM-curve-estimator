import base64
from functools import lru_cache
from pathlib import Path
from typing import Literal

from google.api_core.exceptions import GoogleAPIError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from km_estimator import config
from km_estimator.models import (
    PlotMetadata,
    ProcessingError,
    ProcessingStage,
    RawOCRTokens,
)

MODELS = {
    "pro": config.GEMINI_PRO_MODEL,
    "flash": config.GEMINI_FLASH_MODEL,
}

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@lru_cache(maxsize=4)
def _get_base_model(model: str, timeout: int, max_retries: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODELS[model],
        max_retries=max_retries,
        timeout=timeout,
    )


def _get_ocr_model(model: str, timeout: int, max_retries: int) -> Runnable:
    base = _get_base_model(model, timeout, max_retries)
    return base.with_structured_output(RawOCRTokens).with_retry(
        stop_after_attempt=max_retries,
        wait_exponential_jitter=True,
    )


def _get_metadata_model(model: str, timeout: int, max_retries: int) -> Runnable:
    base = _get_base_model(model, timeout, max_retries)
    return base.with_structured_output(PlotMetadata).with_retry(
        stop_after_attempt=max_retries,
        wait_exponential_jitter=True,
    )


def _read_image(image_path: str) -> tuple[bytes, str] | ProcessingError:
    path = Path(image_path)
    try:
        image_bytes = path.read_bytes()
    except FileNotFoundError:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="file_not_found",
            recoverable=False,
            message=f"Image file not found: {image_path}",
            details={"image_path": image_path},
        )
    except PermissionError:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="permission_denied",
            recoverable=False,
            message=f"Permission denied reading: {image_path}",
            details={"image_path": image_path},
        )
    mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
    return image_bytes, mime


def extract_ocr(
    image_path: str,
    model: Literal["pro", "flash"] = "pro",
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> RawOCRTokens | ProcessingError:
    if model not in MODELS:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="invalid_model",
            recoverable=False,
            message=f"Unknown model: {model}. Must be 'pro' or 'flash'",
            details={"model": model},
        )

    image_result = _read_image(image_path)
    if isinstance(image_result, ProcessingError):
        return image_result
    image_bytes, mime = image_result

    try:
        llm = _get_ocr_model(model, timeout, max_retries)
        b64 = base64.b64encode(image_bytes).decode()
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            {"type": "text", "text": config.OCR_PROMPT},
        ])
        return llm.invoke([msg])
    except ValidationError as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="validation_error",
            recoverable=True,
            message=str(e),
            details={"errors": e.errors()},
        )
    except GoogleAPIError as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type=type(e).__name__,
            recoverable=getattr(e, "retryable", False),
            message=str(e),
            details={"model": model, "image_path": image_path},
        )


def extract_metadata(
    image_path: str,
    ocr_tokens: RawOCRTokens,
    model: Literal["pro", "flash"] = "pro",
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> PlotMetadata | ProcessingError:
    if model not in MODELS:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="invalid_model",
            recoverable=False,
            message=f"Unknown model: {model}. Must be 'pro' or 'flash'",
            details={"model": model},
        )

    image_result = _read_image(image_path)
    if isinstance(image_result, ProcessingError):
        return image_result
    image_bytes, mime = image_result

    try:
        llm = _get_metadata_model(model, timeout, max_retries)
        b64 = base64.b64encode(image_bytes).decode()
        prompt = config.ANALYSIS_PROMPT_TEMPLATE.format(
            ocr_json=ocr_tokens.model_dump_json(indent=2)
        )
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            {"type": "text", "text": prompt},
        ])
        return llm.invoke([msg])
    except ValidationError as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="validation_error",
            recoverable=True,
            message=str(e),
            details={"errors": e.errors()},
        )
    except GoogleAPIError as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type=type(e).__name__,
            recoverable=getattr(e, "retryable", False),
            message=str(e),
            details={"model": model, "image_path": image_path},
        )
