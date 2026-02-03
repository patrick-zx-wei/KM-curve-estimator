import base64
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.utils.json import parse_json_markdown
from langchain_google_genai import ChatGoogleGenerativeAI

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

_cache: dict[tuple, ChatGoogleGenerativeAI] = {}


def _get_model(model: str, timeout: int, max_retries: int):
    key = (model, timeout, max_retries)
    if key not in _cache:
        base = ChatGoogleGenerativeAI(
            model=MODELS[model],
            max_retries=max_retries,
            timeout=timeout,
        )
        _cache[key] = base.with_retry(
            stop_after_attempt=max_retries,
            wait_exponential_jitter=True,
        )
    return _cache[key]


MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _call_vision(
    image_path: str,
    prompt: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> str | ProcessingError:
    if model not in MODELS:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="invalid_model",
            recoverable=False,
            message=f"Unknown model: {model}. Must be 'pro' or 'flash'",
            details={"model": model},
        )

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

    try:
        llm = _get_model(model, timeout, max_retries)
        b64 = base64.b64encode(image_bytes).decode()
        mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            {"type": "text", "text": prompt},
        ])
        return llm.invoke([msg]).content
    except Exception as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type=type(e).__name__,
            recoverable=False,
            message=str(e),
            details={"model": model, "image_path": image_path},
        )


def _parse_json(text: str) -> dict | None:
    try:
        return parse_json_markdown(text)
    except Exception:
        return None


def extract_ocr(
    image_path: str,
    model: Literal["pro", "flash"] = "pro",
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> RawOCRTokens | ProcessingError:
    result = _call_vision(image_path, config.OCR_PROMPT, model, timeout, max_retries)
    if isinstance(result, ProcessingError):
        return result

    data = _parse_json(result)
    if data is None:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="json_parse_error",
            recoverable=True,
            message="Failed to parse OCR response as JSON",
            details={"raw_response": result[:500]},
        )

    try:
        return RawOCRTokens(**data)
    except Exception as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="validation_error",
            recoverable=True,
            message=str(e),
            details={"raw_response": result[:500]},
        )


def extract_metadata(
    image_path: str,
    ocr_tokens: RawOCRTokens,
    model: Literal["pro", "flash"] = "pro",
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> PlotMetadata | ProcessingError:
    ocr_json = ocr_tokens.model_dump_json(indent=2)
    prompt = config.ANALYSIS_PROMPT_TEMPLATE.format(ocr_json=ocr_json)

    result = _call_vision(image_path, prompt, model, timeout, max_retries)
    if isinstance(result, ProcessingError):
        return result

    data = _parse_json(result)
    if data is None:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="json_parse_error",
            recoverable=True,
            message="Failed to parse analysis response as JSON",
            details={"raw_response": result[:500]},
        )

    try:
        return PlotMetadata(**data)
    except Exception as e:
        return ProcessingError(
            stage=ProcessingStage.MMPU,
            error_type="validation_error",
            recoverable=True,
            message=str(e),
            details={"raw_response": result[:500]},
        )
