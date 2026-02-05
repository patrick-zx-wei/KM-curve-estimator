"""OpenAI GPT-5 Mini client for primary OCR extraction."""

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIError
from pydantic import BaseModel, ValidationError

from km_estimator import config
from km_estimator.models import PlotMetadata, ProcessingError, ProcessingStage, RawOCRTokens

T = TypeVar("T", bound=BaseModel)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class GPTResult(Generic[T]):
    """Result from GPT-5 Mini extraction with confidence and token tracking."""

    result: T | None
    error: ProcessingError | None
    confidence: float
    input_tokens: int = 0
    output_tokens: int = 0


@lru_cache(maxsize=4)
def _get_model(output_type: type, timeout: int, retries: int) -> Runnable:
    """Get cached OpenAI model instance with structured output."""
    base = ChatOpenAI(
        model=config.GPT5_MINI_MODEL,
        max_retries=retries,
        timeout=timeout,
    )
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


def _invoke_gpt(
    path: str,
    prompt: str,
    output_type: type[T],
    stage: ProcessingStage,
    timeout: int = config.API_TIMEOUT_SECONDS,
    max_retries: int = config.API_MAX_RETRIES,
) -> GPTResult[T]:
    """Invoke GPT-5 Mini with structured output (LangChain pattern)."""
    img = _read_image(path)
    if isinstance(img, ProcessingError):
        return GPTResult(None, img, 0.0)

    data, mime = img
    b64 = base64.b64encode(data).decode()

    try:
        llm = _get_model(output_type, timeout, max_retries)
        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        )
        result = llm.invoke([msg])
        confidence = getattr(result, "extraction_confidence", None) or 0.5

        # LangChain doesn't expose token usage directly through structured output
        return GPTResult(
            result=result,
            error=None,
            confidence=confidence,
            input_tokens=0,
            output_tokens=0,
        )
    except ValidationError as e:
        return GPTResult(
            None,
            ProcessingError(
                stage=stage,
                error_type="validation_error",
                recoverable=True,
                message=str(e),
            ),
            0.0,
        )
    except APIError as e:
        return GPTResult(
            None,
            ProcessingError(
                stage=stage,
                error_type="openai_api_error",
                recoverable=True,
                message=str(e),
            ),
            0.0,
        )


def extract_ocr_gpt(path: str, **kw) -> GPTResult[RawOCRTokens]:
    """Extract OCR tokens using GPT-5 Mini with model-specific prompt."""
    return _invoke_gpt(
        path, config.OCR_PROMPT_GPT, RawOCRTokens, ProcessingStage.MMPU, **kw
    )


def extract_metadata_gpt(path: str, ocr: RawOCRTokens, **kw) -> GPTResult[PlotMetadata]:
    """Extract plot metadata using GPT-5 Mini."""
    prompt = config.ANALYSIS_PROMPT_TEMPLATE.format(ocr_json=ocr.model_dump_json(indent=2))
    return _invoke_gpt(path, prompt, PlotMetadata, ProcessingStage.MMPU, **kw)
