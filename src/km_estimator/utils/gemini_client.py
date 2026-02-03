import base64
import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from km_estimator.models import (
    PlotMetadata,
    ProcessingError,
    ProcessingStage,
    RawOCRTokens,
)

MODELS = {
    "pro": "gemini-3-pro-preview",
    "flash": "gemini-3-flash-preview",
}

OCR_PROMPT = """Extract all text from this Kaplan-Meier survival curve image.

Return JSON with these fields:
- x_tick_labels: list of x-axis tick labels (e.g., ["0", "12", "24", "36"])
- y_tick_labels: list of y-axis tick labels (e.g., ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
- axis_labels: list of axis labels (e.g., ["Time (months)", "Survival probability"])
- legend_labels: list of legend/group names (e.g., ["Treatment", "Control"])
- risk_table_text: 2D array of risk table values if present, null otherwise
- title: plot title if present, null otherwise
- annotations: list of other text (p-values, hazard ratios, etc.)

Return only valid JSON, no markdown."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze this Kaplan-Meier survival curve image.

Previously extracted text from the image:
{ocr_json}

Using both the image and extracted text, return JSON with:
- x_axis: {{label, start, end, tick_interval, tick_values, scale}}
- y_axis: {{label, start, end, tick_interval, tick_values, scale}}
- curves: [{{name, color_description, line_style}}]
- risk_table: {{time_points, groups: [{{name, counts}}]}} or null
- title: string or null
- annotations: list of strings

Return only valid JSON, no markdown."""

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


def _call_vision(
    image_path: str,
    prompt: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> str | ProcessingError:
    try:
        llm = _get_model(model, timeout, max_retries)
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
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
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_ocr(
    image_path: str,
    model: Literal["pro", "flash"] = "pro",
    timeout: int = 30,
    max_retries: int = 3,
) -> RawOCRTokens | ProcessingError:
    result = _call_vision(image_path, OCR_PROMPT, model, timeout, max_retries)
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
    timeout: int = 30,
    max_retries: int = 3,
) -> PlotMetadata | ProcessingError:
    ocr_json = ocr_tokens.model_dump_json(indent=2)
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(ocr_json=ocr_json)

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
