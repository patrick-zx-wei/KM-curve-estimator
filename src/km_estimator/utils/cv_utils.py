"""OpenCV utilities for image preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator import config
from km_estimator.models import ProcessingError, ProcessingStage

Image: TypeAlias = NDArray[Any]
GrayImage: TypeAlias = NDArray[Any]

MAX_PIXELS_BEFORE_TILING = 16_000_000
_ESPCN_MODEL_DIR = Path(__file__).parent.parent / "models" / "espcn"


@dataclass(frozen=True)
class ImageInfo:
    height: int
    width: int
    channels: int
    is_grayscale: bool
    megapixels: float


@dataclass(frozen=True)
class NoiseEstimate:
    sigma: float
    noise_level: str  # "low", "medium", "high"


@dataclass(frozen=True)
class QualityMetrics:
    variance: float
    sharpness: float
    contrast: float
    noise_level: str
    overall_score: int  # 1-10


# --- I/O ---


def load_image(
    path: str | Path,
    stage: ProcessingStage = ProcessingStage.INPUT,
) -> Image | ProcessingError:
    """Load image, convert to BGR. Returns ProcessingError on failure."""
    path = Path(path)

    if not path.exists():
        return ProcessingError(
            stage=stage,
            error_type="file_not_found",
            recoverable=False,
            message=f"Image file not found: {path}",
            details={"path": str(path)},
        )

    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            return ProcessingError(
                stage=stage,
                error_type="imread_failed",
                recoverable=False,
                message=f"Failed to read image (may be corrupted): {path}",
                details={"path": str(path)},
            )

        # Normalize to BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            c = img.shape[2]
            if c == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif c == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    except PermissionError:
        return ProcessingError(
            stage=stage,
            error_type="permission_denied",
            recoverable=False,
            message=f"Permission denied reading: {path}",
            details={"path": str(path)},
        )
    except Exception as e:
        return ProcessingError(
            stage=stage,
            error_type="io_error",
            recoverable=False,
            message=f"Error reading image: {e}",
            details={"path": str(path), "error": str(e)},
        )


def save_image(
    image: Image,
    path: str | Path,
    stage: ProcessingStage = ProcessingStage.PREPROCESS,
) -> Path | ProcessingError:
    """Save image to disk. Creates parent dirs if needed."""
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(path), image):
            return ProcessingError(
                stage=stage,
                error_type="imwrite_failed",
                recoverable=False,
                message=f"Failed to write image: {path}",
                details={"path": str(path)},
            )
        return path

    except PermissionError:
        return ProcessingError(
            stage=stage,
            error_type="permission_denied",
            recoverable=False,
            message=f"Permission denied writing: {path}",
            details={"path": str(path)},
        )
    except Exception as e:
        return ProcessingError(
            stage=stage,
            error_type="io_error",
            recoverable=False,
            message=f"Error writing image: {e}",
            details={"path": str(path), "error": str(e)},
        )


def get_image_info(image: Image) -> ImageInfo:
    """Extract image metadata."""
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    return ImageInfo(
        height=h,
        width=w,
        channels=c,
        is_grayscale=c == 1,
        megapixels=(h * w) / 1_000_000,
    )


# --- Scaling ---


def calculate_scale_factor(
    current_size: tuple[int, int],
    target_resolution: int = config.TARGET_RESOLUTION,
    min_resolution: int = config.MIN_RESOLUTION,
    max_resolution: int = config.MAX_RESOLUTION,
) -> tuple[float, str] | ProcessingError:
    """
    Returns (scale_factor, method) where method is "espcn", "lanczos_up", "lanczos_down", or "none".
    Errors if image is outside [min_resolution, max_resolution] bounds.
    """
    width, height = current_size
    longest = max(width, height)

    if longest < min_resolution:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="image_too_small",
            recoverable=False,
            message=f"Image too small ({longest}px). Minimum is {min_resolution}px.",
            details={"width": width, "height": height, "min_resolution": min_resolution},
        )

    if longest > max_resolution:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="image_too_large",
            recoverable=True,
            message=f"Image too large ({longest}px). Maximum is {max_resolution}px.",
            details={"width": width, "height": height, "max_resolution": max_resolution},
        )

    if abs(longest - target_resolution) < 50:
        return (1.0, "none")

    if longest > target_resolution:
        return (target_resolution / longest, "lanczos_down")

    scale = target_resolution / longest
    # For small upscales (< 1.5x), use lanczos to avoid overshooting target
    if scale < 1.5:
        return (scale, "lanczos_up")
    # For larger upscales, use ESPCN at nearest factor that reaches target
    elif scale <= 2.0:
        return (2.0, "espcn")
    elif scale <= 3.0:
        return (3.0, "espcn")
    elif scale <= 4.0:
        return (4.0, "espcn")
    else:
        return (scale, "lanczos_up")


def upscale_espcn(
    image: Image,
    scale_factor: int = 2,
    model_path: str | None = None,
) -> Image | ProcessingError:
    """ESPCN super-resolution. Falls back to Lanczos if unavailable."""
    if scale_factor not in (2, 3, 4):
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="invalid_scale_factor",
            recoverable=True,
            message=f"ESPCN only supports scale factors 2, 3, or 4. Got {scale_factor}.",
            details={"scale_factor": scale_factor},
        )

    if not hasattr(cv2, "dnn_superres"):
        return resize_lanczos(image, scale_factor=float(scale_factor))

    if model_path is None:
        model_path = str(_ESPCN_MODEL_DIR / f"ESPCN_x{scale_factor}.pb")

    if not Path(model_path).exists():
        return resize_lanczos(image, scale_factor=float(scale_factor))

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()  # type: ignore[attr-defined]
        sr.readModel(model_path)
        sr.setModel("espcn", scale_factor)
        return sr.upsample(image)
    except cv2.error:
        return resize_lanczos(image, scale_factor=float(scale_factor))
    except Exception as e:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="upscale_failed",
            recoverable=True,
            message=f"ESPCN failed: {e}",
            details={"scale_factor": scale_factor, "error": str(e)},
        )


def resize_lanczos(
    image: Image,
    target_size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
) -> Image:
    """Resize with Lanczos interpolation. Provide target_size OR scale_factor."""
    if image.size == 0:
        return image

    if target_size is not None:
        if target_size[0] <= 0 or target_size[1] <= 0:
            return image
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    if scale_factor is not None and scale_factor > 0:
        h, w = image.shape[:2]
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return image


# --- Tiling ---


def needs_tiling(image: Image) -> bool:
    """Check if image exceeds MAX_PIXELS_BEFORE_TILING."""
    if image.size == 0:
        return False
    h, w = image.shape[:2]
    return h * w > MAX_PIXELS_BEFORE_TILING


def process_in_tiles(
    image: Image,
    operation: Callable[[Image], Image],
    tile_size: int = 2048,
    overlap: int = 64,
) -> Image:
    """Process large image in overlapping tiles with feathered blending."""
    if image.size == 0:
        return image

    h, w = image.shape[:2]

    if h * w <= MAX_PIXELS_BEFORE_TILING:
        return operation(image)

    is_color = len(image.shape) == 3
    channels = image.shape[2] if is_color else 1
    if is_color:
        output = np.zeros((h, w, channels), dtype=np.float32)
    else:
        output = np.zeros((h, w), dtype=np.float32)

    weight_sum = np.zeros((h, w), dtype=np.float32)
    step = tile_size - overlap

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            tile = image[y0:y1, x0:x1]
            processed = operation(tile)

            if processed.shape[:2] != tile.shape[:2]:
                processed = cv2.resize(
                    processed, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LANCZOS4
                )

            th, tw = processed.shape[:2]
            weight = np.ones((th, tw), dtype=np.float32)

            # Feather edges
            if y0 > 0:
                top_n = min(overlap, th)
                top_ramp = np.arange(top_n, dtype=np.float32) / max(1, overlap)
                weight[:top_n, :] *= top_ramp[:, None]
            if x0 > 0:
                left_n = min(overlap, tw)
                left_ramp = np.arange(left_n, dtype=np.float32) / max(1, overlap)
                weight[:, :left_n] *= left_ramp[None, :]
            if y1 < h:
                bottom_n = min(overlap, th)
                bottom_ramp = np.arange(bottom_n, dtype=np.float32) / max(1, overlap)
                weight[th - bottom_n:th, :] *= bottom_ramp[::-1, None]
            if x1 < w:
                right_n = min(overlap, tw)
                right_ramp = np.arange(right_n, dtype=np.float32) / max(1, overlap)
                weight[:, tw - right_n:tw] *= right_ramp[None, ::-1]

            processed_f32 = processed.astype(np.float32)
            if is_color:
                output[y0:y1, x0:x1, :] += processed_f32 * weight[:, :, None]
            else:
                output[y0:y1, x0:x1] += processed_f32 * weight

            weight_sum[y0:y1, x0:x1] += weight

    weight_sum = np.maximum(weight_sum, 1e-10)

    if is_color:
        output = output / weight_sum[:, :, None]
    else:
        output = output / weight_sum

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        output = np.clip(np.rint(output), info.min, info.max)

    return output.astype(image.dtype, copy=False)


def downsample_to_limit(image: Image, max_pixels: int = MAX_PIXELS_BEFORE_TILING) -> Image:
    """Downsample if image exceeds pixel limit."""
    if image.size == 0:
        return image

    h, w = image.shape[:2]
    if h * w <= max_pixels:
        return image

    scale = np.sqrt(max_pixels / (h * w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# --- Denoising ---


def estimate_noise(image: Image) -> NoiseEstimate:
    """Estimate noise via Laplacian MAD."""
    if image.size == 0:
        return NoiseEstimate(sigma=0.0, noise_level="low")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # MAD-based estimation: sigma = MAD / 0.6745
    median_val = np.median(np.abs(laplacian))
    sigma = float(median_val / 0.6745) if median_val > 0 else 0.0

    if sigma < 5:
        level = "low"
    elif sigma < 15:
        level = "medium"
    else:
        level = "high"

    return NoiseEstimate(sigma=sigma, noise_level=level)


def denoise_nlmeans(
    image: Image,
    h: float | None = None,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> Image:
    """NL-means denoising. Auto-calculates h if not provided."""
    if image.size == 0:
        return image

    # Ensure odd window sizes
    template_window_size = max(3, template_window_size) | 1
    search_window_size = max(3, search_window_size) | 1

    if h is None:
        noise = estimate_noise(image)
        h = {"low": 3.0, "medium": 10.0, "high": 15.0}[noise.noise_level]

    h_int = max(1, int(h))

    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            src=image, h=h_int, hColor=h_int,
            templateWindowSize=template_window_size, searchWindowSize=search_window_size,
        )
    else:
        return cv2.fastNlMeansDenoising(
            src=image, h=h_int,
            templateWindowSize=template_window_size, searchWindowSize=search_window_size,
        )


# --- Sharpening ---


def sharpen_unsharp_mask(
    image: Image,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0,
) -> Image:
    """Unsharp mask sharpening."""
    if image.size == 0:
        return image

    kernel_size = max(1, kernel_size) | 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    if threshold == 0:
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    diff = cv2.subtract(image, blurred)
    mask = np.abs(diff) > threshold
    sharpened = image.copy()
    sharpened[mask] = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)[mask]
    return sharpened


# --- Quality ---


def calculate_variance(image: Image) -> float:
    """Image variance (0 = solid color)."""
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(np.var(gray))


def calculate_sharpness(image: Image) -> float:
    """Laplacian variance (higher = sharper)."""
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))


def calculate_contrast(image: Image) -> float:
    """RMS contrast (0-1)."""
    if image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    max_val = np.iinfo(gray.dtype).max if np.issubdtype(gray.dtype, np.integer) else 1.0
    normalized = gray.astype(np.float64) / max_val
    return float(np.sqrt(np.mean((normalized - np.mean(normalized)) ** 2)))


def assess_quality(
    image: Image,
    min_variance: float = config.MIN_IMAGE_VARIANCE,
) -> QualityMetrics | ProcessingError:
    """Overall quality score 1-10. Errors if image is blank."""
    variance = calculate_variance(image)

    if variance < min_variance:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="blank_image",
            recoverable=False,
            message=f"Image appears blank (variance={variance:.2f})",
            details={"variance": variance, "min_variance": min_variance},
        )

    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    noise = estimate_noise(image)

    # Weighted score: 40% sharpness, 30% contrast, 30% noise
    s_score = min(10, max(1, int(sharpness / 100)))
    c_score = min(10, max(1, int(contrast * 30)))
    n_score = {"low": 10, "medium": 6, "high": 3}[noise.noise_level]

    overall = min(10, max(1, int(0.4 * s_score + 0.3 * c_score + 0.3 * n_score)))

    return QualityMetrics(
        variance=variance,
        sharpness=sharpness,
        contrast=contrast,
        noise_level=noise.noise_level,
        overall_score=overall,
    )


# --- Helpers ---


def to_grayscale(image: Image) -> GrayImage:
    """Convert to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    if len(image.shape) == 3:
        c = image.shape[2]
        if c == 1:
            return image[:, :, 0]
        if c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ensure_bgr(image: Image) -> Image:
    """Ensure 3-channel BGR format."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3:
        c = image.shape[2]
        if c == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        if c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
