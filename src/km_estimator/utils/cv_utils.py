"""
OpenCV utility functions for KM-curve preprocessing pipeline.

This module provides reusable image processing functions for:
- Image I/O with validation
- Scaling (ESPCN upscale, Lanczos resize)
- Denoising (NL-means with adaptive strength)
- Sharpening (unsharp mask)
- Quality assessment (variance, sharpness, contrast)

All functions follow the Result | ProcessingError pattern for error handling.
"""

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

# =============================================================================
# TYPE ALIASES
# =============================================================================

# Use Any for dtype to avoid MatLike compatibility issues with OpenCV
Image: TypeAlias = NDArray[Any]  # BGR or grayscale image
GrayImage: TypeAlias = NDArray[Any]  # Single channel grayscale

# Memory threshold: 16 megapixels
MAX_PIXELS_BEFORE_TILING = 16_000_000

# ESPCN model directory (relative to this file)
_ESPCN_MODEL_DIR = Path(__file__).parent.parent / "models" / "espcn"


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class ImageInfo:
    """Information about a loaded image."""

    height: int
    width: int
    channels: int
    is_grayscale: bool
    megapixels: float


@dataclass(frozen=True)
class NoiseEstimate:
    """Estimated noise level in an image."""

    sigma: float  # Estimated noise standard deviation
    noise_level: str  # "low", "medium", "high"


@dataclass(frozen=True)
class QualityMetrics:
    """Image quality metrics."""

    variance: float
    sharpness: float  # Laplacian variance
    contrast: float  # RMS contrast
    noise_level: str
    overall_score: int  # 1-10


# =============================================================================
# SECTION 1: IMAGE I/O
# =============================================================================


def load_image(
    path: str | Path,
    stage: ProcessingStage = ProcessingStage.INPUT,
) -> Image | ProcessingError:
    """
    Load an image from disk with validation.

    Handles:
    - Corrupted images (cv2.imread failure)
    - Grayscale images (converts to BGR for consistency)
    - File not found / permission errors

    Args:
        path: Path to image file
        stage: Processing stage for error reporting

    Returns:
        BGR image array or ProcessingError
    """
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
        # Read image (cv2.imread returns None on failure)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            return ProcessingError(
                stage=stage,
                error_type="imread_failed",
                recoverable=False,
                message=f"Failed to read image (may be corrupted): {path}",
                details={"path": str(path)},
            )

        # Convert to BGR for consistent processing
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            channels = img.shape[2]
            if channels == 1:
                # Single channel 3D array
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif channels == 4:
                # RGBA -> BGR (drop alpha)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # channels == 3 (BGR) falls through unchanged

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
    """
    Save an image to disk.

    Args:
        image: Image to save
        path: Output path
        stage: Processing stage for error reporting

    Returns:
        Path to saved file or ProcessingError
    """
    path = Path(path)

    try:
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(path), image)
        if not success:
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
    """
    Extract metadata about an image.

    Args:
        image: BGR or grayscale image

    Returns:
        ImageInfo with dimensions and channel info
    """
    if len(image.shape) == 2:
        height, width = image.shape
        channels = 1
        is_grayscale = True
    else:
        height, width, channels = image.shape
        is_grayscale = channels == 1

    megapixels = (height * width) / 1_000_000

    return ImageInfo(
        height=height,
        width=width,
        channels=channels,
        is_grayscale=is_grayscale,
        megapixels=megapixels,
    )


# =============================================================================
# SECTION 2: SCALING & RESOLUTION
# =============================================================================


def calculate_scale_factor(
    current_size: tuple[int, int],
    target_resolution: int = config.TARGET_RESOLUTION,
    min_resolution: int = config.MIN_RESOLUTION,
    max_resolution: int = config.MAX_RESOLUTION,
) -> tuple[float, str] | ProcessingError:
    """
    Calculate optimal scale factor for an image.

    Args:
        current_size: (width, height) of current image
        target_resolution: Target size for longest edge
        min_resolution: Reject images below this
        max_resolution: Reject images above this (too large for memory)

    Returns:
        (scale_factor, method) where method is "espcn", "lanczos_up", "lanczos_down", or "none"
        or ProcessingError if outside resolution bounds
    """
    width, height = current_size
    longest_edge = max(width, height)

    # Reject images that are too small
    if longest_edge < min_resolution:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="image_too_small",
            recoverable=False,
            message=f"Image too small ({longest_edge}px). Minimum is {min_resolution}px.",
            details={
                "width": width,
                "height": height,
                "min_resolution": min_resolution,
            },
        )

    # Reject images that are too large (would cause memory issues)
    if longest_edge > max_resolution:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="image_too_large",
            recoverable=True,  # Could be handled with tiling
            message=f"Image too large ({longest_edge}px). Maximum is {max_resolution}px.",
            details={
                "width": width,
                "height": height,
                "max_resolution": max_resolution,
            },
        )

    # Already at or near target resolution
    if abs(longest_edge - target_resolution) < 50:
        return (1.0, "none")

    # Need to downscale to target
    if longest_edge > target_resolution:
        scale = target_resolution / longest_edge
        return (scale, "lanczos_down")

    # Need to upscale
    scale = target_resolution / longest_edge

    # ESPCN works best with integer scale factors 2, 3, or 4
    if scale <= 2.0:
        return (2.0, "espcn")
    elif scale <= 3.0:
        return (3.0, "espcn")
    elif scale <= 4.0:
        return (4.0, "espcn")
    else:
        # For larger scales, use Lanczos
        return (scale, "lanczos_up")


def upscale_espcn(
    image: Image,
    scale_factor: int = 2,
    model_path: str | None = None,
) -> Image | ProcessingError:
    """
    Upscale image using ESPCN super-resolution.

    Falls back to Lanczos if ESPCN model unavailable.

    Args:
        image: Input image
        scale_factor: Upscaling factor (2, 3, or 4)
        model_path: Path to ESPCN model weights (optional)

    Returns:
        Upscaled image or ProcessingError
    """
    if scale_factor not in (2, 3, 4):
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="invalid_scale_factor",
            recoverable=True,
            message=f"ESPCN only supports scale factors 2, 3, or 4. Got {scale_factor}.",
            details={"scale_factor": scale_factor},
        )

    # Check if dnn_superres is available (requires opencv-contrib-python)
    if not hasattr(cv2, "dnn_superres"):
        return resize_lanczos(image, scale_factor=float(scale_factor))

    # Use default model path if not provided
    if model_path is None:
        model_path = str(_ESPCN_MODEL_DIR / f"ESPCN_x{scale_factor}.pb")

    # Check if model file exists
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
            message=f"ESPCN upscaling failed: {e}. Falling back to Lanczos.",
            details={"scale_factor": scale_factor, "error": str(e)},
        )


def resize_lanczos(
    image: Image,
    target_size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
) -> Image:
    """
    Resize image using Lanczos interpolation.

    Args:
        image: Input image
        target_size: (width, height) or None to use scale_factor
        scale_factor: Scale multiplier or None to use target_size

    Returns:
        Resized image
    """
    # Handle empty images
    if image.size == 0:
        return image

    if target_size is not None:
        # Validate target size
        if target_size[0] <= 0 or target_size[1] <= 0:
            return image  # Return original if invalid size
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    if scale_factor is not None and scale_factor > 0:
        height, width = image.shape[:2]
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))
        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
        )

    # No resize needed
    return image


def needs_tiling(image: Image) -> bool:
    """
    Check if an image exceeds the memory threshold for tiling.

    Args:
        image: Input image

    Returns:
        True if image has more than MAX_PIXELS_BEFORE_TILING pixels
    """
    if image.size == 0:
        return False
    height, width = image.shape[:2]
    return height * width > MAX_PIXELS_BEFORE_TILING


def process_in_tiles(
    image: Image,
    operation: Callable[[Image], Image],
    tile_size: int = 2048,
    overlap: int = 64,
) -> Image:
    """
    Process a large image in overlapping tiles to avoid memory issues.

    Splits the image into tiles, applies the operation to each tile,
    then stitches results back together using blending in overlap regions.

    Args:
        image: Input image (should be >16MP to benefit from tiling)
        operation: Function that takes an image tile and returns processed tile.
                   Must preserve tile dimensions.
        tile_size: Size of each tile (default 2048x2048)
        overlap: Overlap between tiles for seamless blending (default 64)

    Returns:
        Processed image with same dimensions as input
    """
    if image.size == 0:
        return image

    height, width = image.shape[:2]

    # If image is small enough, just process directly
    if height * width <= MAX_PIXELS_BEFORE_TILING:
        return operation(image)

    # Determine number of channels
    is_color = len(image.shape) == 3
    channels = image.shape[2] if is_color else 1
    if is_color:
        output = np.zeros((height, width, channels), dtype=image.dtype)
    else:
        output = np.zeros((height, width), dtype=image.dtype)

    # Weight array for blending overlapping regions
    weight_sum = np.zeros((height, width), dtype=np.float32)

    # Calculate effective step (tile_size - overlap)
    step = tile_size - overlap

    # Process each tile
    for y_start in range(0, height, step):
        for x_start in range(0, width, step):
            # Calculate tile boundaries
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Extract tile
            tile = image[y_start:y_end, x_start:x_end]

            # Process tile
            processed_tile = operation(tile)

            # Ensure processed tile has same dimensions as input tile
            if processed_tile.shape[:2] != tile.shape[:2]:
                processed_tile = cv2.resize(
                    processed_tile,
                    (tile.shape[1], tile.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Create weight mask for blending (feathered edges)
            tile_h, tile_w = processed_tile.shape[:2]
            weight = np.ones((tile_h, tile_w), dtype=np.float32)

            # Feather edges for smooth blending
            if y_start > 0:
                # Top edge feather
                for i in range(min(overlap, tile_h)):
                    weight[i, :] *= i / overlap
            if x_start > 0:
                # Left edge feather
                for i in range(min(overlap, tile_w)):
                    weight[:, i] *= i / overlap
            if y_end < height:
                # Bottom edge feather
                for i in range(min(overlap, tile_h)):
                    weight[tile_h - 1 - i, :] *= i / overlap
            if x_end < width:
                # Right edge feather
                for i in range(min(overlap, tile_w)):
                    weight[:, tile_w - 1 - i] *= i / overlap

            # Accumulate weighted result
            if is_color:
                for c in range(channels):
                    output[y_start:y_end, x_start:x_end, c] += (
                        processed_tile[:, :, c].astype(np.float32) * weight
                    ).astype(image.dtype)
            else:
                output[y_start:y_end, x_start:x_end] += (
                    processed_tile.astype(np.float32) * weight
                ).astype(image.dtype)

            weight_sum[y_start:y_end, x_start:x_end] += weight

    # Normalize by weight sum to complete blending
    # Avoid division by zero
    weight_sum = np.maximum(weight_sum, 1e-10)

    if is_color:
        for c in range(channels):
            output[:, :, c] = (
                output[:, :, c].astype(np.float32) / weight_sum
            ).astype(image.dtype)
    else:
        output = (output.astype(np.float32) / weight_sum).astype(image.dtype)

    return output


def downsample_to_limit(
    image: Image,
    max_pixels: int = MAX_PIXELS_BEFORE_TILING,
) -> Image:
    """
    Downsample an image if it exceeds the pixel limit.

    Use this as an alternative to tiling when the operation doesn't
    preserve spatial relationships well with tiles.

    Args:
        image: Input image
        max_pixels: Maximum number of pixels allowed

    Returns:
        Downsampled image if needed, or original if within limit
    """
    if image.size == 0:
        return image

    height, width = image.shape[:2]
    current_pixels = height * width

    if current_pixels <= max_pixels:
        return image

    # Calculate scale factor to fit within limit
    scale = np.sqrt(max_pixels / current_pixels)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)


# =============================================================================
# SECTION 3: DENOISING
# =============================================================================


def estimate_noise(image: Image) -> NoiseEstimate:
    """
    Estimate noise level in an image.

    Uses median absolute deviation on Laplacian for robust estimation.

    Args:
        image: Input image

    Returns:
        NoiseEstimate with sigma and classification
    """
    # Handle empty images first
    if image.size == 0:
        return NoiseEstimate(sigma=0.0, noise_level="low")

    # Convert to grayscale for noise estimation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Robust noise estimation using MAD (Median Absolute Deviation)
    # sigma = MAD / 0.6745 (for normal distribution)
    median_val = np.median(np.abs(laplacian))
    sigma = float(median_val / 0.6745) if median_val > 0 else 0.0

    # Classify noise level
    if sigma < 5:
        noise_level = "low"
    elif sigma < 15:
        noise_level = "medium"
    else:
        noise_level = "high"

    return NoiseEstimate(sigma=float(sigma), noise_level=noise_level)


def denoise_nlmeans(
    image: Image,
    h: float | None = None,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> Image:
    """
    Apply Non-Local Means denoising.

    If h is None, automatically determines strength based on noise estimate.

    Args:
        image: Input image
        h: Filter strength (higher removes more noise). If None, auto-calculated.
        template_window_size: Size of template patch (must be odd)
        search_window_size: Size of area to search (must be odd)

    Returns:
        Denoised image
    """
    # Handle empty images
    if image.size == 0:
        return image

    # Ensure window sizes are valid and odd
    template_window_size = max(3, template_window_size)
    if template_window_size % 2 == 0:
        template_window_size += 1
    search_window_size = max(3, search_window_size)
    if search_window_size % 2 == 0:
        search_window_size += 1

    # Auto-calculate h if not provided
    if h is None:
        noise = estimate_noise(image)
        # h should be roughly proportional to noise sigma
        # Typical values: 3-10 for low noise, 10-15 for medium, 15+ for high
        if noise.noise_level == "low":
            h = 3.0
        elif noise.noise_level == "medium":
            h = 10.0
        else:
            h = 15.0

    # Apply denoising (cast h to int, ensure >= 1 for stubs compatibility)
    h_int = max(1, int(h))
    if len(image.shape) == 3:
        # Color image
        return cv2.fastNlMeansDenoisingColored(
            src=image,
            h=h_int,
            hColor=h_int,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size,
        )
    else:
        # Grayscale image
        return cv2.fastNlMeansDenoising(
            src=image,
            h=h_int,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size,
        )


# =============================================================================
# SECTION 4: SHARPENING
# =============================================================================


def sharpen_unsharp_mask(
    image: Image,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0,
) -> Image:
    """
    Apply unsharp mask sharpening.

    Args:
        image: Input image
        kernel_size: Gaussian kernel size (must be odd, >= 1)
        sigma: Gaussian sigma
        amount: Sharpening strength (1.5 = moderate)
        threshold: Minimum difference to apply sharpening

    Returns:
        Sharpened image
    """
    # Handle empty images
    if image.size == 0:
        return image

    # Ensure kernel size is valid and odd
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create blurred version
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Unsharp mask: sharpened = original + amount * (original - blurred)
    if threshold == 0:
        # Simple case: no threshold
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    else:
        # Apply threshold to avoid amplifying noise
        diff = cv2.subtract(image, blurred)
        mask = np.abs(diff) > threshold
        sharpened = image.copy()
        sharpened[mask] = cv2.addWeighted(
            image, 1.0 + amount, blurred, -amount, 0
        )[mask]

    return sharpened


# =============================================================================
# SECTION 5: QUALITY ASSESSMENT
# =============================================================================


def calculate_variance(image: Image) -> float:
    """
    Calculate image variance (for blank image detection).

    Args:
        image: Input image

    Returns:
        Variance value (0 = solid color)
    """
    if image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return float(np.var(gray))


def calculate_sharpness(image: Image) -> float:
    """
    Calculate sharpness using Laplacian variance.

    Args:
        image: Input image

    Returns:
        Sharpness score (higher = sharper)
    """
    if image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def calculate_contrast(image: Image) -> float:
    """
    Calculate RMS contrast.

    Args:
        image: Input image

    Returns:
        Contrast value (0-1 normalized)
    """
    # Handle empty images first
    if image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize to 0-1 based on dtype
    max_val = np.iinfo(gray.dtype).max if np.issubdtype(gray.dtype, np.integer) else 1.0
    gray_normalized = gray.astype(np.float64) / max_val

    # RMS contrast
    mean = np.mean(gray_normalized)
    rms_contrast = np.sqrt(np.mean((gray_normalized - mean) ** 2))

    return float(rms_contrast)


def assess_quality(
    image: Image,
    min_variance: float = config.MIN_IMAGE_VARIANCE,
) -> QualityMetrics | ProcessingError:
    """
    Comprehensive quality assessment.

    Returns quality score 1-10:
    - 1-3: Poor (likely unrecoverable)
    - 4-6: Acceptable (may need enhancement)
    - 7-10: Good (minimal processing needed)

    Args:
        image: Input image
        min_variance: Threshold for blank image rejection

    Returns:
        QualityMetrics or ProcessingError if image is blank
    """
    variance = calculate_variance(image)

    # Reject blank/solid color images
    if variance < min_variance:
        return ProcessingError(
            stage=ProcessingStage.PREPROCESS,
            error_type="blank_image",
            recoverable=False,
            message=f"Image appears blank or solid color (variance={variance:.2f})",
            details={"variance": variance, "min_variance": min_variance},
        )

    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    noise = estimate_noise(image)

    # Calculate overall score (1-10)
    # Weight: sharpness (40%), contrast (30%), noise (30%)
    sharpness_score = min(10, max(1, int(sharpness / 100)))  # rough scaling
    contrast_score = min(10, max(1, int(contrast * 30)))  # contrast is 0-1
    noise_score = 10 if noise.noise_level == "low" else (6 if noise.noise_level == "medium" else 3)

    overall = int(0.4 * sharpness_score + 0.3 * contrast_score + 0.3 * noise_score)
    overall = min(10, max(1, overall))

    return QualityMetrics(
        variance=variance,
        sharpness=sharpness,
        contrast=contrast,
        noise_level=noise.noise_level,
        overall_score=overall,
    )


# =============================================================================
# SECTION 6: HELPERS
# =============================================================================


def to_grayscale(image: Image) -> GrayImage:
    """
    Convert BGR image to grayscale.

    Args:
        image: BGR image (3 channels) or already grayscale

    Returns:
        Single-channel grayscale image
    """
    if len(image.shape) == 2:
        return image
    if len(image.shape) == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        # Assume BGR for 3 channels
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ensure_bgr(image: Image) -> Image:
    """
    Ensure image is in BGR format.

    Args:
        image: Grayscale or BGR image

    Returns:
        3-channel BGR image
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3:
        channels = image.shape[2]
        if channels == 1:
            # Single channel 3D array -> BGR
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
