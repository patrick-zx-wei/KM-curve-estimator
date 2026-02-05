"""Preprocessing node for image scaling, denoising, and quality assessment."""

import uuid
from pathlib import Path
from tempfile import gettempdir

from km_estimator.models import PipelineState, ProcessingError, ProcessingStage
from km_estimator.utils import cv_utils


def preprocess(state: PipelineState) -> PipelineState:
    """
    Preprocess image: scale, denoise, sharpen, assess quality.

    Updates state with:
    - preprocessed_image_path: Path to processed image
    - quality_score: 1-10 quality rating
    - errors: Any processing errors encountered
    """
    cfg = state.config

    # Load image
    image = cv_utils.load_image(state.image_path, stage=ProcessingStage.PREPROCESS)
    if isinstance(image, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [image]})

    # Check if image needs scaling
    info = cv_utils.get_image_info(image)
    scale_result = cv_utils.calculate_scale_factor(
        current_size=(info.width, info.height),
        target_resolution=cfg.target_resolution,
        min_resolution=cfg.min_resolution,
        max_resolution=cfg.max_resolution,
    )

    if isinstance(scale_result, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [scale_result]})

    scale_factor, method = scale_result

    # Scale if needed
    if method == "espcn":
        scaled = cv_utils.upscale_espcn(image, scale_factor=int(scale_factor))
        if isinstance(scaled, ProcessingError):
            return state.model_copy(update={"errors": state.errors + [scaled]})
        image = scaled
    elif method in ("lanczos_up", "lanczos_down"):
        image = cv_utils.resize_lanczos(image, scale_factor=scale_factor)

    # Denoise (adaptive to noise level)
    if cv_utils.needs_tiling(image):
        image = cv_utils.process_in_tiles(image, cv_utils.denoise_nlmeans)
    else:
        image = cv_utils.denoise_nlmeans(image)

    # Sharpen (fixed moderate)
    if cv_utils.needs_tiling(image):
        image = cv_utils.process_in_tiles(
            image, lambda img: cv_utils.sharpen_unsharp_mask(img, amount=1.5)
        )
    else:
        image = cv_utils.sharpen_unsharp_mask(image, amount=1.5)

    # Assess quality
    quality = cv_utils.assess_quality(image, min_variance=cfg.min_image_variance)
    if isinstance(quality, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [quality]})

    # Save preprocessed image with unique identifier to avoid collisions
    original_path = Path(state.image_path)
    output_dir = Path(gettempdir()) / "km_estimator"
    unique_id = uuid.uuid4().hex[:8]
    output_path = output_dir / f"{original_path.stem}_{unique_id}_preprocessed.png"

    save_result = cv_utils.save_image(image, output_path)
    if isinstance(save_result, ProcessingError):
        return state.model_copy(update={"errors": state.errors + [save_result]})

    return state.model_copy(
        update={
            "preprocessed_image_path": str(save_result),
            "quality_score": quality.overall_score,
        }
    )
