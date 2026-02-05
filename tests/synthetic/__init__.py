"""Synthetic KM curve test harness.

Generate synthetic KM curve images with known ground truth for pipeline evaluation.

Usage:
    from tests.synthetic import generate_difficult, generate_standard
    from tests.synthetic import load_test_case, load_manifest
    from tests.synthetic import run_case, run_all, run_filtered

Generation (no API keys needed):
    generate_difficult()   # 5 hard cases -> tests/fixtures/difficult/
    generate_standard()    # 100 cases   -> tests/fixtures/standard/

Running (requires OPENAI_API_KEY + GEMINI_API_KEY):
    run_case("case_001", "tests/fixtures/standard/")
    run_all("tests/fixtures/standard/")
    run_filtered("tests/fixtures/standard/", difficulty_range=(3, 5))
"""

from .data_gen import SyntheticCurveData, SyntheticTestCase, generate_test_case
from .ground_truth import (
    compare_digitized_curves,
    compare_hard_points,
    load_manifest,
    load_test_case,
)
from .modifiers import (
    Annotations,
    CensoringMarks,
    CompressedTimeAxis,
    GaussianBlur,
    GridLines,
    JPEGArtifacts,
    LegendPlacement,
    LowResolution,
    Modifier,
    NoisyBackground,
    RiskTableDisplay,
    ThickLines,
    ThinLines,
    TruncatedYAxis,
)
from .presets import generate_difficult, generate_standard
from .renderer import render_test_case
from .runner import run_all, run_case, run_filtered

__all__ = [
    # Generation
    "generate_difficult",
    "generate_standard",
    "generate_test_case",
    "render_test_case",
    # Data types
    "SyntheticTestCase",
    "SyntheticCurveData",
    # Persistence
    "load_test_case",
    "load_manifest",
    # Comparison
    "compare_digitized_curves",
    "compare_hard_points",
    # Runner
    "run_case",
    "run_all",
    "run_filtered",
    # Modifiers
    "Modifier",
    "TruncatedYAxis",
    "GridLines",
    "ThickLines",
    "ThinLines",
    "CensoringMarks",
    "RiskTableDisplay",
    "LegendPlacement",
    "Annotations",
    "CompressedTimeAxis",
    "LowResolution",
    "JPEGArtifacts",
    "NoisyBackground",
    "GaussianBlur",
]
