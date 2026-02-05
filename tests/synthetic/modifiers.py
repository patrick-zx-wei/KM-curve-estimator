"""Visual modifier definitions for synthetic KM curve test cases.

Modifiers are split into two stages:
- FIGURE: applied during matplotlib figure construction
- POST_RENDER: applied to the saved image via OpenCV
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ModifierStage(Enum):
    FIGURE = "figure"
    POST_RENDER = "post"


@dataclass
class Modifier:
    stage: ModifierStage


# --- Figure-stage modifiers ---


@dataclass
class TruncatedYAxis(Modifier):
    """Y-axis does not start at 0."""

    y_start: float = 0.4
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class GridLines(Modifier):
    """Add grid lines to the plot."""

    major: bool = True
    minor: bool = False
    alpha: float = 0.3
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class ThickLines(Modifier):
    """Use thicker curve lines."""

    linewidth: float = 3.0
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class ThinLines(Modifier):
    """Use thinner curve lines (harder to detect)."""

    linewidth: float = 0.5
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class CensoringMarks(Modifier):
    """Show | marks at censoring times on the curve."""

    marker_size: float = 6.0
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class RiskTableDisplay(Modifier):
    """Render a number-at-risk table below the plot."""

    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class LegendPlacement(Modifier):
    """Control legend location."""

    location: str = "best"
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class Annotations(Modifier):
    """Add text annotations (p-values, hazard ratios)."""

    texts: list[str] = field(
        default_factory=lambda: ["HR=0.65 (95% CI 0.45-0.94)", "p=0.021"]
    )
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


@dataclass
class CompressedTimeAxis(Modifier):
    """Many tick marks on the time axis (crowded labels)."""

    n_ticks: int = 20
    stage: ModifierStage = field(default=ModifierStage.FIGURE, init=False)


# --- Post-render modifiers ---


@dataclass
class LowResolution(Modifier):
    """Downsample the final image."""

    target_width: int = 300
    stage: ModifierStage = field(default=ModifierStage.POST_RENDER, init=False)


@dataclass
class JPEGArtifacts(Modifier):
    """Re-encode as JPEG with low quality then save back as PNG."""

    quality: int = 30
    stage: ModifierStage = field(default=ModifierStage.POST_RENDER, init=False)


@dataclass
class NoisyBackground(Modifier):
    """Add Gaussian noise to the image."""

    sigma: float = 15.0
    stage: ModifierStage = field(default=ModifierStage.POST_RENDER, init=False)


@dataclass
class GaussianBlur(Modifier):
    """Apply Gaussian blur."""

    kernel_size: int = 5
    stage: ModifierStage = field(default=ModifierStage.POST_RENDER, init=False)


def get_modifier_names(modifiers: list[Modifier]) -> list[str]:
    """Return list of modifier class names for manifest/metadata."""
    return [type(m).__name__ for m in modifiers]
