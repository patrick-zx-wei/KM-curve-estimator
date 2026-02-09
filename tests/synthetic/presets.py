"""Preset generation profiles: difficult and standard suites.

generate_difficult(): 5 hand-picked realistic hard scenarios (Legacy tier).
generate_standard(): 100 cases across 3 tiers (50 pristine / 35 standard / 15 legacy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from .data_gen import SyntheticTestCase, generate_test_case
from .ground_truth import save_manifest, save_test_case
from .modifiers import (
    Annotations,
    BackgroundStyle,
    CensoringMarks,
    CurveDirection,
    FontTypography,
    FrameLayout,
    GridLines,
    JPEGArtifacts,
    LowResolution,
    Modifier,
    NoisyBackground,
    RiskTableDisplay,
    ThickLines,
    ThinLines,
    TruncatedYAxis,
)
from .renderer import render_test_case


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------


@dataclass
class TierConfig:
    """Configuration for a generation tier."""

    name: str
    count: int
    # Figure-stage modifier probabilities
    risk_table_prob: float
    annotations_prob: float
    grid_lines_prob: float
    truncated_y_prob: float
    thin_lines_prob: float
    # Censoring
    high_censoring_prob: float
    normal_censoring_range: tuple[float, float]
    high_censoring_range: tuple[float, float]
    # Image quality (post-render)
    lowres_width_range: tuple[int, int]
    jpeg_quality_range: tuple[int, int]
    noise_sigma_range: tuple[float, float]
    # Difficulty
    min_difficulty: int


TIER_PRISTINE = TierConfig(
    name="pristine",
    count=50,
    risk_table_prob=1.00,
    annotations_prob=0.30,
    grid_lines_prob=0.30,
    truncated_y_prob=0.10,
    thin_lines_prob=0.03,
    high_censoring_prob=0.10,
    normal_censoring_range=(0.02, 0.06),
    high_censoring_range=(0.08, 0.16),
    # Pristine: native resolution, lossless — no post-render applied
    lowres_width_range=(1200, 1200),
    jpeg_quality_range=(100, 100),
    noise_sigma_range=(0.0, 0.0),
    min_difficulty=1,
)

TIER_STANDARD = TierConfig(
    name="standard",
    count=35,
    risk_table_prob=1.00,
    annotations_prob=0.45,
    grid_lines_prob=0.50,
    truncated_y_prob=0.20,
    thin_lines_prob=0.12,
    high_censoring_prob=0.43,
    normal_censoring_range=(0.03, 0.08),
    high_censoring_range=(0.10, 0.20),
    # Standard: web-optimized journal (mild JPEG, slight downscale)
    lowres_width_range=(1000, 1200),
    jpeg_quality_range=(80, 95),
    noise_sigma_range=(0.0, 0.0),
    min_difficulty=2,
)

TIER_LEGACY = TierConfig(
    name="legacy",
    count=15,
    risk_table_prob=1.00,
    annotations_prob=0.60,
    grid_lines_prob=0.55,
    truncated_y_prob=0.40,
    thin_lines_prob=0.20,
    high_censoring_prob=0.66,
    normal_censoring_range=(0.04, 0.10),
    high_censoring_range=(0.12, 0.24),
    # Legacy: scanned / old papers
    lowres_width_range=(900, 1050),
    jpeg_quality_range=(72, 82),
    noise_sigma_range=(2.0, 5.0),
    min_difficulty=4,
)

TIERS = [TIER_PRISTINE, TIER_STANDARD, TIER_LEGACY]
GapPattern = Literal["diverging", "parallel", "converging", "crossover"]
BackgroundStyleLabel = Literal["white", "sas_gray", "ggplot_gray"]
CurveDirectionLabel = Literal["downward", "upward"]
FrameLayoutLabel = Literal["l_axis", "full_box"]
FontFamilyLabel = Literal["sans", "serif"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_and_save(
    test_case: SyntheticTestCase, output_dir: Path
) -> SyntheticTestCase:
    """Render and persist a test case."""
    case_dir = output_dir / test_case.name
    render_test_case(test_case, case_dir)
    save_test_case(test_case, case_dir)
    return test_case


def _pick_line_styles(
    rng: np.random.Generator, n_curves: int
) -> list[str]:
    """Pick line styles: 75% all-solid, 25% mixed (solid + dashed)."""
    if rng.random() < 0.75:
        return ["solid"] * n_curves
    # Mixed: first curve solid, rest dashed
    return ["solid"] + ["dashed"] * (n_curves - 1)


def _sample_literature_style_modifiers(case_rng: np.random.Generator) -> list[Modifier]:
    """Sample style features using literature-informed prevalence.

    Distributions:
    - Background: 75% white, 20% SAS gray, 5% ggplot gray
    - Direction: 85% downward (survival), 15% upward (incidence)
    - Frame: 60% L-axis, 40% full box
    - Font: 80% sans, 20% serif
    """
    background = cast(
        BackgroundStyleLabel,
        str(
            case_rng.choice(
                ["white", "sas_gray", "ggplot_gray"],
                p=[0.75, 0.20, 0.05],
            )
        ),
    )
    direction = cast(
        CurveDirectionLabel,
        str(case_rng.choice(["downward", "upward"], p=[0.85, 0.15])),
    )
    frame_layout = cast(
        FrameLayoutLabel,
        str(case_rng.choice(["l_axis", "full_box"], p=[0.60, 0.40])),
    )
    font_family = cast(
        FontFamilyLabel,
        str(case_rng.choice(["sans", "serif"], p=[0.80, 0.20])),
    )
    return cast(
        list[Modifier],
        [
            BackgroundStyle(style=background),
            CurveDirection(direction=direction),
            FrameLayout(layout=frame_layout),
            FontTypography(family=font_family),
        ],
    )


def _apply_literature_style_profile(style_profile: dict[str, str]) -> list[Modifier]:
    """Build style modifiers from a precomputed profile."""
    return cast(
        list[Modifier],
        [
            BackgroundStyle(
                style=cast(BackgroundStyleLabel, style_profile["background_style"])
            ),
            CurveDirection(
                direction=cast(CurveDirectionLabel, style_profile["curve_direction"])
            ),
            FrameLayout(
                layout=cast(FrameLayoutLabel, style_profile["frame_layout"])
            ),
            FontTypography(
                family=cast(FontFamilyLabel, style_profile["font_typography"])
            ),
        ],
    )


def _apply_tier_modifiers(
    case_rng: np.random.Generator,
    tier: TierConfig,
    lhs_censoring_sample: float,
    n_curves: int,
    style_profile: dict[str, str] | None = None,
) -> tuple[list[Modifier], float, float, list[str]]:
    """Build modifier list and censoring_rate for a case in the given tier.

    Returns (modifiers, censoring_rate, y_start, line_styles).
    """
    if style_profile is None:
        modifiers = _sample_literature_style_modifiers(case_rng)
    else:
        modifiers = _apply_literature_style_profile(style_profile)
    modifiers.append(CensoringMarks())

    # Figure-stage modifiers (probabilistic)
    if case_rng.random() < tier.risk_table_prob:
        modifiers.append(RiskTableDisplay())

    y_start = 0.0
    if case_rng.random() < tier.truncated_y_prob:
        y_start = float(case_rng.choice([0.2, 0.3]))
        modifiers.append(TruncatedYAxis(y_start=y_start))

    if case_rng.random() < tier.grid_lines_prob:
        modifiers.append(GridLines(alpha=0.25))

    if case_rng.random() < tier.annotations_prob:
        modifiers.append(Annotations())

    thin_roll = case_rng.random()
    if thin_roll < tier.thin_lines_prob:
        modifiers.append(ThinLines(linewidth=case_rng.uniform(1.3, 1.8)))
    elif thin_roll < tier.thin_lines_prob + 0.30:
        modifiers.append(ThickLines(linewidth=case_rng.uniform(2.8, 3.4)))

    # Line styles (75% solid, 20% mixed, 5% dotted)
    line_styles = _pick_line_styles(case_rng, n_curves)

    # Censoring rate
    if case_rng.random() < tier.high_censoring_prob:
        lo, hi = tier.high_censoring_range
        censoring_rate = lo + lhs_censoring_sample * (hi - lo)
    else:
        lo, hi = tier.normal_censoring_range
        censoring_rate = lo + lhs_censoring_sample * (hi - lo)

    # Post-render degradation (tier-specific)
    _apply_post_render(case_rng, tier, modifiers)

    # --- Cap harsh modifiers: max 2 post-render, max 4 total ---
    _HARSH_POST = (JPEGArtifacts, LowResolution, NoisyBackground)
    _HARSH_FIGURE = (ThinLines, TruncatedYAxis, Annotations, GridLines)

    # Count non-white background as a harsh figure modifier
    has_harsh_bg = any(
        isinstance(m, BackgroundStyle) and m.style != "white" for m in modifiers
    )

    # 1) Cap post-render harsh to 2
    harsh_post = [m for m in modifiers if isinstance(m, _HARSH_POST)]
    while len(harsh_post) > 2:
        drop = harsh_post.pop(case_rng.integers(len(harsh_post)))
        modifiers.remove(drop)

    # 2) Cap total harsh to 4
    harsh_fig = [m for m in modifiers if isinstance(m, _HARSH_FIGURE)]
    harsh_post = [m for m in modifiers if isinstance(m, _HARSH_POST)]
    total_harsh = len(harsh_post) + len(harsh_fig) + (1 if has_harsh_bg else 0)
    while total_harsh > 4 and harsh_fig:
        drop = harsh_fig.pop(case_rng.integers(len(harsh_fig)))
        modifiers.remove(drop)
        if isinstance(drop, TruncatedYAxis):
            y_start = 0.0
        total_harsh -= 1

    # --- Enforce minimum 2.3px final line thickness ---
    _DPI = 150
    _NATIVE_WIDTH = 1500  # 10in * 150dpi
    _MIN_FINAL_PX = 2.3

    thin_mod = next((m for m in modifiers if isinstance(m, ThinLines)), None)
    lowres_mod = next((m for m in modifiers if isinstance(m, LowResolution)), None)
    linewidth_pt = thin_mod.linewidth if thin_mod else 2.0
    target_w = lowres_mod.target_width if lowres_mod else _NATIVE_WIDTH
    final_px = linewidth_pt * (_DPI / 72) * (target_w / _NATIVE_WIDTH)

    if final_px < _MIN_FINAL_PX and thin_mod:
        # Compute minimum target_width that would satisfy constraint
        needed_w = _MIN_FINAL_PX * 72 * _NATIVE_WIDTH / (_DPI * linewidth_pt)
        if lowres_mod and needed_w <= tier.lowres_width_range[1]:
            lowres_mod.target_width = int(math.ceil(needed_w))
        else:
            # Increase linewidth to satisfy constraint at current resolution
            needed_lw = _MIN_FINAL_PX * 72 * _NATIVE_WIDTH / (_DPI * target_w)
            if needed_lw <= 1.8:
                thin_mod.linewidth = needed_lw
            else:
                # Can't stay thin — remove ThinLines entirely
                modifiers.remove(thin_mod)

    return modifiers, censoring_rate, y_start, line_styles


def _apply_post_render(
    rng: np.random.Generator,
    tier: TierConfig,
    modifiers: list[Modifier],
) -> None:
    """Apply tier-appropriate post-render degradation."""
    if tier.name == "pristine":
        # No degradation — native resolution, lossless PNG
        return

    if tier.name == "standard":
        # Web-optimized: mild JPEG, optional slight downscale
        lo_w, hi_w = tier.lowres_width_range
        lo_q, hi_q = tier.jpeg_quality_range
        if rng.random() < 0.5:
            modifiers.append(
                LowResolution(target_width=int(rng.integers(lo_w, hi_w + 1)))
            )
        modifiers.append(
            JPEGArtifacts(quality=int(rng.integers(lo_q, hi_q + 1)))
        )
        return

    # Legacy: always resolution + JPEG + noise
    lo_w, hi_w = tier.lowres_width_range
    lo_q, hi_q = tier.jpeg_quality_range
    lo_s, hi_s = tier.noise_sigma_range
    modifiers.append(LowResolution(target_width=int(rng.integers(lo_w, hi_w + 1))))
    modifiers.append(JPEGArtifacts(quality=int(rng.integers(lo_q, hi_q + 1))))
    modifiers.append(NoisyBackground(sigma=rng.uniform(lo_s, hi_s)))


# ---------------------------------------------------------------------------
# Difficult preset definitions (Legacy tier)
# ---------------------------------------------------------------------------


def _difficult_low_res_overlap(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="low_res_overlap",
        seed=seed,
        n_curves=3,
        n_per_arm=180,
        max_time=48.0,
        weibull_ks=[0.95, 1.05, 1.15],
        weibull_scale=32.0,
        censoring_rate=0.02,
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            GridLines(alpha=0.25),
            Annotations(),
            LowResolution(target_width=850),
            JPEGArtifacts(quality=70),
        ],
        difficulty=4,
        tier="legacy",
    )


def _difficult_truncated_noisy(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="truncated_noisy",
        seed=seed,
        n_curves=2,
        n_per_arm=200,
        max_time=60.0,
        weibull_ks=[1.35, 1.85],
        weibull_scale=40.0,
        censoring_rate=0.02,
        y_axis_start=0.25,
        modifiers=[
            TruncatedYAxis(y_start=0.25),
            CensoringMarks(),
            RiskTableDisplay(),
            GridLines(alpha=0.25),
            ThinLines(linewidth=1.3),
            NoisyBackground(sigma=5.0),
        ],
        difficulty=4,
        tier="legacy",
    )


def _difficult_crossing_compressed(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="crossing_compressed",
        seed=seed,
        n_curves=2,
        n_per_arm=180,
        max_time=60.0,
        weibull_ks=[0.75, 1.95],
        weibull_scale=35.0,
        censoring_rate=0.015,
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            Annotations(),
            GridLines(major=True, alpha=0.25),
            LowResolution(target_width=900),
            JPEGArtifacts(quality=70),
        ],
        difficulty=4,
        tier="legacy",
    )


def _difficult_four_arm_lowres(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="four_arm_lowres",
        seed=seed,
        n_curves=4,
        n_per_arm=130,
        max_time=36.0,
        weibull_ks=[0.9, 1.0, 1.08, 1.15],
        weibull_scale=25.0,
        censoring_rate=0.02,
        group_names=["Arm A", "Arm B", "Arm C", "Arm D"],
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            GridLines(alpha=0.25),
            Annotations(),
            LowResolution(target_width=880),
        ],
        difficulty=4,
        tier="legacy",
    )


def _difficult_sparse_degraded(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="sparse_degraded",
        seed=seed,
        n_curves=2,
        n_per_arm=95,
        max_time=48.0,
        weibull_ks=[0.95, 1.05],
        weibull_scale=24.0,
        censoring_rate=0.08,
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            GridLines(alpha=0.25),
            Annotations(),
            LowResolution(target_width=820),
            JPEGArtifacts(quality=68),
            NoisyBackground(sigma=5.0),
        ],
        difficulty=5,
        tier="legacy",
    )


DIFFICULT_PRESETS = [
    _difficult_low_res_overlap,
    _difficult_truncated_noisy,
    _difficult_crossing_compressed,
    _difficult_four_arm_lowres,
    _difficult_sparse_degraded,
]


def generate_difficult(
    output_dir: str | Path = "tests/fixtures/difficult",
    base_seed: int = 42,
) -> list[SyntheticTestCase]:
    """Generate 5 difficult test cases (Legacy tier)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for i, preset_fn in enumerate(DIFFICULT_PRESETS):
        tc = preset_fn(seed=base_seed + i)
        _generate_and_save(tc, output_dir)
        cases.append(tc)

    save_manifest(cases, output_dir)
    return cases


# ---------------------------------------------------------------------------
# Standard generation (100 cases: 50 pristine / 35 standard / 15 legacy)
# ---------------------------------------------------------------------------


def _latin_hypercube(n: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin Hypercube Sampling returning n x n_dims array in [0,1]."""
    result = np.zeros((n, n_dims))
    for dim in range(n_dims):
        perm = rng.permutation(n)
        for i in range(n):
            low = perm[i] / n
            high = (perm[i] + 1) / n
            result[i, dim] = rng.uniform(low, high)
    return result


def _build_weighted_schedule(
    n: int,
    labels: list[str],
    probs: list[float],
    rng: np.random.Generator,
) -> list[str]:
    """Build a length-n categorical schedule with exact rounded prevalence."""
    if n <= 0:
        return []
    expected = np.asarray(probs, dtype=np.float64) * float(n)
    counts = np.floor(expected).astype(np.int64)
    remainder = int(n - int(np.sum(counts)))
    if remainder > 0:
        frac = expected - counts
        order = np.argsort(-frac, kind="mergesort")
        for idx in order[:remainder]:
            counts[idx] += 1
    schedule: list[str] = []
    for label, count in zip(labels, counts):
        schedule.extend([label] * int(count))
    rng.shuffle(schedule)
    return schedule


def _sample_weibull_ks(
    rng: np.random.Generator,
    base_k: float,
    n_curves: int,
    min_gap: float = 0.25,
    max_attempts: int = 32,
) -> list[float]:
    """Sample curve-shape parameters with minimum spread to reduce prolonged overlap."""
    if n_curves <= 1:
        return [float(base_k)]

    lower_k = 0.6
    upper_k = 2.0
    best_ks: list[float] = []
    best_gap = -1.0

    for _ in range(max_attempts):
        ks = [float(np.clip(base_k * rng.uniform(0.80, 1.25), lower_k, upper_k)) for _ in range(n_curves)]
        sorted_ks = sorted(ks)
        min_observed_gap = min(sorted_ks[i + 1] - sorted_ks[i] for i in range(len(sorted_ks) - 1))
        if min_observed_gap > best_gap:
            best_gap = min_observed_gap
            best_ks = ks
        if min_observed_gap >= min_gap:
            return ks

    # Fallback: deterministic spread around base_k when random draws are too close.
    center = float(np.clip(base_k, lower_k, upper_k))
    half = (n_curves - 1) / 2.0
    spread = max(min_gap, 0.18)
    return [
        float(np.clip(center + (idx - half) * spread, lower_k, upper_k))
        for idx in range(n_curves)
    ]


def _clip_k(value: float) -> float:
    return float(np.clip(value, 0.6, 2.0))


def _enforce_min_multiplier_gap(
    values: np.ndarray,
    min_gap: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Ensure sorted multipliers are separated by at least min_gap."""
    if values.size <= 1:
        return values

    out = np.sort(values.astype(np.float64, copy=True))
    for idx in range(1, out.size):
        if out[idx] - out[idx - 1] < min_gap:
            out[idx] = out[idx - 1] + min_gap

    if out[-1] > high:
        out -= out[-1] - high
    if out[0] < low:
        out += low - out[0]
    return np.clip(out, low, high)


def _choose_gap_pattern(
    rng: np.random.Generator,
    tier_name: str,
    n_curves: int,
) -> GapPattern:
    """Sample clinically plausible KM gap patterns."""
    if tier_name == "pristine":
        patterns = ["parallel", "diverging", "converging", "crossover"]
        weights = np.array([0.38, 0.26, 0.26, 0.10], dtype=np.float64)
    elif tier_name == "standard":
        patterns = ["parallel", "diverging", "converging", "crossover"]
        weights = np.array([0.34, 0.26, 0.28, 0.12], dtype=np.float64)
    else:
        patterns = ["parallel", "diverging", "converging", "crossover"]
        weights = np.array([0.30, 0.26, 0.28, 0.16], dtype=np.float64)

    # Multi-arm crossover is rare and often visually underdetermined.
    if n_curves >= 3:
        weights[patterns.index("crossover")] *= 0.25
        weights = weights / np.sum(weights)

    return rng.choice(patterns, p=weights)  # type: ignore[return-value]


def _sample_pattern_parameters(
    rng: np.random.Generator,
    n_curves: int,
    base_k: float,
    base_scale: float,
    pattern: GapPattern,
) -> tuple[list[float], list[float]]:
    """Generate per-arm Weibull parameters from a target gap pattern."""
    if n_curves <= 1:
        return [_clip_k(base_k)], [float(base_scale)]

    control_k = _clip_k(base_k * rng.uniform(0.92, 1.08))
    control_scale = float(base_scale * rng.uniform(0.95, 1.05))
    ks = [control_k]
    scales = [control_scale]

    n_extra = n_curves - 1
    if pattern == "parallel":
        improvements = _enforce_min_multiplier_gap(
            rng.uniform(1.12, 1.58, size=n_extra),
            min_gap=0.05,
            low=1.12,
            high=1.58,
        )
        for imp in improvements:
            ks.append(_clip_k(control_k * rng.uniform(0.94, 1.06)))
            scales.append(float(control_scale * imp))
    elif pattern == "diverging":
        control_k = _clip_k(max(control_k, 1.0))
        ks[0] = control_k
        improvements = _enforce_min_multiplier_gap(
            rng.uniform(1.10, 1.48, size=n_extra),
            min_gap=0.05,
            low=1.10,
            high=1.48,
        )
        for idx, imp in enumerate(improvements):
            k_shift = rng.uniform(0.12, 0.32) + 0.04 * idx
            ks.append(_clip_k(control_k - k_shift))
            scales.append(float(control_scale * imp))
    elif pattern == "converging":
        control_k = _clip_k(min(control_k, 1.05))
        ks[0] = control_k
        improvements = _enforce_min_multiplier_gap(
            rng.uniform(1.10, 1.48, size=n_extra),
            min_gap=0.05,
            low=1.10,
            high=1.48,
        )
        improvements = improvements[::-1]
        for idx, imp in enumerate(improvements):
            k_shift = rng.uniform(0.18, 0.45) + 0.05 * idx
            ks.append(_clip_k(control_k + k_shift))
            scales.append(float(control_scale * imp))
    else:  # crossover
        control_k = _clip_k(control_k * rng.uniform(0.78, 0.95))
        ks[0] = control_k
        cross_k = _clip_k(control_k + rng.uniform(0.45, 0.85))
        cross_scale = float(control_scale * rng.uniform(1.35, 1.75))
        ks.append(cross_k)
        scales.append(cross_scale)
        for _ in range(max(0, n_extra - 1)):
            ks.append(_clip_k(control_k * rng.uniform(0.95, 1.08)))
            scales.append(float(control_scale * rng.uniform(1.06, 1.28)))

    return ks, scales


def _step_survival_at(curve: SyntheticTestCase, curve_idx: int, t: float) -> float:
    coords = curve.curves[curve_idx].step_coords
    if not coords:
        return 1.0
    if t < coords[0][0]:
        return 1.0
    for i in range(len(coords) - 1, -1, -1):
        if coords[i][0] <= t:
            return float(coords[i][1])
    return float(coords[0][1])


def _n_at_risk_at(curve_case: SyntheticTestCase, curve_idx: int, t: float) -> int:
    counts = curve_case.curves[curve_idx].n_at_risk
    if not counts:
        return len(curve_case.curves[curve_idx].patients)
    out = counts[0][1]
    for t_i, n_i in counts:
        if t_i <= t + 1e-9:
            out = n_i
        else:
            break
    return int(out)


def _primary_pair_indices(case: SyntheticTestCase) -> tuple[int, int]:
    names = [c.group_name.strip().lower() for c in case.curves]
    control_idx = next((i for i, n in enumerate(names) if n == "control"), 0)
    treatment_idx = next((i for i, _ in enumerate(names) if i != control_idx), 1)
    return control_idx, treatment_idx


def _pattern_fit(
    case: SyntheticTestCase,
    pattern: GapPattern,
) -> tuple[bool, float]:
    """Return (accepted, score) for pattern realism on the primary pair."""
    if len(case.curves) < 2:
        return True, 1.0

    ctrl_idx, trt_idx = _primary_pair_indices(case)
    max_time = float(case.x_axis.end)
    t_early = 0.25 * max_time
    t_mid = 0.50 * max_time
    t_late = 0.75 * max_time

    d_early = _step_survival_at(case, trt_idx, t_early) - _step_survival_at(case, ctrl_idx, t_early)
    d_mid = _step_survival_at(case, trt_idx, t_mid) - _step_survival_at(case, ctrl_idx, t_mid)
    d_late = _step_survival_at(case, trt_idx, t_late) - _step_survival_at(case, ctrl_idx, t_late)

    abs_early = abs(d_early)
    abs_mid = abs(d_mid)
    abs_late = abs(d_late)
    max_abs = max(abs_early, abs_mid, abs_late)
    score = max_abs
    accepted = False

    if pattern == "parallel":
        drift = abs(d_early - d_mid) + abs(d_mid - d_late)
        accepted = (
            max_abs >= 0.06
            and drift <= 0.12
            and (np.sign(d_mid) == np.sign(d_late) or abs_mid < 0.01 or abs_late < 0.01)
        )
        score += 0.08 - min(0.08, drift)
    elif pattern == "diverging":
        sign = np.sign(d_late if abs_late >= 1e-3 else d_mid)
        growth_1 = sign * (d_mid - d_early)
        growth_2 = sign * (d_late - d_mid)
        accepted = (
            sign != 0
            and sign * d_late >= 0.06
            and growth_1 >= -0.01
            and growth_2 >= 0.01
            and sign * (d_late - d_early) >= 0.03
        )
        score += sign * (d_late - d_early)
    elif pattern == "converging":
        sign = np.sign(d_early if abs_early >= 1e-3 else d_mid)
        shrink_1 = sign * (d_early - d_mid)
        shrink_2 = sign * (d_mid - d_late)
        accepted = (
            sign != 0
            and sign * d_early >= 0.06
            and shrink_1 >= 0.0
            and shrink_2 >= -0.01
            and abs_late <= max(0.05, abs_early - 0.02)
        )
        score += sign * (d_early - d_late)
    else:  # crossover
        sign_changes = 0
        if d_early * d_mid < -0.001:
            sign_changes += 1
        if d_mid * d_late < -0.001:
            sign_changes += 1
        accepted = sign_changes >= 1 and max_abs >= 0.06 and abs(d_late - d_early) >= 0.06
        score += abs(d_late - d_early)

    # Guard against deceptive giant late gaps when very few are at risk.
    n0 = min(len(case.curves[ctrl_idx].patients), len(case.curves[trt_idx].patients))
    n_late = min(_n_at_risk_at(case, ctrl_idx, t_late), _n_at_risk_at(case, trt_idx, t_late))
    low_risk_threshold = max(8, int(0.06 * n0))
    if n_late < low_risk_threshold and abs_late > 0.18:
        accepted = False
        score -= 0.15

    return accepted, float(score)


def _best_matching_pattern(case: SyntheticTestCase) -> GapPattern:
    """Pick the best-fitting realized pattern for metadata labeling."""
    patterns: list[GapPattern] = ["parallel", "diverging", "converging", "crossover"]
    accepted_candidates: list[tuple[float, GapPattern]] = []
    scored_candidates: list[tuple[float, GapPattern]] = []
    for pat in patterns:
        accepted, score = _pattern_fit(case, pat)
        scored_candidates.append((score, pat))
        if accepted:
            accepted_candidates.append((score, pat))
    if accepted_candidates:
        return max(accepted_candidates, key=lambda item: item[0])[1]
    return max(scored_candidates, key=lambda item: item[0])[1]


def _compute_difficulty(
    n_curves: int,
    weibull_ks: list[float],
    censoring_rate: float,
    y_start: float,
    n_per_arm: int,
    has_post_render: bool,
) -> int:
    """Estimate difficulty score (1-5)."""
    score = 0.0

    # More curves = harder
    score += {1: 0, 2: 0.5, 3: 1.5, 4: 2.5, 5: 3.0}.get(n_curves, 3)

    # Extreme Weibull shapes
    for k in weibull_ks:
        if k < 0.6 or k > 2.0:
            score += 0.5

    # Crossing curves (some k<1 and some k>1)
    if any(k < 1 for k in weibull_ks) and any(k > 1 for k in weibull_ks):
        score += 1.0

    # Heavy censoring
    if censoring_rate > 0.05:
        score += 0.5
    if censoring_rate > 0.10:
        score += 0.5

    # Truncated y-axis
    if y_start > 0:
        score += 0.5

    # Small sample
    if n_per_arm < 50:
        score += 0.5

    # Post-render degradation
    if has_post_render:
        score += 1.0

    return min(5, max(1, int(round(score))))


def generate_standard(
    output_dir: str | Path = "tests/fixtures/standard",
    base_seed: int = 1000,
) -> list[SyntheticTestCase]:
    """Generate 100 test cases across 3 tiers (50 pristine / 35 standard / 15 legacy)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = 100
    rng = np.random.default_rng(base_seed)
    style_rng = np.random.default_rng(base_seed + 7919)

    # LHS over 5 dimensions: [log_k, scale_factor, censoring_rate, n_per_arm, max_time_idx]
    lhs = _latin_hypercube(n_total, 5, rng)
    background_schedule = _build_weighted_schedule(
        n_total,
        labels=["white", "sas_gray", "ggplot_gray"],
        probs=[0.75, 0.20, 0.05],
        rng=style_rng,
    )
    direction_schedule = _build_weighted_schedule(
        n_total,
        labels=["downward", "upward"],
        probs=[0.85, 0.15],
        rng=style_rng,
    )
    frame_schedule = _build_weighted_schedule(
        n_total,
        labels=["l_axis", "full_box"],
        probs=[0.60, 0.40],
        rng=style_rng,
    )
    font_schedule = _build_weighted_schedule(
        n_total,
        labels=["sans", "serif"],
        probs=[0.80, 0.20],
        rng=style_rng,
    )

    max_time_choices = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 96.0, 120.0]

    cases = []
    for i in range(n_total):
        sample = lhs[i]
        case_seed = base_seed + 100 + i
        case_rng = np.random.default_rng(case_seed)

        # Determine tier by index range
        if i < TIER_PRISTINE.count:
            tier = TIER_PRISTINE
        elif i < TIER_PRISTINE.count + TIER_STANDARD.count:
            tier = TIER_STANDARD
        else:
            tier = TIER_LEGACY

        max_time = max_time_choices[int(sample[4] * 7.99)]

        # Statistical parameters from LHS
        n_curves = int(case_rng.choice([1, 2, 3], p=[0.15, 0.70, 0.15]))
        log_k = np.log(0.6) + sample[0] * (np.log(2.0) - np.log(0.6))
        base_k = np.exp(log_k)
        n_per_arm = int(50 + sample[3] * 450)  # 50-500
        scale_factor = 0.65 + sample[1] * 0.45
        # Ensure events spread across the axis: for high k (steep hazard),
        # the Weibull 90th percentile is scale * (-ln(0.1))^(1/k).  We need
        # that to reach at least 2/3 of max_time.
        min_scale = (2 / 3) * max_time / max((-np.log(0.1)) ** (1 / base_k), 1e-6)
        base_scale = max(scale_factor * max_time, min_scale)

        style_profile = {
            "background_style": background_schedule[i],
            "curve_direction": direction_schedule[i],
            "frame_layout": frame_schedule[i],
            "font_typography": font_schedule[i],
        }

        # Tier-specific modifiers, censoring, y_start, line styles
        modifiers, censoring_rate, y_start, line_styles = _apply_tier_modifiers(
            case_rng, tier, sample[2], n_curves, style_profile=style_profile
        )
        # Cap censoring rate so the median censoring time is at least half of
        # max_time.  This prevents curves from flatlineing early because all
        # patients are censored, while still allowing enough events for
        # crossover / diverging / converging patterns to emerge.
        max_censoring_rate = np.log(2) / (0.5 * max_time)
        censoring_rate = min(censoring_rate, max_censoring_rate)

        gap_pattern = _choose_gap_pattern(case_rng, tier.name, n_curves)
        if n_curves <= 2:
            group_names = ["Control", "Treatment"][:n_curves]
        elif n_curves == 3:
            group_names = ["Control", "Treatment A", "Treatment B"]
        else:
            group_names = [f"Arm {chr(65 + j)}" for j in range(n_curves)]

        if gap_pattern == "crossover":
            enforce_separation = False
            min_separation = 0.035
        elif gap_pattern == "converging":
            enforce_separation = bool(case_rng.random() < 0.75)
            min_separation = 0.045
        elif gap_pattern == "parallel":
            enforce_separation = bool(case_rng.random() < 0.80)
            min_separation = 0.060
        else:
            enforce_separation = bool(case_rng.random() < 0.80)
            min_separation = 0.055

        if n_curves >= 3:
            min_separation += 0.005

        has_post = any(
            isinstance(m, (LowResolution, JPEGArtifacts, NoisyBackground))
            for m in modifiers
        )
        tc: SyntheticTestCase | None = None
        best_score = -1e9
        best_case: SyntheticTestCase | None = None

        # Retry until the primary arm-pair matches the intended gap pattern.
        for attempt in range(14):
            pattern_rng = np.random.default_rng(case_seed * 97 + attempt * 1009 + 17)
            weibull_ks, weibull_scales = _sample_pattern_parameters(
                pattern_rng,
                n_curves=n_curves,
                base_k=base_k,
                base_scale=base_scale,
                pattern=gap_pattern,
            )
            difficulty = _compute_difficulty(
                n_curves, weibull_ks, censoring_rate, y_start, n_per_arm, has_post
            )
            difficulty = max(tier.min_difficulty, difficulty)

            candidate = generate_test_case(
                name=f"case_{i + 1:03d}",
                seed=case_seed + attempt * 13,
                n_curves=n_curves,
                n_per_arm=n_per_arm,
                max_time=max_time,
                weibull_ks=weibull_ks,
                weibull_scales=weibull_scales,
                weibull_scale=base_scale,
                censoring_rate=censoring_rate,
                y_axis_start=y_start,
                group_names=group_names,
                modifiers=modifiers,
                difficulty=difficulty,
                tier=tier.name,
                line_styles=line_styles,
                enforce_curve_separation=enforce_separation,
                min_curve_separation=min_separation,
                max_curve_generation_attempts=8,
                gap_pattern=gap_pattern,
            )

            accepted, score = _pattern_fit(candidate, gap_pattern)

            # Reject candidates where curves don't travel far enough
            travel_ends = [
                c.step_coords[-1][0]
                for c in candidate.curves
                if len(c.step_coords) > 1
            ]
            if travel_ends:
                travel_ok = (
                    min(travel_ends) >= (2 / 3) * max_time
                    and max(travel_ends) >= 0.9 * max_time
                )
            else:
                travel_ok = False
            if not travel_ok:
                accepted = False
                score -= 5.0

            if score > best_score:
                best_score = score
                best_case = candidate
            if accepted:
                tc = candidate
                break

        if tc is None:
            tc = best_case
        if tc is None:
            raise RuntimeError("Failed to generate synthetic test case")
        realized_ok, _ = _pattern_fit(tc, gap_pattern)
        if not realized_ok:
            tc.gap_pattern = _best_matching_pattern(tc)

        _generate_and_save(tc, output_dir)
        cases.append(tc)

    save_manifest(cases, output_dir)
    return cases
