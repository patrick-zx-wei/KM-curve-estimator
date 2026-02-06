"""Preset generation profiles: difficult and standard suites.

generate_difficult(): 5 hand-picked realistic hard scenarios (Legacy tier).
generate_standard(): 100 cases across 3 tiers (50 pristine / 35 standard / 15 legacy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_gen import SyntheticTestCase, generate_test_case
from .ground_truth import save_manifest, save_test_case
from .modifiers import (
    Annotations,
    CensoringMarks,
    CompressedTimeAxis,
    GaussianBlur,
    GridLines,
    JPEGArtifacts,
    LowResolution,
    Modifier,
    NoisyBackground,
    RiskTableDisplay,
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
    risk_table_prob=0.60,
    annotations_prob=0.30,
    grid_lines_prob=0.30,
    truncated_y_prob=0.10,
    thin_lines_prob=0.10,
    high_censoring_prob=0.10,
    normal_censoring_range=(0.01, 0.05),
    high_censoring_range=(0.06, 0.15),
    # Pristine: native resolution, lossless — no post-render applied
    lowres_width_range=(1200, 1200),
    jpeg_quality_range=(100, 100),
    noise_sigma_range=(0.0, 0.0),
    min_difficulty=1,
)

TIER_STANDARD = TierConfig(
    name="standard",
    count=35,
    risk_table_prob=0.70,
    annotations_prob=0.45,
    grid_lines_prob=0.50,
    truncated_y_prob=0.20,
    thin_lines_prob=0.28,
    high_censoring_prob=0.43,
    normal_censoring_range=(0.01, 0.05),
    high_censoring_range=(0.06, 0.15),
    # Standard: web-optimized journal (mild JPEG, slight downscale)
    lowres_width_range=(1000, 1200),
    jpeg_quality_range=(80, 95),
    noise_sigma_range=(0.0, 0.0),
    min_difficulty=2,
)

TIER_LEGACY = TierConfig(
    name="legacy",
    count=15,
    risk_table_prob=0.80,
    annotations_prob=0.60,
    grid_lines_prob=0.55,
    truncated_y_prob=0.40,
    thin_lines_prob=0.40,
    high_censoring_prob=0.66,
    normal_censoring_range=(0.01, 0.05),
    high_censoring_range=(0.06, 0.15),
    # Legacy: scanned / old papers
    lowres_width_range=(800, 950),
    jpeg_quality_range=(65, 75),
    noise_sigma_range=(3.0, 7.0),
    min_difficulty=4,
)

TIERS = [TIER_PRISTINE, TIER_STANDARD, TIER_LEGACY]


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


def _apply_tier_modifiers(
    case_rng: np.random.Generator,
    tier: TierConfig,
    lhs_censoring_sample: float,
) -> tuple[list[Modifier], float, float]:
    """Build modifier list and censoring_rate for a case in the given tier.

    Returns (modifiers, censoring_rate, y_start).
    """
    modifiers: list[Modifier] = [CensoringMarks()]

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

    if case_rng.random() < tier.thin_lines_prob:
        modifiers.append(ThinLines(linewidth=case_rng.uniform(0.8, 1.2)))

    # Censoring rate
    if case_rng.random() < tier.high_censoring_prob:
        lo, hi = tier.high_censoring_range
        censoring_rate = lo + lhs_censoring_sample * (hi - lo)
    else:
        lo, hi = tier.normal_censoring_range
        censoring_rate = lo + lhs_censoring_sample * (hi - lo)

    # Post-render degradation (tier-specific)
    _apply_post_render(case_rng, tier, modifiers)

    return modifiers, censoring_rate, y_start


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
            ThinLines(linewidth=0.9),
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
        weibull_ks=[0.7, 2.1],
        weibull_scale=35.0,
        censoring_rate=0.015,
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            Annotations(),
            CompressedTimeAxis(n_ticks=12),
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
            GaussianBlur(kernel_size=3),
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
        weibull_ks=[1.0, 1.0],
        weibull_scale=18.0,
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
        if k < 0.5 or k > 5.0:
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

    # LHS over 5 dimensions: [log_k, scale_factor, censoring_rate, n_per_arm, max_time_idx]
    lhs = _latin_hypercube(n_total, 5, rng)

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
        n_curves = int(case_rng.choice([2, 3, 4, 5], p=[0.50, 0.30, 0.15, 0.05]))
        log_k = np.log(0.6) + sample[0] * (np.log(2.2) - np.log(0.6))
        base_k = np.exp(log_k)
        weibull_ks = [
            base_k * case_rng.uniform(0.75, 1.35) for _ in range(n_curves)
        ]
        n_per_arm = int(50 + sample[3] * 450)  # 50-500
        scale_factor = 0.35 + sample[1] * 0.75

        # Tier-specific modifiers, censoring, y_start
        modifiers, censoring_rate, y_start = _apply_tier_modifiers(
            case_rng, tier, sample[2]
        )

        weibull_scale_val = scale_factor * max_time
        if n_curves <= 2:
            group_names = ["Control", "Treatment"][:n_curves]
        elif n_curves == 3:
            group_names = ["Control", "Treatment A", "Treatment B"]
        else:
            group_names = [f"Arm {chr(65 + j)}" for j in range(n_curves)]

        has_post = any(
            isinstance(m, (LowResolution, JPEGArtifacts, NoisyBackground, GaussianBlur))
            for m in modifiers
        )
        difficulty = _compute_difficulty(
            n_curves, weibull_ks, censoring_rate, y_start, n_per_arm, has_post
        )
        difficulty = max(tier.min_difficulty, difficulty)

        tc = generate_test_case(
            name=f"case_{i + 1:03d}",
            seed=case_seed,
            n_curves=n_curves,
            n_per_arm=n_per_arm,
            max_time=max_time,
            weibull_ks=weibull_ks,
            weibull_scale=weibull_scale_val,
            censoring_rate=censoring_rate,
            y_axis_start=y_start,
            group_names=group_names,
            modifiers=modifiers,
            difficulty=difficulty,
            tier=tier.name,
        )
        _generate_and_save(tc, output_dir)
        cases.append(tc)

    save_manifest(cases, output_dir)
    return cases
