"""Preset generation profiles: realistic-hard difficult and standard suites.

generate_difficult(): 5 hand-picked realistic hard scenarios.
generate_standard(): 100 realistic hard cases for stress evaluation.
"""

from __future__ import annotations

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


def _generate_and_save(
    test_case: SyntheticTestCase, output_dir: Path
) -> SyntheticTestCase:
    """Render and persist a test case."""
    case_dir = output_dir / test_case.name
    render_test_case(test_case, case_dir)
    save_test_case(test_case, case_dir)
    return test_case


# --- Difficult preset definitions ---


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
            LowResolution(target_width=760),
            JPEGArtifacts(quality=62),
        ],
        difficulty=4,
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
            ThinLines(linewidth=0.9),
            NoisyBackground(sigma=9.0),
        ],
        difficulty=4,
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
            CompressedTimeAxis(n_ticks=12),
            GridLines(major=True, alpha=0.25),
        ],
        difficulty=4,
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
            LowResolution(target_width=820),
            GaussianBlur(kernel_size=3),
        ],
        difficulty=4,
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
        censoring_rate=0.04,
        modifiers=[
            CensoringMarks(),
            LowResolution(target_width=700),
            JPEGArtifacts(quality=60),
            NoisyBackground(sigma=8.0),
        ],
        difficulty=4,
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
    """Generate 5 difficult test cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for i, preset_fn in enumerate(DIFFICULT_PRESETS):
        tc = preset_fn(seed=base_seed + i)
        _generate_and_save(tc, output_dir)
        cases.append(tc)

    save_manifest(cases, output_dir)
    return cases


# --- Standard generation (100 cases, ~60/25/15 easy/medium/hard) ---


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
    score += {1: 0, 2: 0.5, 3: 1.5, 4: 2.5}.get(n_curves, 2)

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
    """Generate 100 realistic-hard test cases."""
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

        max_time = max_time_choices[int(sample[4] * 7.99)]

        # Realistic hard-only profile: challenging but still readable.
        n_curves = case_rng.choice([2, 3, 4], p=[0.45, 0.40, 0.15])
        log_k = np.log(0.6) + sample[0] * (np.log(2.2) - np.log(0.6))
        base_k = np.exp(log_k)
        weibull_ks = [
            base_k * case_rng.uniform(0.75, 1.35) for _ in range(n_curves)
        ]
        n_per_arm = int(80 + sample[3] * 180)  # 80-260
        censoring_rate = 0.01 + sample[2] * 0.04  # 0.01-0.05
        scale_factor = 0.35 + sample[1] * 0.75
        y_start = 0.0
        modifiers: list[Modifier] = [CensoringMarks()]
        if case_rng.random() < 0.7:
            modifiers.append(RiskTableDisplay())
        if case_rng.random() < 0.2:
            y_start = float(case_rng.choice([0.2, 0.3]))
            modifiers.append(TruncatedYAxis(y_start=y_start))
        if case_rng.random() < 0.45:
            modifiers.append(GridLines(alpha=0.25))
        if case_rng.random() < 0.12:
            modifiers.append(Annotations())

        # Mild-to-moderate degradation (avoid unreadable adversarial renders).
        post_roll = case_rng.random()
        if post_roll < 0.35:
            modifiers.append(LowResolution(target_width=int(case_rng.integers(650, 900))))
        elif post_roll < 0.65:
            modifiers.append(JPEGArtifacts(quality=int(case_rng.integers(55, 76))))
        elif post_roll < 0.9:
            modifiers.append(NoisyBackground(sigma=case_rng.uniform(5, 12)))
        else:
            modifiers.append(LowResolution(target_width=int(case_rng.integers(700, 900))))
            modifiers.append(JPEGArtifacts(quality=int(case_rng.integers(60, 78))))
        if case_rng.random() < 0.15:
            modifiers.append(ThinLines(linewidth=case_rng.uniform(0.8, 1.2)))

        weibull_scale_val = scale_factor * max_time
        group_names = None
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
        difficulty = max(4, difficulty)

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
        )
        _generate_and_save(tc, output_dir)
        cases.append(tc)

    save_manifest(cases, output_dir)
    return cases
