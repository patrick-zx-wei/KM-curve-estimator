"""Preset generation profiles: generate_difficult() and generate_standard().

generate_difficult(): 5 hand-picked worst-case scenarios.
generate_standard(): 100 cases with realistic distribution (~60/25/15 easy/medium/hard).
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
        n_per_arm=120,
        max_time=48.0,
        weibull_ks=[1.0, 1.1, 1.2],
        weibull_scale=30.0,
        censoring_rate=0.02,
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            LowResolution(target_width=350),
            JPEGArtifacts(quality=30),
        ],
        difficulty=5,
    )


def _difficult_truncated_noisy(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="truncated_noisy",
        seed=seed,
        n_curves=2,
        n_per_arm=200,
        max_time=60.0,
        weibull_ks=[1.5, 2.0],
        weibull_scale=40.0,
        censoring_rate=0.015,
        y_axis_start=0.4,
        modifiers=[
            TruncatedYAxis(y_start=0.4),
            CensoringMarks(),
            ThinLines(linewidth=0.5),
            NoisyBackground(sigma=18.0),
        ],
        difficulty=5,
    )


def _difficult_crossing_compressed(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="crossing_compressed",
        seed=seed,
        n_curves=2,
        n_per_arm=180,
        max_time=60.0,
        weibull_ks=[0.5, 2.5],
        weibull_scale=35.0,
        censoring_rate=0.01,
        modifiers=[
            CensoringMarks(),
            CompressedTimeAxis(n_ticks=20),
            GridLines(major=True, alpha=0.4),
        ],
        difficulty=5,
    )


def _difficult_four_arm_lowres(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="four_arm_lowres",
        seed=seed,
        n_curves=4,
        n_per_arm=80,
        max_time=36.0,
        weibull_ks=[0.9, 1.0, 1.1, 1.15],
        weibull_scale=25.0,
        censoring_rate=0.02,
        group_names=["Arm A", "Arm B", "Arm C", "Arm D"],
        modifiers=[
            CensoringMarks(),
            RiskTableDisplay(),
            LowResolution(target_width=400),
            GaussianBlur(kernel_size=3),
        ],
        difficulty=5,
    )


def _difficult_sparse_degraded(seed: int) -> SyntheticTestCase:
    return generate_test_case(
        name="sparse_degraded",
        seed=seed,
        n_curves=2,
        n_per_arm=30,
        max_time=48.0,
        weibull_ks=[1.0, 1.0],
        weibull_scale=15.0,
        censoring_rate=0.08,  # high → ~90% censoring
        modifiers=[
            CensoringMarks(),
            JPEGArtifacts(quality=25),
            NoisyBackground(sigma=15.0),
        ],
        difficulty=5,
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
    """Generate 100 standard test cases with realistic distribution.

    Distribution: ~60 easy, ~25 medium, ~15 hard.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = 100
    rng = np.random.default_rng(base_seed)

    # LHS over 5 dimensions: [log_k, scale_factor, censoring_rate, n_per_arm, max_time_idx]
    lhs = _latin_hypercube(n_total, 5, rng)

    # Difficulty tier assignments: 60 easy, 25 medium, 15 hard
    tiers = ["easy"] * 60 + ["medium"] * 25 + ["hard"] * 15
    rng.shuffle(tiers)

    max_time_choices = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 96.0, 120.0]

    cases = []
    for i in range(n_total):
        tier = tiers[i]
        sample = lhs[i]
        case_seed = base_seed + 100 + i
        case_rng = np.random.default_rng(case_seed)

        max_time = max_time_choices[int(sample[4] * 7.99)]

        if tier == "easy":
            # 1-2 curves, k near 1.0, clean image
            n_curves = case_rng.choice([1, 2], p=[0.3, 0.7])
            log_k = np.log(0.8) + sample[0] * (np.log(1.5) - np.log(0.8))
            base_k = np.exp(log_k)
            weibull_ks = [
                base_k * case_rng.uniform(0.9, 1.1) for _ in range(n_curves)
            ]
            n_per_arm = int(100 + sample[3] * 200)  # 100-300
            censoring_rate = 0.005 + sample[2] * 0.025  # 0.005-0.03
            scale_factor = 0.4 + sample[1] * 0.8  # 0.4-1.2
            y_start = 0.0
            modifiers: list[Modifier] = []
            # Optionally add simple modifiers
            if case_rng.random() < 0.4:
                modifiers.append(CensoringMarks())
            if case_rng.random() < 0.5:
                modifiers.append(RiskTableDisplay())

        elif tier == "medium":
            # 2-3 curves, wider k range, minor visual degradation
            n_curves = case_rng.choice([2, 3], p=[0.6, 0.4])
            log_k = np.log(0.5) + sample[0] * (np.log(3.0) - np.log(0.5))
            base_k = np.exp(log_k)
            weibull_ks = [
                base_k * case_rng.uniform(0.7, 1.4) for _ in range(n_curves)
            ]
            n_per_arm = int(60 + sample[3] * 240)  # 60-300
            censoring_rate = 0.01 + sample[2] * 0.04  # 0.01-0.05
            scale_factor = 0.3 + sample[1] * 1.0
            y_start = 0.0
            modifiers = [CensoringMarks()]
            if case_rng.random() < 0.4:
                modifiers.append(GridLines())
            if case_rng.random() < 0.6:
                modifiers.append(RiskTableDisplay())
            if case_rng.random() < 0.2:
                y_start = case_rng.choice([0.3, 0.4, 0.5])
                modifiers.append(TruncatedYAxis(y_start=y_start))
            if case_rng.random() < 0.15:
                modifiers.append(Annotations())

        else:  # hard
            # 2-4 curves, extreme params, post-render degradation
            n_curves = case_rng.choice([2, 3, 4], p=[0.4, 0.35, 0.25])
            log_k = np.log(0.3) + sample[0] * (np.log(8.0) - np.log(0.3))
            base_k = np.exp(log_k)
            weibull_ks = [
                base_k * case_rng.uniform(0.5, 2.0) for _ in range(n_curves)
            ]
            n_per_arm = int(30 + sample[3] * 270)  # 30-300
            censoring_rate = 0.02 + sample[2] * 0.06  # 0.02-0.08
            scale_factor = 0.2 + sample[1] * 1.3
            y_start = 0.0
            modifiers = [CensoringMarks()]
            if case_rng.random() < 0.5:
                modifiers.append(RiskTableDisplay())
            if case_rng.random() < 0.4:
                y_start = case_rng.choice([0.3, 0.4, 0.5])
                modifiers.append(TruncatedYAxis(y_start=y_start))
            if case_rng.random() < 0.5:
                modifiers.append(GridLines(alpha=0.4))
            # Post-render degradation (at least one)
            post_roll = case_rng.random()
            if post_roll < 0.3:
                modifiers.append(LowResolution(target_width=int(case_rng.integers(300, 500))))
            elif post_roll < 0.6:
                modifiers.append(JPEGArtifacts(quality=int(case_rng.integers(25, 50))))
            elif post_roll < 0.8:
                modifiers.append(NoisyBackground(sigma=case_rng.uniform(10, 25)))
            else:
                modifiers.append(LowResolution(target_width=int(case_rng.integers(300, 450))))
                modifiers.append(JPEGArtifacts(quality=int(case_rng.integers(30, 50))))
            if case_rng.random() < 0.3:
                modifiers.append(ThinLines(linewidth=case_rng.uniform(0.3, 0.8)))

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
