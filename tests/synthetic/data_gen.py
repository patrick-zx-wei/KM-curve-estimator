"""Synthetic KM curve data generation.

Generates patient-level survival data from Weibull distributions,
computes KM step functions, risk tables, and confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from km_estimator.models.ipd_output import PatientRecord
from km_estimator.models.plot_metadata import (
    AxisConfig,
    CurveInfo,
    RiskGroup,
    RiskTable,
)

from .modifiers import Modifier

# Default color/style cycles for multi-arm plots
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
COLOR_NAMES = ["blue", "orange", "green", "red", "purple"]
LINE_STYLES = ["solid", "dashed"]
GROUP_NAMES = ["Control", "Treatment", "Arm A", "Arm B", "Arm C"]


@dataclass
class SyntheticCurveData:
    """Ground-truth data for one KM curve."""

    group_name: str
    patients: list[PatientRecord]
    step_coords: list[tuple[float, float]]
    censoring_times: list[float]
    n_at_risk: list[tuple[float, int]]
    color: str
    color_name: str
    line_style: str


@dataclass
class SyntheticTestCase:
    """A complete synthetic test scenario."""

    name: str
    seed: int
    curves: list[SyntheticCurveData]
    x_axis: AxisConfig
    y_axis: AxisConfig
    risk_table: RiskTable | None
    title: str | None
    annotations: list[str]
    modifiers: list[Modifier]
    difficulty: int = 1
    tier: str = "standard"
    image_path: str | None = None
    draft_image_path: str | None = None


def _km_from_ipd(patients: list[PatientRecord]) -> list[tuple[float, float]]:
    """Reconstruct KM curve from patient records.

    Local copy to avoid importing through km_estimator.nodes which
    eagerly loads LLM client dependencies.
    """
    if not patients:
        return [(0.0, 1.0)]

    curve: list[tuple[float, float]] = [(0.0, 1.0)]
    n_at_risk = len(patients)
    survival = 1.0

    sorted_patients = sorted(patients, key=lambda p: p.time)

    i = 0
    while i < len(sorted_patients):
        t = sorted_patients[i].time

        events = 0
        censorings = 0
        while i < len(sorted_patients) and sorted_patients[i].time == t:
            if sorted_patients[i].event:
                events += 1
            else:
                censorings += 1
            i += 1

        if events > 0 and n_at_risk > 0:
            survival *= (n_at_risk - events) / n_at_risk
            curve.append((t, survival))

        n_at_risk -= events + censorings

    return curve


def _compute_n_at_risk(
    patients: list[PatientRecord],
    time_points: list[float],
) -> list[tuple[float, int]]:
    """Compute number at risk at each time point."""
    result = []
    for t in time_points:
        n = sum(1 for p in patients if p.time >= t)
        result.append((t, n))
    return result


def _compute_greenwood_ci(
    patients: list[PatientRecord],
    eval_times: np.ndarray,
    z: float = 1.96,
) -> list[tuple[float, float, float]]:
    """Compute KM survival with Greenwood confidence intervals.

    Returns list of (survival, ci_lower, ci_upper) at each eval_time.
    """
    sorted_patients = sorted(patients, key=lambda p: p.time)
    if not sorted_patients:
        return [(1.0, 1.0, 1.0)] * len(eval_times)

    # Build event/censoring table
    event_times: list[float] = []
    n_events: list[int] = []
    n_censored: list[int] = []

    i = 0
    while i < len(sorted_patients):
        t = sorted_patients[i].time
        d = 0
        c = 0
        while i < len(sorted_patients) and sorted_patients[i].time == t:
            if sorted_patients[i].event:
                d += 1
            else:
                c += 1
            i += 1
        event_times.append(t)
        n_events.append(d)
        n_censored.append(c)

    # Compute KM + Greenwood variance at each unique event time
    n_total = len(sorted_patients)
    km_times = [0.0]
    km_survivals = [1.0]
    km_variances = [0.0]

    n_at_risk = n_total
    survival = 1.0
    variance_sum = 0.0

    for t, d, c in zip(event_times, n_events, n_censored):
        if d > 0 and n_at_risk > 0:
            survival *= (n_at_risk - d) / n_at_risk
            if n_at_risk > d:
                variance_sum += d / (n_at_risk * (n_at_risk - d))
            km_times.append(t)
            km_survivals.append(survival)
            km_variances.append(survival**2 * variance_sum)
        n_at_risk -= d + c

    # Interpolate to eval_times (step function)
    results = []
    for t in eval_times:
        # Find latest km_time <= t
        idx = 0
        for j in range(len(km_times)):
            if km_times[j] <= t:
                idx = j
            else:
                break
        s = km_survivals[idx]
        v = km_variances[idx]
        se = np.sqrt(v) if v > 0 else 0.0
        ci_lower = max(0.0, s - z * se)
        ci_upper = min(1.0, s + z * se)
        results.append((s, ci_lower, ci_upper))

    return results


def generate_survival_data(
    n_patients: int = 100,
    max_time: float = 36.0,
    weibull_k: float = 1.0,
    weibull_scale: float = 36.0,
    censoring_rate: float = 0.02,
    admin_censoring: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[list[PatientRecord], list[float]]:
    """Generate synthetic survival data from Weibull distribution.

    Returns (patients, censoring_times_on_curve).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Event times from Weibull
    event_times = rng.weibull(weibull_k, size=n_patients) * weibull_scale

    # Censoring times from exponential
    if censoring_rate > 0:
        censor_times = rng.exponential(1.0 / censoring_rate, size=n_patients)
    else:
        censor_times = np.full(n_patients, np.inf)

    # Administrative censoring
    if admin_censoring:
        censor_times = np.minimum(censor_times, max_time)

    # Observed = min(event, censor)
    observed = np.minimum(event_times, censor_times)
    events = event_times <= censor_times

    patients = []
    censoring_on_curve = []
    for t, e in zip(observed, events):
        patients.append(PatientRecord(time=float(t), event=bool(e)))
        if not e:
            censoring_on_curve.append(float(t))

    patients.sort(key=lambda p: p.time)
    censoring_on_curve.sort()

    return patients, censoring_on_curve


def _compute_risk_table_intervals(max_time: float, n_intervals: int = 6) -> list[float]:
    """Compute evenly-spaced risk table time points."""
    interval = max_time / n_intervals
    # Round to nice numbers
    if interval >= 10:
        interval = round(interval / 10) * 10
    elif interval >= 1:
        interval = round(interval)
    else:
        interval = round(interval, 1)
    if interval == 0:
        interval = max_time / n_intervals

    points = []
    t = 0.0
    while t <= max_time:
        points.append(round(t, 2))
        t += interval
    if not points or abs(points[-1] - max_time) > 1e-6:
        points.append(round(max_time, 2))
    points = sorted(set(points))
    return points


def _ensure_endpoint_tick(ticks: list[float], max_time: float) -> list[float]:
    """Ensure the axis endpoint is explicitly present in tick labels."""
    rounded_max = round(max_time, 2)
    out = [round(float(t), 2) for t in ticks]
    if not any(abs(t - rounded_max) < 1e-6 for t in out):
        out.append(rounded_max)
    return sorted(set(out))


def _step_survival_at(step_coords: list[tuple[float, float]], t: float) -> float:
    """Evaluate step-function survival at time t."""
    if not step_coords:
        return 1.0
    if t < step_coords[0][0]:
        return 1.0
    for i in range(len(step_coords) - 1, -1, -1):
        if step_coords[i][0] <= t:
            return float(step_coords[i][1])
    return float(step_coords[0][1])


def _minimum_curve_separation(curves: list[SyntheticCurveData], max_time: float) -> float:
    """Return minimum pairwise vertical separation across sampled time points."""
    if len(curves) < 2:
        return float("inf")

    sample_times = np.linspace(0.0, max_time, 48)
    min_sep = float("inf")
    for t in sample_times:
        vals = [_step_survival_at(c.step_coords, float(t)) for c in curves]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                min_sep = min(min_sep, abs(vals[i] - vals[j]))
    return min_sep


def generate_test_case(
    name: str,
    seed: int,
    n_curves: int = 2,
    n_per_arm: int = 150,
    max_time: float = 60.0,
    weibull_ks: list[float] | None = None,
    weibull_scale: float | None = None,
    censoring_rate: float = 0.02,
    y_axis_start: float = 0.0,
    group_names: list[str] | None = None,
    title: str | None = "Kaplan-Meier Survival Curve",
    annotations: list[str] | None = None,
    modifiers: list[Modifier] | None = None,
    include_risk_table: bool = True,
    difficulty: int = 1,
    tier: str = "standard",
    line_styles: list[str] | None = None,
    enforce_curve_separation: bool = False,
    min_curve_separation: float = 0.08,
    max_curve_generation_attempts: int = 6,
) -> SyntheticTestCase:
    """Generate a complete synthetic test case.

    Args:
        name: Case identifier
        seed: Random seed for reproducibility
        n_curves: Number of survival curves (arms)
        n_per_arm: Patients per arm
        max_time: Maximum follow-up time
        weibull_ks: Shape parameters per curve (controls hazard pattern)
        weibull_scale: Scale parameter (if None, defaults to 0.7 * max_time)
        censoring_rate: Exponential censoring rate
        y_axis_start: Y-axis minimum (0.0 = standard, >0 = truncated)
        group_names: Names for each curve
        title: Plot title
        annotations: Text annotations
        modifiers: Visual modifiers to apply
        include_risk_table: Whether to generate risk table data
        difficulty: Difficulty score (1-5)
        enforce_curve_separation: Retry generation until arm separation threshold is met
        min_curve_separation: Minimum vertical separation between any two curves
        max_curve_generation_attempts: Maximum retries for separation-constrained generation
    """
    rng = np.random.default_rng(seed)

    if weibull_ks is None:
        weibull_ks = [1.0] * n_curves
    if weibull_scale is None:
        weibull_scale = 0.7 * max_time
    if group_names is None:
        group_names = GROUP_NAMES[:n_curves]
    if annotations is None:
        annotations = []
    if modifiers is None:
        modifiers = []
    if line_styles is None:
        line_styles = ["solid"] * n_curves

    # Risk table time points
    risk_time_points = _compute_risk_table_intervals(max_time)

    # X-axis tick values
    x_tick_interval = risk_time_points[1] if len(risk_time_points) > 1 else max_time / 6
    x_ticks = _ensure_endpoint_tick(risk_time_points, max_time)

    attempts = (
        max(1, max_curve_generation_attempts)
        if enforce_curve_separation and n_curves > 1
        else 1
    )
    best_curves: list[SyntheticCurveData] = []
    best_risk_groups: list[RiskGroup] = []
    best_sep = -1.0

    for _ in range(attempts):
        curves: list[SyntheticCurveData] = []
        risk_groups: list[RiskGroup] = []

        for i in range(n_curves):
            patients, censoring_times = generate_survival_data(
                n_patients=n_per_arm,
                max_time=max_time,
                weibull_k=weibull_ks[i],
                weibull_scale=weibull_scale,
                censoring_rate=censoring_rate,
                admin_censoring=True,
                rng=rng,
            )

            step_coords = _km_from_ipd(patients)
            n_at_risk = _compute_n_at_risk(patients, risk_time_points)

            # Filter censoring marks to only times where the curve is drawn
            last_event_t = step_coords[-1][0] if len(step_coords) > 1 else 0.0
            censoring_times = [t for t in censoring_times if t <= last_event_t]

            # ~95% of curves get a small horizontal stub after the last event
            # (the other 5% end in a vertical cliff as an edge case)
            if len(step_coords) > 1 and rng.random() < 0.95:
                stub_len = 0.035 * max_time
                stub_end = min(last_event_t + stub_len, max_time)
                if stub_end > last_event_t:
                    step_coords.append((stub_end, step_coords[-1][1]))

            curves.append(
                SyntheticCurveData(
                    group_name=group_names[i],
                    patients=patients,
                    step_coords=step_coords,
                    censoring_times=censoring_times,
                    n_at_risk=n_at_risk,
                    color=COLORS[i % len(COLORS)],
                    color_name=COLOR_NAMES[i % len(COLOR_NAMES)],
                    line_style=line_styles[i % len(line_styles)],
                )
            )

            risk_groups.append(
                RiskGroup(
                    name=group_names[i],
                    counts=[n for _, n in n_at_risk],
                )
            )

        min_sep = _minimum_curve_separation(curves, max_time)
        if min_sep > best_sep:
            best_curves = curves
            best_risk_groups = risk_groups
            best_sep = min_sep
        if not enforce_curve_separation or min_sep >= min_curve_separation:
            break

    curves = best_curves
    risk_groups = best_risk_groups

    risk_table = None
    if include_risk_table:
        risk_table = RiskTable(
            time_points=risk_time_points,
            groups=risk_groups,
        )

    # Y-axis ticks
    y_ticks = [round(v, 1) for v in np.arange(y_axis_start, 1.01, 0.2)]
    if 1.0 not in y_ticks:
        y_ticks.append(1.0)

    x_axis = AxisConfig(
        label="Time (months)",
        start=0.0,
        end=max_time,
        tick_interval=x_tick_interval,
        tick_values=x_ticks,
        scale="linear",
    )

    y_axis = AxisConfig(
        label="Survival Probability",
        start=y_axis_start,
        end=1.0,
        tick_interval=0.2,
        tick_values=y_ticks,
        scale="linear",
    )

    return SyntheticTestCase(
        name=name,
        seed=seed,
        curves=curves,
        x_axis=x_axis,
        y_axis=y_axis,
        risk_table=risk_table,
        title=title,
        annotations=annotations,
        modifiers=modifiers,
        difficulty=difficulty,
        tier=tier,
    )
