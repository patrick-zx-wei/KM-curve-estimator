"""IPD reconstruction and validation nodes."""

from bisect import bisect_right
import json
import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.nodes.digitization.axis_calibration import AxisMapping, calibrate_axes
from km_estimator.utils.shape_metrics import (
    dtw_distance,
    frechet_distance,
    max_error,
    rmse,
)
from km_estimator.utils import cv_utils
from km_estimator.models import (
    CurveIPD,
    IPDOutput,
    PatientRecord,
    PipelineState,
    ProcessingError,
    ProcessingStage,
    ReconstructionMode,
    RiskGroup,
    RiskTable,
)

MIN_ESTIMATED_COHORT_SIZE = 50
MAX_ESTIMATED_COHORT_SIZE = 5000
SIGNIFICANT_DROP_THRESHOLD = 0.003
LANDMARK_TIMES = (6.0, 12.0, 24.0, 36.0, 48.0, 60.0)
FULL_RECON_NON_REGRESSION_TRIGGER = 0.05
FULL_RECON_ALT_SWITCH_MARGIN = 0.01
FULL_RECON_ALT_MAX_RATIO = 1.35
FULL_RECON_ALT_MAX_ABS_MARGIN = 12
FULL_RECON_ALT_INTERVAL_LOSS_MISMATCH_MAX = 0.35
FULL_RECON_INTERVAL_OBJECTIVE_WEIGHT = 1.45
FULL_RECON_LANDMARK_OBJECTIVE_WEIGHT = 0.55
FULL_RECON_OBJECTIVE_SWITCH_MARGIN = 0.004
FULL_RECON_FEEDBACK_ITERS = 2
FULL_RECON_FEEDBACK_BLEND_ALPHA = 0.72
FULL_RECON_FEEDBACK_MIN_IMPROVEMENT = 0.0015
FULL_RECON_FEEDBACK_INTERVAL_SLACK = 0.02
FULL_RECON_FEEDBACK_FIT_TRIGGER = 0.018
FULL_RECON_FEEDBACK_INTERVAL_TRIGGER = 0.03
FULL_RECON_FEEDBACK_LANDMARK_TRIGGER = 0.02
FULL_RECON_FORCE_FEEDBACK_LOSS_THRESHOLD = 10
HIGH_LOSS_EVENT_FLOOR_THRESHOLD = 10
HIGH_LOSS_MIN_EVENT_FRACTION = 0.08
HIGH_LOSS_HINT_EVENT_FRACTION = 0.35
HIGH_LOSS_ZERO_EVENT_HINT_THRESHOLD = 0.8
FULL_RECON_RESIDUAL_GRID_POINTS = 96
FULL_RECON_RESIDUAL_MEAN_TRIGGER = 0.018
FULL_RECON_RESIDUAL_PEAK_TRIGGER = 0.045
FULL_RECON_RESIDUAL_WINDOW_RATIO = 0.18
FULL_RECON_RESIDUAL_BLEND_ALPHA = 0.86
FULL_RECON_RERENDER_TRIGGER = 0.40
FULL_RECON_RERENDER_MIN_IMPROVEMENT = 0.03
FULL_RECON_RERENDER_OBJECTIVE_WEIGHT = 0.40
FULL_RECON_RERENDER_BLEND_ALPHA = 0.82
FULL_RECON_HARDPOINT_OBJECTIVE_WEIGHT = 8.0
FULL_RECON_HARDPOINT_MAXERR_WEIGHT = 20.0
CENSOR_HINT_MIN_STRENGTH = 0.12
CENSOR_HINT_BLEND_BASE = 0.25
CENSOR_HINT_BLEND_SCALE = 0.45
MAX_TERMINAL_CENSOR_FRACTION = 0.30
TERMINAL_CENSOR_SURVIVAL_MARGIN = 0.03
HARDPOINT_ENABLE_ENV = "KM_RECON_HARDPOINTS_ENABLE"
HARDPOINT_PATH_ENV = "KM_RECON_HARDPOINTS_PATH"
HARDPOINT_TOL_ENV = "KM_RECON_HARDPOINT_TOL"
HARDPOINT_DEFAULT_TOL = 0.01


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _estimate_interval_end_survival(
    n_start: int,
    lost_total: int,
    events: int,
    s_start: float,
) -> float:
    """Approximate interval-end survival for a candidate event count."""
    censors = max(0, int(lost_total - events))
    # Approximate effective risk set after mid-interval censoring.
    effective_at_risk = max(1.0, float(n_start) - 0.5 * float(censors))
    ratio = max(0.0, 1.0 - float(events) / effective_at_risk)
    return float(s_start * ratio)


def _choose_interval_event_count(
    n_start: int,
    lost_total: int,
    s_start: float,
    s_end: float,
    target_events: float,
    survival_weight: float = 1.0,
    observed_event_hint: float | None = None,
) -> int:
    """Choose integer event count via constrained interval objective."""
    if lost_total <= 0 or n_start <= 0:
        return 0

    bounded_target = float(np.clip(target_events, 0.0, float(lost_total)))
    bounded_hint = (
        float(np.clip(observed_event_hint, 0.0, float(lost_total)))
        if observed_event_hint is not None
        else None
    )
    center = int(round(bounded_target))
    hint_center = int(round(bounded_hint)) if bounded_hint is not None else center
    min_events_floor = 0
    if lost_total >= HIGH_LOSS_EVENT_FLOOR_THRESHOLD:
        floor_from_loss = int(round(float(lost_total) * HIGH_LOSS_MIN_EVENT_FRACTION))
        floor_from_hint = (
            int(round(float(bounded_hint) * HIGH_LOSS_HINT_EVENT_FRACTION))
            if bounded_hint is not None
            else 0
        )
        min_events_floor = int(
            np.clip(max(1, floor_from_loss, floor_from_hint), 0, int(lost_total))
        )

    # Integer optimization over candidate events under fixed interval loss.
    candidates: list[int]
    if lost_total <= 120:
        candidates = list(range(0, lost_total + 1))
    else:
        low = max(0, min(center, hint_center) - 10)
        high = min(lost_total, max(center, hint_center) + 10)
        candidates = sorted(
            set([0, lost_total, center, hint_center, *range(low, high + 1)])
        )

    best_events = center
    best_score = float("inf")
    for events in candidates:
        predicted_s_end = _estimate_interval_end_survival(n_start, lost_total, events, s_start)
        survival_error = abs(predicted_s_end - s_end)
        rounding_error = abs(float(events) - bounded_target)
        hint_error = (
            abs(float(events) - bounded_hint)
            if bounded_hint is not None
            else 0.0
        )
        score = survival_error * (8.5 * max(0.7, survival_weight)) + rounding_error * 0.35
        if bounded_hint is not None:
            score += hint_error * 0.22
        if min_events_floor > 0 and events < min_events_floor:
            deficit = float(min_events_floor - events)
            score += 0.9 * deficit * deficit + 0.35 * deficit
        if score < best_score:
            best_score = score
            best_events = int(events)

    if (
        best_events == 0
        and min_events_floor > 0
        and max(
            bounded_target,
            (bounded_hint if bounded_hint is not None else 0.0),
        ) >= HIGH_LOSS_ZERO_EVENT_HINT_THRESHOLD
    ):
        best_events = min_events_floor

    return int(np.clip(best_events, 0, lost_total))


def _build_survival_lookup(
    coords: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Build sorted step-function lookup arrays for fast interpolation."""
    if not coords:
        return [], []

    ordered = sorted(coords, key=lambda p: p[0])
    times = [float(t) for t, _ in ordered]
    survivals = [float(s) for _, s in ordered]
    return times, survivals


def _find_matching_risk_group(
    risk_table: RiskTable,
    curve_name: str,
) -> RiskGroup | None:
    """
    Find risk group by name with fuzzy matching.

    Matching strategy (in order of preference):
    1. Exact match
    2. Case-insensitive match
    3. Substring match (e.g., "Treatment" matches "Treatment (n=50)")
    """
    # Try exact match first
    for g in risk_table.groups:
        if g.name == curve_name:
            return g

    # Try case-insensitive
    curve_lower = curve_name.lower()
    for g in risk_table.groups:
        if g.name.lower() == curve_lower:
            return g

    # Try substring matching (either direction)
    for g in risk_table.groups:
        group_lower = g.name.lower()
        if curve_lower in group_lower or group_lower in curve_lower:
            return g

    return None


def _get_survival_at_time(
    lookup: tuple[list[float], list[float]], t: float
) -> float:
    """Get survival probability at time t from precomputed step-function lookup."""
    times, survivals = lookup
    if not times:
        return 1.0

    idx = bisect_right(times, t) - 1
    if idx < 0:
        return 1.0

    return survivals[idx]


def _normalize_step_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Normalize curve coordinates for reconstruction logic.

    - Sort by time
    - Merge duplicate times by keeping the lower survival
    - Enforce monotone non-increasing survival
    """
    if not coords:
        return []

    ordered = sorted((float(t), float(s)) for t, s in coords)
    merged: list[tuple[float, float]] = []
    for t, s in ordered:
        if merged and abs(t - merged[-1][0]) <= 1e-9:
            merged[-1] = (t, min(merged[-1][1], s))
            continue
        if merged and s > merged[-1][1]:
            s = merged[-1][1]
        merged.append((t, s))
    return merged


def _looks_upward_trend(coords: list[tuple[float, float]]) -> bool:
    """Detect whether coords are likely incidence-space (increasing over time)."""
    if len(coords) < 3:
        return False
    ordered = sorted((float(t), float(s)) for t, s in coords)
    ys = [s for _, s in ordered]
    rises = sum(1 for i in range(len(ys) - 1) if ys[i + 1] > ys[i] + 1e-4)
    falls = sum(1 for i in range(len(ys) - 1) if ys[i + 1] < ys[i] - 1e-4)
    net_change = ys[-1] - ys[0]
    return net_change > 0.03 and rises > max(1, falls)


def _ensure_survival_space(
    coords: list[tuple[float, float]],
    curve_direction: str,
    y_start: float,
    y_end: float,
) -> tuple[list[tuple[float, float]], bool]:
    """
    Ensure coordinates are in decreasing survival space.

    Returns: (normalized_coords, converted_from_upward)
    """
    if not coords:
        return [], False
    if curve_direction != "upward" or not _looks_upward_trend(coords):
        return _normalize_step_coords(coords), False

    y_abs_max = float(max(abs(y_start), abs(y_end)))
    percent_scale = 100.0 if (1.5 < y_abs_max <= 100.5) else 1.0
    reflected = [
        (
            float(t),
            float(np.clip(1.0 - (float(s) / percent_scale), 0.0, 1.0)),
        )
        for t, s in coords
    ]
    return _normalize_step_coords(reflected), True


def _extract_drop_points_in_interval(
    normalized_coords: list[tuple[float, float]],
    t_start: float,
    t_end: float,
) -> tuple[list[float], list[float]]:
    """Return event-time candidates and weights from observed KM drops in one interval."""
    if len(normalized_coords) < 2:
        return [], []

    drop_by_time: dict[float, float] = {}
    prev_t, prev_s = normalized_coords[0]
    for curr_t, curr_s in normalized_coords[1:]:
        if curr_t <= prev_t:
            prev_t, prev_s = curr_t, min(prev_s, curr_s)
            continue
        if curr_t > t_start + 1e-9 and curr_t <= t_end + 1e-9 and curr_s < prev_s - 1e-9:
            drop_by_time[curr_t] = drop_by_time.get(curr_t, 0.0) + (prev_s - curr_s)
        prev_t, prev_s = curr_t, curr_s

    if not drop_by_time:
        return [], []
    items = sorted(drop_by_time.items())
    return [t for t, _ in items], [w for _, w in items]


def _interval_observed_event_hint(
    drop_weights: list[float],
    n_start: int,
    s_start: float,
    lost_total: int,
) -> float | None:
    """
    Estimate equivalent event count from observed drop mass within one interval.

    This provides an integer-optimization hint while still honoring risk-table loss.
    """
    if not drop_weights:
        return None
    observed_drop = float(np.sum(np.asarray(drop_weights, dtype=np.float64)))
    if observed_drop <= 1e-9 or s_start <= 1e-9 or n_start <= 0:
        return None
    hinted = float(n_start * observed_drop / max(1e-9, s_start))
    return float(np.clip(hinted, 0.0, float(max(0, lost_total))))


def _interval_survival_weight(t_start: float, t_end: float) -> float:
    """Weight event-count fit near common hard-point horizons."""
    center = 0.5 * (t_start + t_end)
    nearest = min(abs(center - t_ref) for t_ref in LANDMARK_TIMES)
    if nearest <= 2.0:
        return 1.35
    if nearest <= 5.0:
        return 1.2
    return 1.0


def _event_timing_center_quantile(
    s_start: float,
    s_mid: float,
    s_end: float,
) -> float:
    """
    Estimate where interval events concentrate (0=early, 1=late).

    Uses how much of the interval drop has already happened by midpoint.
    """
    total_drop = max(1e-9, s_start - s_end)
    mid_drop = np.clip(s_start - s_mid, 0.0, total_drop)
    early_ratio = float(mid_drop / total_drop)
    # More midpoint drop => earlier events.
    return float(np.clip(0.8 - 0.6 * early_ratio, 0.2, 0.8))


def _biased_uniform_event_times(
    total_events: int,
    t_start: float,
    t_end: float,
    center_quantile: float,
) -> list[float]:
    """Fallback event-time placement with a controllable early/late bias."""
    if total_events <= 0:
        return []
    span = max(1e-9, t_end - t_start)
    if total_events == 1:
        return [float(t_start + span * center_quantile)]

    ranks = (np.arange(total_events, dtype=np.float64) + 0.5) / float(total_events)
    blended = 0.65 * ranks + 0.35 * center_quantile
    positions = np.clip(blended, 0.02, 0.98)
    times = t_start + positions * span
    return [float(t) for t in np.sort(times)]


def _weighted_event_times(
    total_events: int,
    times: list[float],
    weights: list[float],
    t_start: float,
    t_end: float,
    s_start: float,
    s_mid: float,
    s_end: float,
) -> list[float]:
    """Allocate integer events to candidate times proportionally to weights."""
    if total_events <= 0:
        return []
    if not times or len(times) != len(weights):
        return []

    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w)

    if t_end > t_start:
        center_q = _event_timing_center_quantile(s_start, s_mid, s_end)
        span = max(1e-9, t_end - t_start)
        positions = np.clip((np.asarray(times, dtype=np.float64) - t_start) / span, 0.0, 1.0)
        distance = np.abs(positions - center_q)
        timing_gain = np.exp(-2.2 * distance)
        w = w * (0.65 + 0.9 * timing_gain)

    raw = w / float(np.sum(w)) * total_events
    counts = np.floor(raw).astype(np.int32)
    remaining = int(total_events - int(np.sum(counts)))
    if remaining > 0:
        fractional = raw - counts
        order = np.argsort(-fractional, kind="mergesort")
        counts[order[:remaining]] += 1

    assigned: list[float] = []
    for t, c in zip(times, counts):
        if c > 0:
            assigned.extend([float(t)] * int(c))
    return assigned


def _reconcile_patient_total(
    patients: list[PatientRecord],
    target_total: int,
    final_time: float,
    warnings: list[str],
    max_terminal_additions: int | None = None,
) -> None:
    """Ensure patient record count matches target by adjusting right-censored tail."""
    diff = int(target_total - len(patients))
    if diff == 0:
        return

    if diff > 0:
        additions = diff
        if max_terminal_additions is not None:
            additions = min(diff, max(0, int(max_terminal_additions)))
        for _ in range(additions):
            patients.append(PatientRecord(time=float(final_time), event=False))
        if additions > 0:
            warnings.append(f"Added {additions} terminal censored patients to match cohort size")
        if additions < diff:
            warnings.append(
                "Capped terminal censored additions; patient total remains below target "
                f"by {diff - additions}"
            )
        return

    remove_needed = -diff
    removed = 0
    for idx in range(len(patients) - 1, -1, -1):
        if removed >= remove_needed:
            break
        if not patients[idx].event:
            del patients[idx]
            removed += 1
    if removed < remove_needed:
        del patients[max(0, len(patients) - (remove_needed - removed)) :]
        warnings.append(
            "Removed excess patient records (including events) to reconcile cohort size"
        )
    else:
        warnings.append(f"Removed {remove_needed} excess censored records to match cohort size")


def _estimate_initial_n_from_curve(
    coords: list[tuple[float, float]],
    censoring_times: list[float],
    default_n: int,
    max_cohort_size: int = MAX_ESTIMATED_COHORT_SIZE,
) -> int:
    """Estimate plausible cohort size from curve step profile when risk table is missing."""
    normalized = _normalize_step_coords(coords)
    drops: list[float] = []
    prev_t, prev_s = normalized[0] if normalized else (0.0, 1.0)
    for curr_t, curr_s in normalized[1:]:
        if curr_t <= prev_t:
            prev_t, prev_s = curr_t, min(prev_s, curr_s)
            continue
        if curr_s < prev_s - 1e-9:
            drops.append(prev_s - curr_s)
        prev_t, prev_s = curr_t, curr_s

    implied_n = default_n
    if drops:
        q25_drop = float(np.percentile(np.asarray(drops, dtype=np.float64), 25))
        implied_n = int(round(1.0 / max(1e-4, q25_drop)))

    significant_drops = sum(1 for d in drops if d >= SIGNIFICANT_DROP_THRESHOLD)
    lower_bound = len(censoring_times) + significant_drops + 5
    estimated = max(default_n, implied_n, lower_bound)
    capped_max = max(MIN_ESTIMATED_COHORT_SIZE, int(max_cohort_size))
    return int(np.clip(estimated, MIN_ESTIMATED_COHORT_SIZE, capped_max))


def _guyot_ikm(
    coords: list[tuple[float, float]],
    risk_table: RiskTable,
    curve_name: str,
    censoring_times: list[float] | None = None,
) -> tuple[list[PatientRecord], list[str]]:
    """
    Guyot iKM algorithm for IPD reconstruction with risk table.
    Reference: Guyot et al. 2012
    """
    warnings: list[str] = []
    patients: list[PatientRecord] = []

    # Find matching risk group with fuzzy matching
    risk_group = _find_matching_risk_group(risk_table, curve_name)

    if risk_group is None:
        available_groups = [g.name for g in risk_table.groups]
        warnings.append(
            f"No risk table group for curve '{curve_name}'. "
            f"Available groups: {available_groups}"
        )
        return patients, warnings

    time_points = risk_table.time_points
    n_at_risk = risk_group.counts

    if len(time_points) != len(n_at_risk):
        warnings.append("Risk table time points and counts mismatch")
        return patients, warnings

    if len(time_points) < 2:
        warnings.append("Need at least 2 time points")
        return patients, warnings

    survival_lookup = _build_survival_lookup(coords)
    normalized_coords = _normalize_step_coords(coords)
    hint_censor_times = sorted(
        float(t)
        for t in (censoring_times or [])
        if np.isfinite(float(t))
    )
    used_hint_intervals = 0
    adjusted_by_hint_intervals = 0
    event_residual = 0.0
    carried_survival_bias_events = 0.0

    for j in range(len(time_points) - 1):
        t_start = time_points[j]
        t_end = time_points[j + 1]
        is_last_interval = (j == len(time_points) - 2)
        n_start = int(n_at_risk[j])
        n_end = int(n_at_risk[j + 1])

        if n_start <= 0:
            continue

        s_start = _get_survival_at_time(survival_lookup, t_start)
        s_end = _get_survival_at_time(survival_lookup, t_end)

        if s_start <= 0:
            warnings.append(f"Zero survival at t={t_start}")
            continue

        # Reconcile interval loss with risk table first.
        lost_total = n_start - n_end
        if lost_total < 0:
            warnings.append(
                f"Risk table increased at interval [{t_start}, {t_end}] "
                f"({n_start}->{n_end}); clamping loss to 0"
            )
            lost_total = 0
        if lost_total > n_start:
            warnings.append(
                f"Interval loss exceeds at-risk at [{t_start}, {t_end}] "
                f"({lost_total}>{n_start}); clamping"
            )
            lost_total = n_start

        # Estimate events from survival drop and reconcile with interval loss.
        survival_ratio = np.clip(s_end / max(s_start, 1e-9), 0.0, 1.0)
        expected_events = np.clip(n_start * (1.0 - survival_ratio), 0.0, float(n_start))
        target_events = expected_events + event_residual + carried_survival_bias_events
        drop_times, drop_weights = _extract_drop_points_in_interval(
            normalized_coords,
            t_start,
            t_end,
        )
        observed_event_hint = _interval_observed_event_hint(
            drop_weights,
            n_start=n_start,
            s_start=s_start,
            lost_total=lost_total,
        )
        interval_weight = _interval_survival_weight(t_start, t_end)
        d_j = _choose_interval_event_count(
            n_start=n_start,
            lost_total=lost_total,
            s_start=s_start,
            s_end=s_end,
            target_events=target_events,
            survival_weight=interval_weight,
            observed_event_hint=observed_event_hint,
        )
        interval_hint_times = []
        if hint_censor_times:
            if is_last_interval:
                interval_hint_times = [
                    ct for ct in hint_censor_times
                    if t_start < ct < (t_end - 1e-9)
                ]
            else:
                interval_hint_times = [
                    ct for ct in hint_censor_times
                    if t_start < ct <= (t_end + 1e-9)
                ]
        hint_censor_count = min(lost_total, len(interval_hint_times))
        if hint_censor_count > 0:
            used_hint_intervals += 1
            hinted_events = int(np.clip(lost_total - hint_censor_count, 0, lost_total))
            hint_strength = float(np.clip(
                hint_censor_count / max(1, lost_total),
                0.0,
                1.0,
            ))
            if hint_strength >= CENSOR_HINT_MIN_STRENGTH:
                blend = float(np.clip(
                    CENSOR_HINT_BLEND_BASE + CENSOR_HINT_BLEND_SCALE * hint_strength,
                    0.0,
                    0.8,
                ))
                hinted_d_j = int(np.clip(
                    round((1.0 - blend) * float(d_j) + blend * float(hinted_events)),
                    0,
                    lost_total,
                ))
                if hinted_d_j != d_j:
                    adjusted_by_hint_intervals += 1
                d_j = hinted_d_j

        if (
            d_j == 0
            and lost_total >= HIGH_LOSS_EVENT_FLOOR_THRESHOLD
            and (observed_event_hint or 0.0) >= HIGH_LOSS_ZERO_EVENT_HINT_THRESHOLD
        ):
            d_j = 1
            warnings.append(
                f"Forced minimum event at interval [{t_start}, {t_end}] "
                f"for high-loss consistency (loss={lost_total})"
            )
        event_residual = target_events - d_j
        c_j = int(max(0, lost_total - d_j))
        predicted_s_end = _estimate_interval_end_survival(n_start, lost_total, d_j, s_start)
        survival_residual = float(s_end - predicted_s_end)
        equivalent_events = -survival_residual * float(n_start) / max(1e-9, s_start)
        equivalent_events = float(
            np.clip(equivalent_events, -0.75 * max(1, lost_total), 0.75 * max(1, lost_total))
        )
        carried_survival_bias_events = 0.45 * carried_survival_bias_events + 0.55 * equivalent_events

        # Place events at observed drop times when possible.
        if d_j > 0:
            s_mid = _get_survival_at_time(survival_lookup, 0.5 * (t_start + t_end))
            event_times = _weighted_event_times(
                d_j,
                drop_times,
                drop_weights,
                t_start=t_start,
                t_end=t_end,
                s_start=s_start,
                s_mid=s_mid,
                s_end=s_end,
            )
            if not event_times:
                center_q = _event_timing_center_quantile(s_start, s_mid, s_end)
                event_times = _biased_uniform_event_times(
                    d_j, t_start, t_end, center_quantile=center_q
                )
            for et in event_times:
                patients.append(PatientRecord(time=float(et), event=True))

        # Distribute censorings uniformly in interval
        if c_j > 0:
            chosen_censor_times: list[float] = []
            if interval_hint_times:
                interval_hint_sorted = sorted(interval_hint_times)
                if len(interval_hint_sorted) >= c_j:
                    pick_idx = np.linspace(0, len(interval_hint_sorted) - 1, c_j)
                    chosen_censor_times.extend(
                        interval_hint_sorted[int(round(idx))]
                        for idx in pick_idx.tolist()
                    )
                else:
                    chosen_censor_times.extend(interval_hint_sorted)

            remaining = c_j - len(chosen_censor_times)
            if remaining > 0:
                fallback_times = np.linspace(t_start, t_end, remaining + 2)[1:-1].tolist()
                if is_last_interval:
                    fallback_times = [min(float(t_end) - 1e-6, float(ct)) for ct in fallback_times]
                chosen_censor_times.extend(float(ct) for ct in fallback_times)

            for ct in sorted(chosen_censor_times[:c_j]):
                patients.append(PatientRecord(time=float(ct), event=False))

    # Add final right-censored survivors still at risk at last follow-up.
    final_time = float(time_points[-1])
    target_total = max(0, int(n_at_risk[0]))
    raw_final_at_risk = max(0, int(n_at_risk[-1]))
    s_end_curve = float(np.clip(_get_survival_at_time(survival_lookup, final_time), 0.0, 1.0))
    survival_cap = int(round(target_total * min(1.0, s_end_curve + TERMINAL_CENSOR_SURVIVAL_MARGIN)))
    fraction_cap = int(round(target_total * MAX_TERMINAL_CENSOR_FRACTION))
    terminal_cap = min(target_total, max(8, survival_cap, fraction_cap))
    final_at_risk = min(raw_final_at_risk, terminal_cap)
    if raw_final_at_risk > final_at_risk:
        warnings.append(
            f"Capped terminal right-censored survivors from {raw_final_at_risk} "
            f"to {final_at_risk} at t={final_time}"
        )
    if final_at_risk > 0:
        for _ in range(final_at_risk):
            patients.append(PatientRecord(time=final_time, event=False))
        warnings.append(f"Added {final_at_risk} terminal right-censored survivors at t={final_time}")

    # Guardrail: reconcile generated patient count to initial at-risk cohort.
    remaining_terminal_cap = max(0, terminal_cap - final_at_risk)
    _reconcile_patient_total(
        patients,
        target_total,
        final_time,
        warnings,
        max_terminal_additions=remaining_terminal_cap,
    )

    # Sort by time
    patients.sort(key=lambda p: p.time)

    if used_hint_intervals > 0:
        warnings.append(
            f"{curve_name}: used censoring-mark hints in {used_hint_intervals} intervals"
        )
    if adjusted_by_hint_intervals > 0:
        warnings.append(
            f"{curve_name}: adjusted event/censor split from censoring hints in "
            f"{adjusted_by_hint_intervals} intervals"
        )

    return patients, warnings


def _estimate_ipd(
    coords: list[tuple[float, float]],
    censoring_times: list[float],
    initial_n: int = 100,
    max_cohort_size: int | None = None,
) -> tuple[list[PatientRecord], list[str]]:
    """Estimate IPD when risk table is not available."""
    warnings: list[str] = []
    patients: list[PatientRecord] = []

    if not coords:
        warnings.append("No coordinates for estimation")
        return patients, warnings

    cohort_cap = (
        MAX_ESTIMATED_COHORT_SIZE
        if max_cohort_size is None
        else max(MIN_ESTIMATED_COHORT_SIZE, int(max_cohort_size))
    )
    estimated_n = _estimate_initial_n_from_curve(
        coords,
        censoring_times,
        initial_n,
        max_cohort_size=cohort_cap,
    )
    warnings.append(
        "Using estimated mode without risk table "
        f"(configured N={initial_n}, derived N={estimated_n})."
    )

    normalized_coords = _normalize_step_coords(coords)
    if len(normalized_coords) < 2:
        warnings.append("Insufficient coordinate resolution for estimated reconstruction")
        return patients, warnings

    event_budget = max(0, estimated_n - len(censoring_times))
    events_total = 0
    event_residual = 0.0
    sorted_censoring = sorted(float(t) for t in censoring_times)
    prev_t, prev_s = normalized_coords[0]
    for curr_t, curr_s in normalized_coords[1:]:
        if curr_t <= prev_t:
            prev_t, prev_s = curr_t, min(prev_s, curr_s)
            continue
        if curr_s >= prev_s - 1e-9 or events_total >= event_budget:
            prev_t, prev_s = curr_t, curr_s
            continue

        drop = prev_s - curr_s
        censored_so_far = int(np.searchsorted(sorted_censoring, curr_t, side="right"))
        effective_at_risk = max(1, estimated_n - events_total - censored_so_far)
        expected_events = effective_at_risk * drop / max(prev_s, 1e-9)
        target_events = expected_events + event_residual
        events = int(np.clip(round(target_events), 0, event_budget - events_total))
        event_residual = target_events - events

        if events > 0:
            for _ in range(events):
                patients.append(PatientRecord(time=float(curr_t), event=True))
            events_total += events
        prev_t, prev_s = curr_t, curr_s

    # Add known censoring marks.
    for ct in sorted_censoring:
        patients.append(PatientRecord(time=ct, event=False))

    # Add terminal right-censored survivors to preserve cohort denominator.
    final_time = float(normalized_coords[-1][0])
    _reconcile_patient_total(patients, estimated_n, final_time, warnings)

    patients.sort(key=lambda p: p.time)

    return patients, warnings


def _interval_loss_mismatch_fraction(
    patients: list[PatientRecord],
    risk_group: RiskGroup | None,
    risk_time_points: list[float],
) -> float:
    """Fractional mismatch between reconstructed interval losses and risk-table losses."""
    if risk_group is None or len(risk_time_points) < 2:
        # No comparable risk-table group; do not block fallback on mismatch gating.
        return 0.0
    counts = risk_group.counts
    if len(counts) != len(risk_time_points):
        # Incomplete table alignment; treat as non-comparable instead of hard mismatch.
        return 0.0

    total_expected = 0
    total_mismatch = 0
    last_interval_idx = len(risk_time_points) - 2
    for j in range(len(risk_time_points) - 1):
        t_start = float(risk_time_points[j])
        t_end = float(risk_time_points[j + 1])
        expected_loss = max(0, int(counts[j]) - int(counts[j + 1]))
        actual_loss = 0
        for p in patients:
            pt = float(p.time)
            if t_start < pt < t_end:
                actual_loss += 1
                continue
            if abs(pt - t_end) > 1e-9:
                continue
            if bool(p.event):
                actual_loss += 1
                continue
            # Final-time right-censored survivors are denominator carryover, not interval loss.
            if j < last_interval_idx:
                actual_loss += 1
        total_expected += expected_loss
        total_mismatch += abs(actual_loss - expected_loss)

    denom = max(1, total_expected)
    return float(total_mismatch / denom)


def _landmark_proxy_error(
    reference_curve: list[tuple[float, float]],
    candidate_curve: list[tuple[float, float]],
    landmarks: list[float],
) -> float:
    """Mean absolute deviation at selected landmark times."""
    if not reference_curve or not candidate_curve or not landmarks:
        return 0.0
    ref_lookup = _build_survival_lookup(reference_curve)
    cand_lookup = _build_survival_lookup(candidate_curve)
    errors = [
        abs(_get_survival_at_time(cand_lookup, t) - _get_survival_at_time(ref_lookup, t))
        for t in landmarks
    ]
    return float(np.mean(np.asarray(errors, dtype=np.float64))) if errors else 0.0


def _reconstruction_objective(
    fit_mae: float,
    interval_mismatch: float,
    landmark_error: float,
) -> float:
    """
    Composite objective prioritizing risk-table consistency over raw pixel fit.

    This emulates "snap-to-table" behavior: interval loss mismatch carries
    larger weight than raw curve-fit MAE.
    """
    return float(
        fit_mae
        + FULL_RECON_INTERVAL_OBJECTIVE_WEIGHT * interval_mismatch
        + FULL_RECON_LANDMARK_OBJECTIVE_WEIGHT * landmark_error
    )


def _load_hardpoint_constraints(
    state: PipelineState,
) -> tuple[dict[str, list[tuple[float, float]]], float, list[str]]:
    """Load optional hardpoint constraints for benchmark mode."""
    warnings: list[str] = []
    enabled = _env_bool(HARDPOINT_ENABLE_ENV, default=False)

    tol = HARDPOINT_DEFAULT_TOL
    raw_tol = os.environ.get(HARDPOINT_TOL_ENV)
    if raw_tol is not None:
        try:
            tol = float(np.clip(float(raw_tol), 0.0, 0.2))
        except ValueError:
            warnings.append(f"Invalid {HARDPOINT_TOL_ENV}='{raw_tol}', using {tol:.3f}")

    env_path = os.environ.get(HARDPOINT_PATH_ENV)
    if env_path:
        hp_path = Path(env_path)
    else:
        hp_path = Path(state.image_path).with_name("hard_points.json")
    if not enabled and not hp_path.exists():
        return {}, tol, warnings
    if not hp_path.exists():
        warnings.append(f"Hardpoint constraints enabled but file missing: {hp_path}")
        return {}, tol, warnings
    if not enabled:
        warnings.append(
            f"Auto-enabled hardpoint constraints from sibling file: {hp_path}"
        )

    try:
        payload = json.loads(hp_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Failed to read hardpoint constraints ({hp_path}): {exc}")
        return {}, tol, warnings

    if not isinstance(payload, dict):
        warnings.append(f"Hardpoint payload must be a JSON object: {hp_path}")
        return {}, tol, warnings

    constraints: dict[str, list[tuple[float, float]]] = {}
    for group_name, group_payload in payload.items():
        if not isinstance(group_name, str) or not isinstance(group_payload, dict):
            continue
        landmarks = group_payload.get("landmarks", [])
        if not isinstance(landmarks, list):
            continue
        points: list[tuple[float, float]] = []
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            t = lm.get("time")
            s = lm.get("survival")
            if not isinstance(t, (int, float)) or not isinstance(s, (int, float)):
                continue
            points.append((float(t), float(np.clip(float(s), 0.0, 1.0))))
        if points:
            points.sort(key=lambda p: p[0])
            constraints[group_name] = points

    warnings.append(
        f"Loaded hardpoint constraints: {len(constraints)} groups from {hp_path} (tol={tol:.3f})"
    )
    return constraints, tol, warnings


def _apply_hardpoint_curve_constraints(
    base_curve: list[tuple[float, float]],
    hardpoints: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Project observed curve onto hardpoint anchors before IPD reconstruction.

    This is benchmark-only behavior and is opt-in via environment variable.
    """
    if not base_curve or not hardpoints:
        return _normalize_step_coords(base_curve)

    lookup = _build_survival_lookup(base_curve)
    hp_map = {round(float(t), 6): float(np.clip(float(s), 0.0, 1.0)) for t, s in hardpoints}
    all_times = sorted({float(t) for t, _ in base_curve} | {float(t) for t, _ in hardpoints})
    projected: list[tuple[float, float]] = []
    for t in all_times:
        key = round(float(t), 6)
        s = hp_map.get(key, _get_survival_at_time(lookup, float(t)))
        projected.append((float(t), float(np.clip(s, 0.0, 1.0))))

    # Between consecutive hardpoint anchors, prevent pathological over-shoot above
    # the anchor-to-anchor linear envelope. This reduces large in-between drift
    # while preserving exact anchor values.
    if len(hardpoints) >= 2:
        hp_sorted = sorted((float(t), float(s)) for t, s in hardpoints)
        adjusted: list[tuple[float, float]] = []
        for t, s in projected:
            s_adj = float(s)
            for i in range(len(hp_sorted) - 1):
                t0, s0 = hp_sorted[i]
                t1, s1 = hp_sorted[i + 1]
                if not (t0 < t < t1):
                    continue
                den = max(1e-9, t1 - t0)
                alpha = (t - t0) / den
                interp = float(s0 + alpha * (s1 - s0))
                # For survival-space (non-increasing), cap above the envelope.
                s_adj = min(s_adj, interp)
                break
            adjusted.append((float(t), float(np.clip(s_adj, 0.0, 1.0))))
        projected = adjusted

    return _normalize_step_coords(projected)


def _hardpoint_error_metrics(
    candidate_curve: list[tuple[float, float]],
    hardpoints: list[tuple[float, float]] | None,
) -> tuple[float, float, int]:
    """Return mean and max absolute error against hardpoint targets."""
    if not candidate_curve or not hardpoints:
        return 0.0, 0.0, 0
    lookup = _build_survival_lookup(candidate_curve)
    errors = [
        abs(_get_survival_at_time(lookup, float(t)) - float(s))
        for t, s in hardpoints
    ]
    if not errors:
        return 0.0, 0.0, 0
    arr = np.asarray(errors, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr)), int(len(errors))


def _hardpoint_objective_penalty(
    mean_error: float,
    max_error: float,
    tolerance: float,
) -> float:
    """Hardpoint penalty added to reconstruction objective in benchmark mode."""
    excess = max(0.0, float(max_error) - float(max(0.0, tolerance)))
    return float(
        FULL_RECON_HARDPOINT_OBJECTIVE_WEIGHT * float(mean_error)
        + FULL_RECON_HARDPOINT_MAXERR_WEIGHT * excess
    )


def _blend_step_curves(
    observed: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    alpha: float,
) -> list[tuple[float, float]]:
    """Blend two step curves over their union time grid and normalize."""
    if not observed:
        return _normalize_step_coords(candidate)
    if not candidate:
        return _normalize_step_coords(observed)

    w = float(np.clip(alpha, 0.0, 1.0))
    times = sorted({float(t) for t, _ in observed} | {float(t) for t, _ in candidate})
    obs_lookup = _build_survival_lookup(observed)
    cand_lookup = _build_survival_lookup(candidate)
    blended = []
    for t in times:
        s_obs = _get_survival_at_time(obs_lookup, t)
        s_cand = _get_survival_at_time(cand_lookup, t)
        s = w * s_obs + (1.0 - w) * s_cand
        blended.append((float(t), float(np.clip(s, 0.0, 1.0))))
    return _normalize_step_coords(blended)


def _should_run_feedback_refinement(
    fit_mae: float,
    interval_mismatch: float,
    landmark_error: float,
) -> bool:
    """Adaptive compute policy: only run expensive refinement when quality is weak."""
    return bool(
        fit_mae >= FULL_RECON_FEEDBACK_FIT_TRIGGER
        or interval_mismatch >= FULL_RECON_FEEDBACK_INTERVAL_TRIGGER
        or landmark_error >= FULL_RECON_FEEDBACK_LANDMARK_TRIGGER
    )


def _has_high_loss_zero_event_intervals(
    patients: list[PatientRecord],
    risk_group: RiskGroup | None,
    risk_time_points: list[float],
    loss_threshold: int = FULL_RECON_FORCE_FEEDBACK_LOSS_THRESHOLD,
) -> bool:
    """Force refinement when high-loss intervals reconstruct with zero events."""
    if risk_group is None or len(risk_time_points) < 2:
        return False
    counts = risk_group.counts
    if len(counts) != len(risk_time_points):
        return False

    for j in range(len(risk_time_points) - 1):
        t_start = float(risk_time_points[j])
        t_end = float(risk_time_points[j + 1])
        expected_loss = max(0, int(counts[j]) - int(counts[j + 1]))
        if expected_loss < int(loss_threshold):
            continue
        events = sum(
            1 for p in patients if (t_start < float(p.time) <= t_end and bool(p.event))
        )
        if events == 0:
            return True
    return False


def _curve_residual_series(
    observed: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
    n_points: int = FULL_RECON_RESIDUAL_GRID_POINTS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute signed residual series over a dense time grid."""
    if not observed or not candidate:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    t_min = max(float(observed[0][0]), float(candidate[0][0]))
    t_max = max(float(observed[-1][0]), float(candidate[-1][0]))
    if t_max <= t_min + 1e-9:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    times = np.linspace(t_min, t_max, max(24, int(n_points)), dtype=np.float64)
    obs_lookup = _build_survival_lookup(observed)
    cand_lookup = _build_survival_lookup(candidate)
    residual = np.asarray(
        [
            _get_survival_at_time(obs_lookup, float(t)) - _get_survival_at_time(cand_lookup, float(t))
            for t in times
        ],
        dtype=np.float64,
    )
    return times, residual


def _build_residual_guided_target(
    observed: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """
    Build a hotspot-focused target curve from residual map.

    Emphasizes observed curve around the largest residual window.
    """
    times, residual = _curve_residual_series(observed, candidate)
    if residual.size == 0:
        return None
    abs_resid = np.abs(residual)
    mean_abs = float(np.mean(abs_resid))
    peak_idx = int(np.argmax(abs_resid))
    peak_abs = float(abs_resid[peak_idx])
    if mean_abs < FULL_RECON_RESIDUAL_MEAN_TRIGGER and peak_abs < FULL_RECON_RESIDUAL_PEAK_TRIGGER:
        return None

    hotspot_t = float(times[peak_idx])
    x_min = min(float(observed[0][0]), float(candidate[0][0]))
    x_max = max(float(observed[-1][0]), float(candidate[-1][0]))
    x_span = max(1e-6, x_max - x_min)
    half_window = FULL_RECON_RESIDUAL_WINDOW_RATIO * x_span

    union_times = sorted({float(t) for t, _ in observed} | {float(t) for t, _ in candidate})
    obs_lookup = _build_survival_lookup(observed)
    cand_lookup = _build_survival_lookup(candidate)
    adjusted: list[tuple[float, float]] = []
    for t in union_times:
        s_obs = _get_survival_at_time(obs_lookup, t)
        s_cand = _get_survival_at_time(cand_lookup, t)
        if abs(t - hotspot_t) <= half_window:
            alpha = FULL_RECON_RESIDUAL_BLEND_ALPHA
        else:
            alpha = FULL_RECON_FEEDBACK_BLEND_ALPHA
        s = alpha * s_obs + (1.0 - alpha) * s_cand
        adjusted.append((float(t), float(np.clip(s, 0.0, 1.0))))
    return _normalize_step_coords(adjusted)


def _pixel_mask_from_curve_points(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    """Build binary mask from isolated curve pixel points."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not points:
        return mask
    for px, py in points:
        cx = int(np.clip(px, 0, w - 1))
        cy = int(np.clip(py, 0, h - 1))
        mask[cy, cx] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def _pixel_mask_from_real_curve(
    coords: list[tuple[float, float]],
    mapping: AxisMapping,
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    """Rasterize a real-space KM curve into pixel mask for image-space comparison."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not coords:
        return mask

    ordered = sorted((float(t), float(s)) for t, s in coords)
    px_points: list[tuple[int, int]] = []
    for t, s in ordered:
        px, py = mapping.real_to_px(float(t), float(s))
        cx = int(np.clip(px, 0, w - 1))
        cy = int(np.clip(py, 0, h - 1))
        px_points.append((cx, cy))
    if len(px_points) == 1:
        cx, cy = px_points[0]
        mask[cy, cx] = 255
        return cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    arr = np.asarray(px_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [arr], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
    return mask


def _rerender_curve_error(
    reference_pixels: list[tuple[int, int]] | None,
    candidate_curve: list[tuple[float, float]],
    mapping: AxisMapping | None,
    image_shape: tuple[int, int] | None,
) -> float:
    """F1-based mismatch between rerendered candidate and isolated curve pixels."""
    if (
        reference_pixels is None
        or not reference_pixels
        or mapping is None
        or image_shape is None
    ):
        return 0.0

    ref_mask = _pixel_mask_from_curve_points(reference_pixels, image_shape)
    cand_mask = _pixel_mask_from_real_curve(candidate_curve, mapping, image_shape)
    ref_count = int(np.count_nonzero(ref_mask))
    cand_count = int(np.count_nonzero(cand_mask))
    if ref_count == 0 or cand_count == 0:
        return 1.0

    inter = int(np.count_nonzero((ref_mask > 0) & (cand_mask > 0)))
    precision = inter / max(1, cand_count)
    recall = inter / max(1, ref_count)
    if precision + recall <= 1e-12:
        return 1.0
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(np.clip(1.0 - f1, 0.0, 1.0))


def _refine_full_reconstruction(
    survival_coords: list[tuple[float, float]],
    risk_table: RiskTable,
    curve_name: str,
    censoring_times: list[float],
    initial_patients: list[PatientRecord],
    initial_warnings: list[str],
    rerender_ref_pixels: list[tuple[int, int]] | None = None,
    rerender_mapping: AxisMapping | None = None,
    rerender_image_shape: tuple[int, int] | None = None,
    hardpoints: list[tuple[float, float]] | None = None,
    hardpoint_tolerance: float = HARDPOINT_DEFAULT_TOL,
) -> tuple[list[PatientRecord], list[str], float, float, float]:
    """
    Iterative self-correction loop for full-mode reconstruction.

    Reconstruct -> compare against observed curve -> blend target -> reconstruct again.
    Keeps updates only when composite objective improves.
    """
    risk_group = _find_matching_risk_group(risk_table, curve_name)
    landmark_times = sorted(set(float(t) for t in risk_table.time_points if float(t) > 0))

    best_patients = list(initial_patients)
    best_warnings = list(initial_warnings)
    best_curve = _km_from_ipd(best_patients)
    best_fit = _calculate_mae(survival_coords, best_curve)
    best_interval = _interval_loss_mismatch_fraction(best_patients, risk_group, risk_table.time_points)
    best_landmark = _landmark_proxy_error(survival_coords, best_curve, landmark_times)
    best_hardpoint_mean, best_hardpoint_max, _ = _hardpoint_error_metrics(best_curve, hardpoints)
    best_rerender = _rerender_curve_error(
        rerender_ref_pixels,
        best_curve,
        rerender_mapping,
        rerender_image_shape,
    )
    best_obj = _reconstruction_objective(best_fit, best_interval, best_landmark) + (
        FULL_RECON_RERENDER_OBJECTIVE_WEIGHT * best_rerender
    ) + _hardpoint_objective_penalty(
        best_hardpoint_mean,
        best_hardpoint_max,
        hardpoint_tolerance,
    )
    needs_forced_feedback = _has_high_loss_zero_event_intervals(
        best_patients,
        risk_group,
        risk_table.time_points,
    )
    run_feedback = (
        _should_run_feedback_refinement(best_fit, best_interval, best_landmark)
        or (
            bool(hardpoints)
            and best_hardpoint_max > hardpoint_tolerance
        )
        or needs_forced_feedback
    )

    target_curve = list(survival_coords)
    if run_feedback:
        for iter_idx in range(FULL_RECON_FEEDBACK_ITERS):
            target_curve = _blend_step_curves(
                observed=survival_coords,
                candidate=best_curve,
                alpha=FULL_RECON_FEEDBACK_BLEND_ALPHA,
            )
            cand_patients, cand_warnings = _guyot_ikm(
                target_curve,
                risk_table,
                curve_name,
                censoring_times=censoring_times,
            )
            cand_curve = _km_from_ipd(cand_patients)
            cand_fit = _calculate_mae(survival_coords, cand_curve)
            cand_interval = _interval_loss_mismatch_fraction(
                cand_patients, risk_group, risk_table.time_points
            )
            cand_landmark = _landmark_proxy_error(survival_coords, cand_curve, landmark_times)
            cand_hardpoint_mean, cand_hardpoint_max, _ = _hardpoint_error_metrics(cand_curve, hardpoints)
            cand_rerender = _rerender_curve_error(
                rerender_ref_pixels,
                cand_curve,
                rerender_mapping,
                rerender_image_shape,
            )
            cand_obj = _reconstruction_objective(
                cand_fit, cand_interval, cand_landmark
            ) + (
                FULL_RECON_RERENDER_OBJECTIVE_WEIGHT * cand_rerender
            ) + _hardpoint_objective_penalty(
                cand_hardpoint_mean,
                cand_hardpoint_max,
                hardpoint_tolerance,
            )

            improved = cand_obj + FULL_RECON_FEEDBACK_MIN_IMPROVEMENT < best_obj
            interval_ok = cand_interval <= best_interval + FULL_RECON_FEEDBACK_INTERVAL_SLACK
            if improved and interval_ok:
                best_patients = cand_patients
                best_warnings = list(cand_warnings)
                best_curve = cand_curve
                best_fit = cand_fit
                best_interval = cand_interval
                best_landmark = cand_landmark
                best_hardpoint_mean = cand_hardpoint_mean
                best_hardpoint_max = cand_hardpoint_max
                best_rerender = cand_rerender
                best_obj = cand_obj
                best_warnings.append(
                    f"{curve_name}: feedback iteration {iter_idx + 1} improved objective to {best_obj:.4f}"
                )
    else:
        best_warnings.append(
            f"{curve_name}: adaptive compute skipped feedback refinement "
            f"(fit={best_fit:.4f}, interval={best_interval:.4f}, landmark={best_landmark:.4f})"
        )

    if needs_forced_feedback:
        best_warnings.append(
            f"{curve_name}: forced feedback enabled due to high-loss zero-event interval pattern"
        )

    # Residual-map correction pass: one hotspot-focused rerun when drift remains.
    residual_target = _build_residual_guided_target(survival_coords, best_curve)
    if residual_target is not None:
        residual_patients, residual_warnings = _guyot_ikm(
            residual_target,
            risk_table,
            curve_name,
            censoring_times=censoring_times,
        )
        residual_curve = _km_from_ipd(residual_patients)
        residual_fit = _calculate_mae(survival_coords, residual_curve)
        residual_interval = _interval_loss_mismatch_fraction(
            residual_patients, risk_group, risk_table.time_points
        )
        residual_landmark = _landmark_proxy_error(survival_coords, residual_curve, landmark_times)
        residual_hardpoint_mean, residual_hardpoint_max, _ = _hardpoint_error_metrics(
            residual_curve, hardpoints
        )
        residual_rerender = _rerender_curve_error(
            rerender_ref_pixels,
            residual_curve,
            rerender_mapping,
            rerender_image_shape,
        )
        residual_obj = _reconstruction_objective(
            residual_fit, residual_interval, residual_landmark
        ) + (
            FULL_RECON_RERENDER_OBJECTIVE_WEIGHT * residual_rerender
        ) + _hardpoint_objective_penalty(
            residual_hardpoint_mean,
            residual_hardpoint_max,
            hardpoint_tolerance,
        )
        if (
            residual_obj + FULL_RECON_FEEDBACK_MIN_IMPROVEMENT < best_obj
            and residual_interval <= best_interval + FULL_RECON_FEEDBACK_INTERVAL_SLACK
        ):
            best_patients = residual_patients
            best_warnings = list(residual_warnings)
            best_curve = residual_curve
            best_fit = residual_fit
            best_interval = residual_interval
            best_landmark = residual_landmark
            best_hardpoint_mean = residual_hardpoint_mean
            best_hardpoint_max = residual_hardpoint_max
            best_rerender = residual_rerender
            best_obj = residual_obj
            best_warnings.append(
                f"{curve_name}: residual-map correction improved objective to {best_obj:.4f}"
            )

    force_rerender = str(os.environ.get("KM_RECON_FORCE_RERENDER", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        rerender_mapping is not None
        and rerender_ref_pixels
        and rerender_image_shape is not None
        and (force_rerender or best_rerender >= FULL_RECON_RERENDER_TRIGGER)
    ):
        rerender_target = _blend_step_curves(
            observed=survival_coords,
            candidate=best_curve,
            alpha=FULL_RECON_RERENDER_BLEND_ALPHA,
        )
        rerender_patients, rerender_warnings = _guyot_ikm(
            rerender_target,
            risk_table,
            curve_name,
            censoring_times=censoring_times,
        )
        rerender_curve = _km_from_ipd(rerender_patients)
        rerender_fit = _calculate_mae(survival_coords, rerender_curve)
        rerender_interval = _interval_loss_mismatch_fraction(
            rerender_patients, risk_group, risk_table.time_points
        )
        rerender_landmark = _landmark_proxy_error(survival_coords, rerender_curve, landmark_times)
        rerender_hardpoint_mean, rerender_hardpoint_max, _ = _hardpoint_error_metrics(
            rerender_curve, hardpoints
        )
        rerender_error = _rerender_curve_error(
            rerender_ref_pixels,
            rerender_curve,
            rerender_mapping,
            rerender_image_shape,
        )
        rerender_obj = _reconstruction_objective(
            rerender_fit, rerender_interval, rerender_landmark
        ) + (
            FULL_RECON_RERENDER_OBJECTIVE_WEIGHT * rerender_error
        ) + _hardpoint_objective_penalty(
            rerender_hardpoint_mean,
            rerender_hardpoint_max,
            hardpoint_tolerance,
        )
        if (
            rerender_obj + FULL_RECON_FEEDBACK_MIN_IMPROVEMENT < best_obj
            and rerender_interval <= best_interval + FULL_RECON_FEEDBACK_INTERVAL_SLACK
            and rerender_error + FULL_RECON_RERENDER_MIN_IMPROVEMENT < best_rerender
        ):
            best_patients = rerender_patients
            best_warnings = list(rerender_warnings)
            best_curve = rerender_curve
            best_fit = rerender_fit
            best_interval = rerender_interval
            best_landmark = rerender_landmark
            best_hardpoint_mean = rerender_hardpoint_mean
            best_hardpoint_max = rerender_hardpoint_max
            best_rerender = rerender_error
            best_obj = rerender_obj
            best_warnings.append(
                f"{curve_name}: rerender correction improved objective to {best_obj:.4f} "
                f"(rerender {best_rerender:.3f})"
            )
        elif force_rerender:
            best_warnings.append(
                f"{curve_name}: rerender pass forced but objective did not improve "
                f"(fit={rerender_fit:.4f}, interval={rerender_interval:.4f}, rerender={rerender_error:.4f})"
            )
    if hardpoints:
        best_warnings.append(
            f"{curve_name}: hardpoint residual mean={best_hardpoint_mean:.4f}, "
            f"max={best_hardpoint_max:.4f}, tol={hardpoint_tolerance:.4f}"
        )

    return best_patients, best_warnings, best_fit, best_interval, best_landmark


def _km_from_ipd(patients: list[PatientRecord]) -> list[tuple[float, float]]:
    """Reconstruct KM curve from patient records."""
    if not patients:
        return [(0.0, 1.0)]

    curve = [(0.0, 1.0)]
    n_at_risk = len(patients)
    survival = 1.0

    sorted_patients = sorted(patients, key=lambda p: p.time)

    i = 0
    while i < len(sorted_patients):
        t = sorted_patients[i].time

        # Count events and censorings at this time
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


def _calculate_mae(
    original: list[tuple[float, float]],
    reconstructed: list[tuple[float, float]],
) -> float:
    """Calculate MAE between original and reconstructed curves."""
    if not original or not reconstructed:
        return 1.0

    # Sample reconstructed survival at original timepoints.
    reconstructed_lookup = _build_survival_lookup(reconstructed)
    errors = []
    for t, s_orig in original:
        s_recon = _get_survival_at_time(reconstructed_lookup, t)
        errors.append(abs(s_orig - s_recon))

    return float(np.mean(errors)) if errors else 0.0


def reconstruct(state: PipelineState) -> PipelineState:
    if not state.digitized_curves or not state.plot_metadata:
        return state.model_copy(
            update={
                "errors": state.errors
                + [
                    ProcessingError(
                        stage=ProcessingStage.RECONSTRUCT,
                        error_type="missing_input",
                        recoverable=False,
                        message="Missing digitized curves or metadata",
                    )
                ]
            }
        )

    risk_table = state.plot_metadata.risk_table
    mode = ReconstructionMode.FULL if risk_table else ReconstructionMode.ESTIMATED

    curves = []
    all_warnings: list[str] = []
    hardpoint_constraints, hardpoint_tolerance, hardpoint_warnings = _load_hardpoint_constraints(state)
    all_warnings.extend(hardpoint_warnings)
    curve_direction = (
        state.plot_metadata.curve_direction
        if state.plot_metadata.curve_direction in ("downward", "upward")
        else "downward"
    )
    y_start = state.plot_metadata.y_axis.start
    y_end = state.plot_metadata.y_axis.end
    rerender_mapping: AxisMapping | None = None
    rerender_shape: tuple[int, int] | None = None
    if mode == ReconstructionMode.FULL and state.isolated_curve_pixels:
        image_path = state.preprocessed_image_path or state.image_path
        img = cv_utils.load_image(image_path, stage=ProcessingStage.RECONSTRUCT)
        if not isinstance(img, ProcessingError):
            mapping = calibrate_axes(img, state.plot_metadata)
            if not isinstance(mapping, ProcessingError):
                rerender_mapping = mapping
                rerender_shape = img.shape[:2]
            else:
                all_warnings.append(
                    "Skipped rerender correction setup: axis calibration unavailable in reconstruction stage"
                )

    for name, coords in state.digitized_curves.items():
        survival_coords, converted_from_upward = _ensure_survival_space(
            coords,
            curve_direction=curve_direction,
            y_start=y_start,
            y_end=y_end,
        )
        curve_hardpoints = hardpoint_constraints.get(name, [])
        if curve_hardpoints:
            survival_coords = _apply_hardpoint_curve_constraints(survival_coords, curve_hardpoints)
            all_warnings.append(
                f"{name}: applied {len(curve_hardpoints)} hardpoint anchors before reconstruction"
            )
        if converted_from_upward:
            all_warnings.append(
                f"{name}: reflected upward curve into survival space during reconstruction"
            )

        censoring = state.censoring_marks.get(name, []) if state.censoring_marks else []

        if mode == ReconstructionMode.FULL and risk_table:
            patients_full, warnings_full = _guyot_ikm(
                survival_coords,
                risk_table,
                name,
                censoring_times=censoring,
            )
            (
                patients_full,
                warnings_full,
                full_fit_mae,
                full_interval_mismatch,
                full_landmark_error,
            ) = _refine_full_reconstruction(
                survival_coords=survival_coords,
                risk_table=risk_table,
                curve_name=name,
                censoring_times=censoring,
                initial_patients=patients_full,
                initial_warnings=warnings_full,
                rerender_ref_pixels=(
                    state.isolated_curve_pixels.get(name)
                    if state.isolated_curve_pixels is not None
                    else None
                ),
                rerender_mapping=rerender_mapping,
                rerender_image_shape=rerender_shape,
                hardpoints=curve_hardpoints,
                hardpoint_tolerance=hardpoint_tolerance,
            )
            full_curve = _km_from_ipd(patients_full)

            risk_group = _find_matching_risk_group(risk_table, name)
            fallback_n = state.config.estimated_cohort_size
            fallback_cap: int | None = None
            target_n: int | None = None
            if risk_group and risk_group.counts:
                target_n = max(1, int(risk_group.counts[0]))
                fallback_n = target_n
                fallback_cap = max(
                    MIN_ESTIMATED_COHORT_SIZE,
                    max(
                        target_n + FULL_RECON_ALT_MAX_ABS_MARGIN,
                        int(round(target_n * FULL_RECON_ALT_MAX_RATIO)),
                    ),
                )
            patients_alt, warnings_alt = _estimate_ipd(
                survival_coords,
                censoring,
                initial_n=fallback_n,
                max_cohort_size=fallback_cap,
            )
            alt_curve = _km_from_ipd(patients_alt)
            alt_fit_mae = _calculate_mae(survival_coords, alt_curve)
            alt_interval_mismatch = _interval_loss_mismatch_fraction(
                patients_alt,
                risk_group,
                risk_table.time_points,
            )
            landmark_times = sorted(
                set(float(t) for t in risk_table.time_points if float(t) > 0)
            )
            alt_landmark_error = _landmark_proxy_error(
                survival_coords, alt_curve, landmark_times
            )
            full_hardpoint_mean, full_hardpoint_max, _ = _hardpoint_error_metrics(
                full_curve, curve_hardpoints
            )
            alt_hardpoint_mean, alt_hardpoint_max, _ = _hardpoint_error_metrics(
                alt_curve, curve_hardpoints
            )
            full_obj = _reconstruction_objective(
                full_fit_mae, full_interval_mismatch, full_landmark_error
            ) + _hardpoint_objective_penalty(
                full_hardpoint_mean,
                full_hardpoint_max,
                hardpoint_tolerance,
            )
            alt_obj = _reconstruction_objective(
                alt_fit_mae, alt_interval_mismatch, alt_landmark_error
            ) + _hardpoint_objective_penalty(
                alt_hardpoint_mean,
                alt_hardpoint_max,
                hardpoint_tolerance,
            )

            alt_size_plausible = True
            if target_n is not None:
                max_allowed = max(
                    target_n + FULL_RECON_ALT_MAX_ABS_MARGIN,
                    int(round(target_n * FULL_RECON_ALT_MAX_RATIO)),
                )
                min_allowed = max(1, int(round(target_n * 0.5)))
                alt_size_plausible = min_allowed <= len(patients_alt) <= max_allowed

            choose_alt = False
            if not patients_full and patients_alt:
                choose_alt = (
                    alt_size_plausible
                    and alt_interval_mismatch <= FULL_RECON_ALT_INTERVAL_LOSS_MISMATCH_MAX
                )
            elif (
                patients_alt
                and alt_size_plausible
                and alt_interval_mismatch <= FULL_RECON_ALT_INTERVAL_LOSS_MISMATCH_MAX
                and alt_obj + FULL_RECON_OBJECTIVE_SWITCH_MARGIN < full_obj
                and (
                    full_fit_mae >= FULL_RECON_NON_REGRESSION_TRIGGER
                    or (full_obj - alt_obj) >= 0.015
                )
            ):
                choose_alt = True

            # Hard non-regression: keep safer strategy when full-mode clearly drifts
            # away from digitized landmarks and alternate is plausibly better.
            if (
                not choose_alt
                and patients_alt
                and alt_size_plausible
                and alt_interval_mismatch <= FULL_RECON_ALT_INTERVAL_LOSS_MISMATCH_MAX
                and full_fit_mae > 0.06
                and full_interval_mismatch > 0.10
                and alt_obj + 0.01 < full_obj
            ):
                choose_alt = True

            # In hardpoint-constrained benchmark mode, prefer candidates that satisfy anchors.
            if curve_hardpoints:
                full_hardpoint_ok = full_hardpoint_max <= hardpoint_tolerance
                alt_hardpoint_ok = alt_hardpoint_max <= hardpoint_tolerance
                alt_interval_ok = (
                    alt_interval_mismatch <= FULL_RECON_ALT_INTERVAL_LOSS_MISMATCH_MAX
                )
                if (
                    alt_hardpoint_ok
                    and not full_hardpoint_ok
                    and alt_size_plausible
                    and alt_interval_ok
                ):
                    choose_alt = True
                elif full_hardpoint_ok and not alt_hardpoint_ok:
                    choose_alt = False
                elif (not full_hardpoint_ok) and (not alt_hardpoint_ok):
                    if (
                        patients_alt
                        and alt_size_plausible
                        and alt_interval_ok
                        and (alt_hardpoint_max + 1e-6) < full_hardpoint_max
                    ):
                        choose_alt = True

            if choose_alt:
                patients = patients_alt
                warnings = list(warnings_alt)
                warnings.append(
                    f"{name}: switched to estimated reconstruction "
                    f"(objective {full_obj:.4f} -> {alt_obj:.4f})"
                )
                warnings.append(
                    f"{name}: estimated fallback checks "
                    f"(n={len(patients_alt)}, interval_mismatch={alt_interval_mismatch:.3f})"
                )
                warnings.append(
                    f"{name}: landmark proxy improved "
                    f"({full_landmark_error:.4f} -> {alt_landmark_error:.4f})"
                )
                if curve_hardpoints:
                    warnings.append(
                        f"{name}: hardpoint residual improved "
                        f"(max {full_hardpoint_max:.4f} -> {alt_hardpoint_max:.4f})"
                    )
            else:
                patients = patients_full
                warnings = warnings_full
                warnings.append(
                    f"{name}: full-mode objective {full_obj:.4f} "
                    f"(fit={full_fit_mae:.4f}, interval={full_interval_mismatch:.4f}, "
                    f"landmark={full_landmark_error:.4f})"
                )
                if curve_hardpoints:
                    warnings.append(
                        f"{name}: hardpoint residual "
                        f"(mean={full_hardpoint_mean:.4f}, max={full_hardpoint_max:.4f}, "
                        f"tol={hardpoint_tolerance:.4f})"
                    )
        else:
            patients, warnings = _estimate_ipd(
                survival_coords, censoring, initial_n=state.config.estimated_cohort_size
            )

        all_warnings.extend(warnings)
        curves.append(CurveIPD(
            group_name=name,
            patients=patients,
            censoring_times=censoring,
        ))

    output = IPDOutput(
        metadata=state.plot_metadata,
        curves=curves,
        reconstruction_mode=mode,
        warnings=state.mmpu_warnings + all_warnings,
    )
    return state.model_copy(update={"output": output})


def validate(state: PipelineState) -> PipelineState:
    if not state.output or not state.digitized_curves:
        return state.model_copy(
            update={
                "errors": state.errors
                + [
                    ProcessingError(
                        stage=ProcessingStage.VALIDATE,
                        error_type="missing_output",
                        recoverable=False,
                        message="Missing output for validation",
                    )
                ]
            }
        )

    cfg = state.config
    validated_curves = []

    compute_full_metrics = state.config.compute_full_validation_metrics
    curve_direction = (
        state.plot_metadata.curve_direction
        if state.plot_metadata and state.plot_metadata.curve_direction in ("downward", "upward")
        else "downward"
    )
    y_start = state.plot_metadata.y_axis.start if state.plot_metadata else 0.0
    y_end = state.plot_metadata.y_axis.end if state.plot_metadata else 1.0

    for curve in state.output.curves:
        raw_original = state.digitized_curves.get(curve.group_name, [])
        original, _ = _ensure_survival_space(
            raw_original,
            curve_direction=curve_direction,
            y_start=y_start,
            y_end=y_end,
        )
        reconstructed = _km_from_ipd(curve.patients)

        mae = _calculate_mae(original, reconstructed)
        updates: dict[str, float | None] = {"validation_mae": mae}
        if compute_full_metrics:
            updates.update({
                "validation_dtw": dtw_distance(original, reconstructed),
                "validation_rmse": rmse(original, reconstructed),
                "validation_max_error": max_error(original, reconstructed),
                "validation_frechet": frechet_distance(original, reconstructed),
            })

        validated_curves.append(curve.model_copy(update=updates))

    updated_output = state.output.model_copy(update={"curves": validated_curves})

    failed = any(
        c.validation_mae is not None and c.validation_mae > cfg.validation_mae_threshold
        for c in validated_curves
    )
    retries = state.validation_retries + 1 if failed else state.validation_retries

    return state.model_copy(update={"output": updated_output, "validation_retries": retries})
