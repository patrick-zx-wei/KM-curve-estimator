"""IPD reconstruction and validation nodes."""

from bisect import bisect_right

import numpy as np

from km_estimator.utils.shape_metrics import (
    dtw_distance,
    frechet_distance,
    max_error,
    rmse,
)
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
) -> int:
    """Choose integer event count minimizing survival mismatch and rounding drift."""
    if lost_total <= 0 or n_start <= 0:
        return 0

    bounded_target = float(np.clip(target_events, 0.0, float(lost_total)))
    center = int(round(bounded_target))

    # Evaluate all candidates for smaller intervals; otherwise use a bounded local search.
    candidates: list[int]
    if lost_total <= 40:
        candidates = list(range(0, lost_total + 1))
    else:
        low = max(0, center - 8)
        high = min(lost_total, center + 8)
        candidates = sorted(set([0, lost_total, *range(low, high + 1)]))

    best_events = center
    best_score = float("inf")
    for events in candidates:
        predicted_s_end = _estimate_interval_end_survival(n_start, lost_total, events, s_start)
        survival_error = abs(predicted_s_end - s_end)
        rounding_error = abs(float(events) - bounded_target)
        score = survival_error * (8.0 * max(0.7, survival_weight)) + rounding_error * 0.35
        if score < best_score:
            best_score = score
            best_events = int(events)

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
) -> None:
    """Ensure patient record count matches target by adjusting right-censored tail."""
    diff = int(target_total - len(patients))
    if diff == 0:
        return

    if diff > 0:
        for _ in range(diff):
            patients.append(PatientRecord(time=float(final_time), event=False))
        warnings.append(f"Added {diff} terminal censored patients to match cohort size")
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
    event_residual = 0.0
    carried_survival_bias_events = 0.0

    for j in range(len(time_points) - 1):
        t_start = time_points[j]
        t_end = time_points[j + 1]
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
        interval_weight = _interval_survival_weight(t_start, t_end)
        d_j = _choose_interval_event_count(
            n_start=n_start,
            lost_total=lost_total,
            s_start=s_start,
            s_end=s_end,
            target_events=target_events,
            survival_weight=interval_weight,
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
            drop_times, drop_weights = _extract_drop_points_in_interval(
                normalized_coords,
                t_start,
                t_end,
            )
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
            censor_times = np.linspace(t_start, t_end, c_j + 2)[1:-1].tolist()
            for ct in censor_times:
                patients.append(PatientRecord(time=float(ct), event=False))

    # Add final right-censored survivors still at risk at last follow-up.
    final_time = float(time_points[-1])
    final_at_risk = max(0, int(n_at_risk[-1]))
    if final_at_risk > 0:
        for _ in range(final_at_risk):
            patients.append(PatientRecord(time=final_time, event=False))
        warnings.append(f"Added {final_at_risk} terminal right-censored survivors at t={final_time}")

    # Guardrail: reconcile generated patient count to initial at-risk cohort.
    target_total = max(0, int(n_at_risk[0]))
    _reconcile_patient_total(patients, target_total, final_time, warnings)

    # Sort by time
    patients.sort(key=lambda p: p.time)

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
    curve_direction = (
        state.plot_metadata.curve_direction
        if state.plot_metadata.curve_direction in ("downward", "upward")
        else "downward"
    )
    y_start = state.plot_metadata.y_axis.start
    y_end = state.plot_metadata.y_axis.end

    for name, coords in state.digitized_curves.items():
        survival_coords, converted_from_upward = _ensure_survival_space(
            coords,
            curve_direction=curve_direction,
            y_start=y_start,
            y_end=y_end,
        )
        if converted_from_upward:
            all_warnings.append(
                f"{name}: reflected upward curve into survival space during reconstruction"
            )

        censoring = state.censoring_marks.get(name, []) if state.censoring_marks else []

        if mode == ReconstructionMode.FULL and risk_table:
            patients_full, warnings_full = _guyot_ikm(survival_coords, risk_table, name)
            full_curve = _km_from_ipd(patients_full)
            full_fit_mae = _calculate_mae(survival_coords, full_curve)

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
            full_landmark_error = _landmark_proxy_error(
                survival_coords, full_curve, landmark_times
            )
            alt_landmark_error = _landmark_proxy_error(
                survival_coords, alt_curve, landmark_times
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
                and alt_fit_mae + FULL_RECON_ALT_SWITCH_MARGIN < full_fit_mae
                and (
                    full_fit_mae >= FULL_RECON_NON_REGRESSION_TRIGGER
                    or (full_fit_mae - alt_fit_mae) >= 0.02
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
                and alt_fit_mae + 0.012 < full_fit_mae
                and alt_landmark_error + 0.01 < full_landmark_error
            ):
                choose_alt = True

            if choose_alt:
                patients = patients_alt
                warnings = list(warnings_alt)
                warnings.append(
                    f"{name}: switched to estimated reconstruction "
                    f"(fit MAE {full_fit_mae:.4f} -> {alt_fit_mae:.4f})"
                )
                warnings.append(
                    f"{name}: estimated fallback checks "
                    f"(n={len(patients_alt)}, interval_mismatch={alt_interval_mismatch:.3f})"
                )
                warnings.append(
                    f"{name}: landmark proxy improved "
                    f"({full_landmark_error:.4f} -> {alt_landmark_error:.4f})"
                )
            else:
                patients = patients_full
                warnings = warnings_full
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
