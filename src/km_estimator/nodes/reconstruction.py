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


def _weighted_event_times(total_events: int, times: list[float], weights: list[float]) -> list[float]:
    """Allocate integer events to candidate times proportionally to weights."""
    if total_events <= 0:
        return []
    if not times or len(times) != len(weights):
        return []

    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w)

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
    return int(np.clip(estimated, MIN_ESTIMATED_COHORT_SIZE, MAX_ESTIMATED_COHORT_SIZE))


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
        target_events = expected_events + event_residual
        d_j = int(np.clip(round(target_events), 0, lost_total))
        event_residual = target_events - d_j
        c_j = int(max(0, lost_total - d_j))

        # Place events at observed drop times when possible.
        if d_j > 0:
            drop_times, drop_weights = _extract_drop_points_in_interval(
                normalized_coords,
                t_start,
                t_end,
            )
            event_times = _weighted_event_times(d_j, drop_times, drop_weights)
            if not event_times:
                event_times = np.linspace(t_start, t_end, d_j + 2)[1:-1].tolist()
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
) -> tuple[list[PatientRecord], list[str]]:
    """Estimate IPD when risk table is not available."""
    warnings: list[str] = []
    patients: list[PatientRecord] = []

    if not coords:
        warnings.append("No coordinates for estimation")
        return patients, warnings

    estimated_n = _estimate_initial_n_from_curve(coords, censoring_times, initial_n)
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

    for name, coords in state.digitized_curves.items():
        censoring = state.censoring_marks.get(name, []) if state.censoring_marks else []

        if mode == ReconstructionMode.FULL and risk_table:
            patients, warnings = _guyot_ikm(coords, risk_table, name)
        else:
            patients, warnings = _estimate_ipd(
                coords, censoring, initial_n=state.config.estimated_cohort_size
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

    for curve in state.output.curves:
        original = state.digitized_curves.get(curve.group_name, [])
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
