"""IPD reconstruction and validation nodes."""

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
    coords: list[tuple[float, float]], t: float
) -> float:
    """Get survival probability at time t from digitized curve (step function)."""
    if not coords:
        return 1.0

    times = [c[0] for c in coords]
    survivals = [c[1] for c in coords]

    # Before first observation
    if t < times[0]:
        return 1.0

    # At or after last observation
    if t >= times[-1]:
        return survivals[-1]

    # Find the latest time point <= t
    for i in range(len(times) - 1, -1, -1):
        if times[i] <= t:
            return survivals[i]

    return 1.0


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

    for j in range(len(time_points) - 1):
        t_start = time_points[j]
        t_end = time_points[j + 1]
        n_start = n_at_risk[j]
        n_end = n_at_risk[j + 1]

        if n_start <= 0:
            continue

        s_start = _get_survival_at_time(coords, t_start)
        s_end = _get_survival_at_time(coords, t_end)

        if s_start <= 0:
            warnings.append(f"Zero survival at t={t_start}")
            continue

        # Events from survival drop
        survival_ratio = s_end / s_start
        d_j = max(0, round(n_start * (1 - survival_ratio)))

        # Clamp events to available at-risk
        if d_j > n_start:
            d_j = n_start
            warnings.append(f"Clamped events at interval [{t_start}, {t_end}]")

        # Censored = at-risk lost minus events
        c_j = max(0, n_start - d_j - n_end)

        # Distribute events uniformly in interval
        if d_j > 0:
            event_times = np.linspace(t_start, t_end, d_j + 2)[1:-1]
            for et in event_times:
                patients.append(PatientRecord(time=float(et), event=True))

        # Distribute censorings uniformly in interval
        if c_j > 0:
            censor_times = np.linspace(t_start, t_end, c_j + 2)[1:-1]
            for ct in censor_times:
                patients.append(PatientRecord(time=float(ct), event=False))

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

    warnings.append(
        f"Using estimated mode without risk table (assumed N={initial_n}). "
        "Patient counts are scaled estimates, not absolute values."
    )

    times = [c[0] for c in coords]
    n_remaining = initial_n
    prev_s = 1.0

    # Sample at regular intervals
    sample_times = np.linspace(times[0], times[-1], min(50, len(times)))

    for t in sample_times:
        s = _get_survival_at_time(coords, t)
        if s < prev_s and n_remaining > 0:
            # Survival dropped - estimate events
            drop = prev_s - s
            events = max(1, round(n_remaining * drop / prev_s))
            events = min(events, n_remaining)

            for _ in range(events):
                patients.append(PatientRecord(time=float(t), event=True))

            n_remaining -= events
        prev_s = s

    # Add known censoring marks
    for ct in censoring_times:
        patients.append(PatientRecord(time=ct, event=False))

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

    # Sample at original timepoints
    errors = []
    for t, s_orig in original:
        s_recon = _get_survival_at_time(reconstructed, t)
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

    for curve in state.output.curves:
        original = state.digitized_curves.get(curve.group_name, [])
        reconstructed = _km_from_ipd(curve.patients)

        # Calculate all validation metrics
        mae = _calculate_mae(original, reconstructed)
        dtw = dtw_distance(original, reconstructed)
        curve_rmse = rmse(original, reconstructed)
        curve_max_error = max_error(original, reconstructed)
        curve_frechet = frechet_distance(original, reconstructed)

        validated_curves.append(curve.model_copy(update={
            "validation_mae": mae,
            "validation_dtw": dtw,
            "validation_rmse": curve_rmse,
            "validation_max_error": curve_max_error,
            "validation_frechet": curve_frechet,
        }))

    updated_output = state.output.model_copy(update={"curves": validated_curves})

    failed = any(
        c.validation_mae is not None and c.validation_mae > cfg.validation_mae_threshold
        for c in validated_curves
    )
    retries = state.validation_retries + 1 if failed else state.validation_retries

    return state.model_copy(update={"output": updated_output, "validation_retries": retries})
