"""Minimal curve cleanup and conversion for digitization_v2."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .axis_map import CurveDirection, PlotModel

SPIKE_WINDOW = 3
SOFT_MONO_TOLERANCE = 0.01
SOFT_MONO_MAX_RUN = 5


def _running_median(values: list[float], window: int) -> list[float]:
    if len(values) <= 2 or window <= 1:
        return values
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(float(np.median(np.asarray(values[lo:hi], dtype=np.float32))))
    return out


def _soft_monotone(
    values: list[float],
    direction: CurveDirection,
) -> tuple[list[float], int, bool]:
    """
    Non-destructive monotone correction.

    - Only edits short violation runs.
    - If violations are widespread, do not edit; request retrace instead.
    """
    if not values:
        return values, 0, False
    out = list(values)
    violating: list[int] = []
    if direction in ("downward", "unknown"):
        for i in range(1, len(out)):
            if out[i] > out[i - 1] + SOFT_MONO_TOLERANCE:
                violating.append(i)
    else:
        for i in range(1, len(out)):
            if out[i] < out[i - 1] - SOFT_MONO_TOLERANCE:
                violating.append(i)

    if not violating:
        return out, 0, False

    max_edits = max(1, int(round(len(values) * 0.01)))
    if len(violating) > max_edits:
        return values, 0, True

    runs: list[list[int]] = []
    cur = [violating[0]]
    for idx in violating[1:]:
        if idx == cur[-1] + 1:
            cur.append(idx)
        else:
            runs.append(cur)
            cur = [idx]
    runs.append(cur)

    n_fix = 0
    for run in runs:
        if len(run) > SOFT_MONO_MAX_RUN:
            return values, n_fix, True
        for i in run:
            if direction in ("downward", "unknown"):
                allowed = out[i - 1] + SOFT_MONO_TOLERANCE
                if out[i] > allowed:
                    out[i] = 0.85 * allowed + 0.15 * out[i]
                    n_fix += 1
            else:
                allowed = out[i - 1] - SOFT_MONO_TOLERANCE
                if out[i] < allowed:
                    out[i] = 0.85 * allowed + 0.15 * out[i]
                    n_fix += 1
    return out, n_fix, False


def _to_survival_value(y_real: float, y_start: float, y_end: float, direction: CurveDirection) -> float:
    if direction != "upward":
        return float(y_real)
    denom = max(1e-9, float(y_end - y_start))
    incidence = (float(y_real) - float(y_start)) / denom
    return float(np.clip(1.0 - incidence, 0.0, 1.0))


def convert_pixel_curves_to_survival(
    pixel_curves: dict[str, list[tuple[int, int]]],
    plot_model: PlotModel,
    direction: CurveDirection,
) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    """Convert traced pixel curves to real-space survival coordinates."""
    warnings: list[str] = []
    out: dict[str, list[tuple[float, float]]] = {}
    y_start = float(plot_model.mapping.y_axis.start)
    y_end = float(plot_model.mapping.y_axis.end)

    for name, pts in pixel_curves.items():
        if not pts:
            out[name] = []
            warnings.append(f"W_EMPTY_TRACE:{name}")
            continue

        # Deduplicate by x with median y.
        by_x: dict[int, list[int]] = defaultdict(list)
        for px, py in pts:
            by_x[int(px)].append(int(py))
        ordered_x = sorted(by_x)
        xs_real: list[float] = []
        ys_surv: list[float] = []
        for px in ordered_x:
            py = int(round(float(np.median(np.asarray(by_x[px], dtype=np.float32)))))
            xr, yr = plot_model.px_to_real(px, py)
            xs_real.append(float(xr))
            ys_surv.append(_to_survival_value(float(yr), y_start, y_end, direction))

        ys_smooth = _running_median(ys_surv, SPIKE_WINDOW)
        ys_soft, n_fix, retrace_needed = _soft_monotone(ys_smooth, direction="downward")
        if n_fix > 0:
            warnings.append(f"W_SOFT_MONOTONE_ADJUST:{name}:{n_fix}")
        if retrace_needed:
            warnings.append(f"E_NEEDS_RETRACE_MONOTONE:{name}")

        coords = [(float(t), float(s)) for t, s in zip(xs_real, ys_soft)]
        out[name] = coords

    if direction == "upward":
        warnings.append("W_UPWARD_REFLECTED_TO_SURVIVAL")
    return out, warnings
