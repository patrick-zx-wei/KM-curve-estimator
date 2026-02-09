"""Legend/color prior handling for digitization_v2."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from km_estimator.models import PlotMetadata

from .axis_map import PlotModel

# Common color names to RGB (normalized 0-1)
COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "red": (0.84, 0.15, 0.16),
    "blue": (0.12, 0.47, 0.71),
    "green": (0.17, 0.63, 0.17),
    "black": (0.0, 0.0, 0.0),
    "orange": (1.0, 0.50, 0.05),
    "purple": (0.58, 0.40, 0.74),
    "brown": (0.55, 0.34, 0.29),
    "pink": (0.89, 0.47, 0.76),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "cyan": (0.09, 0.75, 0.81),
    "magenta": (0.89, 0.10, 0.89),
    "yellow": (0.74, 0.74, 0.13),
}


def parse_curve_color(color_description: str | None) -> tuple[float, float, float] | None:
    """Extract RGB from color description like 'solid blue' or 'dashed red'."""
    if not color_description:
        return None
    desc = color_description.lower()
    for color_name, rgb in COLOR_MAP.items():
        if color_name in desc:
            return rgb
    return None


LAB_PRIOR_MISMATCH_THRESHOLD = 28.0
LAB_PRIOR_REJECT_THRESHOLD = 70.0
MIN_OBSERVED_COLOR_PIXELS = 300
OBSERVED_SATURATION_MIN = 38
OBSERVED_VALUE_MAX = 255


def _rgb01_to_lab(color: tuple[float, float, float]) -> tuple[float, float, float]:
    rgb = np.asarray(
        [[[int(round(color[0] * 255)), int(round(color[1] * 255)), int(round(color[2] * 255))]]],
        dtype=np.uint8,
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return float(lab[0]), float(lab[1]), float(lab[2])


def _lab_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(aa - bb))


@dataclass(frozen=True)
class ArmColorModel:
    name: str
    expected_lab: tuple[float, float, float] | None
    observed_lab: tuple[float, float, float] | None
    source: str  # "legend", "observed", "none"
    valid: bool
    reliability: float
    warning_codes: tuple[str, ...]

    def reference_lab(self) -> tuple[float, float, float] | None:
        if self.reliability <= 0.0:
            return None
        if self.valid and self.observed_lab is not None:
            return self.observed_lab
        return self.expected_lab


def _collect_observed_centers(
    roi_bgr: NDArray[np.uint8],
    exclude_mask: NDArray[np.uint8],
    n_centers: int,
) -> list[tuple[float, float, float]]:
    """Estimate dominant curve colors from high-saturation pixels."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    valid = (
        (sat >= OBSERVED_SATURATION_MIN)
        & (val <= OBSERVED_VALUE_MAX)
        & (exclude_mask == 0)
    )
    ys, xs = np.where(valid)
    if ys.size < max(MIN_OBSERVED_COLOR_PIXELS, n_centers * 60):
        return []

    samples = lab[ys, xs]
    # Hue-stratified downsample: cap per-hue-bin to prevent any single color
    # from dominating k-means (e.g., bright orange at V=255 outnumbering green).
    hues = hsv[:, :, 0][ys, xs]
    budget = 25000
    n_bins = 12
    bin_edges = np.linspace(0, 180, n_bins + 1)
    bin_idx = np.clip(np.digitize(hues, bin_edges) - 1, 0, n_bins - 1)
    populated = np.unique(bin_idx)
    per_bin = max(1, budget // max(1, len(populated)))
    keep = np.zeros(samples.shape[0], dtype=bool)
    for b in populated:
        mask_b = bin_idx == b
        idxs = np.where(mask_b)[0]
        if idxs.size <= per_bin:
            keep[idxs] = True
        else:
            step = int(np.ceil(idxs.size / per_bin))
            keep[idxs[::step]] = True
    samples = np.asarray(samples[keep], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
    compactness, labels, centers = cv2.kmeans(
        samples,
        K=max(1, min(n_centers + 2, samples.shape[0])),
        bestLabels=None,
        criteria=criteria,
        attempts=1,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    if not np.isfinite(compactness):
        return []

    center_list = [tuple(float(v) for v in center) for center in centers]
    # Deterministic order by lightness then a/b channels.
    center_list.sort(key=lambda c: (c[0], c[1], c[2]))
    return center_list


def _greedy_assign_centers(
    curve_names: list[str],
    expected: dict[str, tuple[float, float, float] | None],
    centers: list[tuple[float, float, float]],
) -> dict[str, tuple[float, float, float] | None]:
    assigned: dict[str, tuple[float, float, float] | None] = {name: None for name in curve_names}
    if not centers:
        return assigned

    # Build all pair costs for deterministic greedy assignment.
    pairs: list[tuple[float, str, int]] = []
    for name in curve_names:
        e = expected.get(name)
        if e is None:
            continue
        for idx, c in enumerate(centers):
            pairs.append((_lab_distance(e, c), name, idx))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    used_names: set[str] = set()
    used_centers: set[int] = set()
    for _, name, idx in pairs:
        if name in used_names or idx in used_centers:
            continue
        assigned[name] = centers[idx]
        used_names.add(name)
        used_centers.add(idx)

    # Fill unassigned names with remaining centers in deterministic order.
    remaining_names = [n for n in curve_names if assigned[n] is None]
    remaining_centers = [centers[i] for i in range(len(centers)) if i not in used_centers]
    for name, center in zip(remaining_names, remaining_centers):
        assigned[name] = center
    return assigned


def build_color_models(
    image: NDArray[np.uint8],
    meta: PlotMetadata,
    plot_model: PlotModel,
) -> tuple[dict[str, ArmColorModel], list[str]]:
    """Build validated per-arm color models from legend priors + observed colors."""
    warnings: list[str] = []
    x0, y0, x1, y1 = plot_model.plot_region
    roi = image[y0:y1, x0:x1]

    curve_names = [curve.name for curve in meta.curves]
    expected: dict[str, tuple[float, float, float] | None] = {}
    for curve in meta.curves:
        rgb = parse_curve_color(curve.color_description)
        expected[curve.name] = _rgb01_to_lab(rgb) if rgb is not None else None

    exclude_mask = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
    centers = _collect_observed_centers(roi, exclude_mask, n_centers=max(1, len(curve_names)))
    if not centers:
        warnings.append("W_COLOR_UNINFORMATIVE")
    observed_by_name = _greedy_assign_centers(curve_names, expected, centers)

    models: dict[str, ArmColorModel] = {}
    for name in curve_names:
        local_codes: list[str] = []
        exp_lab = expected.get(name)
        obs_lab = observed_by_name.get(name)

        if exp_lab is None and obs_lab is None:
            models[name] = ArmColorModel(
                name=name,
                expected_lab=None,
                observed_lab=None,
                source="none",
                valid=False,
                reliability=0.0,
                warning_codes=("W_COLOR_PRIOR_MISSING",),
            )
            warnings.append(f"W_COLOR_PRIOR_MISSING:{name}")
            continue

        if exp_lab is not None and obs_lab is not None:
            dist = _lab_distance(exp_lab, obs_lab)
            if dist > LAB_PRIOR_REJECT_THRESHOLD:
                local_codes.append("W_COLOR_PRIOR_REJECTED")
                warnings.append(f"W_COLOR_PRIOR_REJECTED:{name}:{dist:.1f}")
                models[name] = ArmColorModel(
                    name=name,
                    expected_lab=None,
                    observed_lab=obs_lab,
                    source="observed",
                    valid=True,
                    reliability=0.35,
                    warning_codes=tuple(local_codes),
                )
            elif dist > LAB_PRIOR_MISMATCH_THRESHOLD:
                local_codes.append("W_COLOR_PRIOR_MISMATCH")
                warnings.append(f"W_COLOR_PRIOR_MISMATCH:{name}:{dist:.1f}")
                models[name] = ArmColorModel(
                    name=name,
                    expected_lab=None,
                    observed_lab=obs_lab,
                    source="observed",
                    valid=True,
                    reliability=0.45,
                    warning_codes=tuple(local_codes),
                )
            else:
                models[name] = ArmColorModel(
                    name=name,
                    expected_lab=exp_lab,
                    observed_lab=obs_lab,
                    source="legend",
                    valid=True,
                    reliability=1.0,
                    warning_codes=tuple(local_codes),
                )
            continue

        # Single-source fallback.
        if obs_lab is not None:
            local_codes.append("W_COLOR_FROM_OBSERVED_ONLY")
            warnings.append(f"W_COLOR_FROM_OBSERVED_ONLY:{name}")
            models[name] = ArmColorModel(
                name=name,
                expected_lab=exp_lab,
                observed_lab=obs_lab,
                source="observed",
                valid=True,
                reliability=0.55,
                warning_codes=tuple(local_codes),
            )
            continue

        local_codes.append("W_COLOR_OBSERVED_MISSING")
        warnings.append(f"W_COLOR_OBSERVED_MISSING:{name}")
        models[name] = ArmColorModel(
            name=name,
            expected_lab=exp_lab,
            observed_lab=None,
            source="legend",
            valid=True,
            reliability=0.40,
            warning_codes=tuple(local_codes),
        )

    return models, warnings
