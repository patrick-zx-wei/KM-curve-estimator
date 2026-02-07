"""Shared evidence cube and per-arm score maps for digitization_v2."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .axis_map import PlotModel
from .legend_color import ArmColorModel

RIDGE_WEIGHT = 0.30
EDGE_WEIGHT = 0.20
COLOR_WEIGHT = 0.45
AXIS_PENALTY_WEIGHT = 0.85
AXIS_PENALTY_WEIGHT_UPWARD = 0.45
AXIS_PENALTY_WEIGHT_UNKNOWN = 0.65
TEXT_PENALTY_WEIGHT = 0.35
COLOR_GOOD_DISTANCE = 42.0
CANDIDATE_AXIS_THRESH = 0.25
CANDIDATE_AXIS_THRESH_UPWARD = 0.55
CANDIDATE_AXIS_THRESH_UNKNOWN = 0.40
HSL_KMEDOIDS_MAX_POINTS = 2600
HSL_KMEDOIDS_MAX_ITERS = 6
HSL_KMEDOIDS_KNN_K = 12
HSL_BG_LIGHTNESS_MAX = 0.96
HSL_BG_SATURATION_MIN = 0.05
HSL_CLUSTER_WEIGHT = np.asarray([1.0, 1.6, 1.6], dtype=np.float32)
CONSENSUS_BLEND_WEIGHT = 0.30


def _normalize01(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _ridge_response(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    ridge = np.abs(lap).astype(np.float32)
    return _normalize01(ridge)


def _edge_response(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    edges = cv2.Canny(gray, 35, 110).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    return _normalize01(edges.astype(np.float32))


def _text_penalty(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Approximate text-like regions as a soft penalty map."""
    inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        6,
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = np.zeros_like(gray, dtype=np.uint8)
    h, w = gray.shape
    area_floor = max(8, int((h * w) * 0.00002))
    area_ceil = max(area_floor + 1, int((h * w) * 0.004))
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < area_floor or area > area_ceil:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        # Text glyph-ish blobs: compact-ish and not large bars.
        if aspect > 7.5:
            continue
        out[labels == idx] = 255
    out = cv2.dilate(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def _color_likelihood(
    roi_lab: NDArray[np.float32],
    reference_lab: tuple[float, float, float] | None,
    reliability: float,
) -> NDArray[np.float32]:
    if reference_lab is None or reliability <= 0.0:
        return np.zeros(roi_lab.shape[:2], dtype=np.float32)

    ref = np.asarray(reference_lab, dtype=np.float32)
    dist = np.linalg.norm(roi_lab - ref[None, None, :], axis=2).astype(np.float32)
    # Saturating positive-only color contribution.
    likelihood = np.clip((COLOR_GOOD_DISTANCE - dist) / COLOR_GOOD_DISTANCE, 0.0, 1.0)
    return likelihood.astype(np.float32)


def _pairwise_sqdist(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    out = aa + bb - (2.0 * np.dot(a, b.T))
    return np.maximum(out, 0.0).astype(np.float32)


def _init_medoids_farthest(
    x: NDArray[np.float32],
    k: int,
) -> NDArray[np.int32]:
    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, min(k, n))
    medoids = [0]
    if k == 1:
        return np.asarray(medoids, dtype=np.int32)
    d2 = np.sum((x - x[0:1]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(d2))
        medoids.append(idx)
        d2 = np.minimum(d2, np.sum((x - x[idx:idx + 1]) ** 2, axis=1))
    return np.asarray(medoids, dtype=np.int32)


def _pam_kmedoids(
    x: NDArray[np.float32],
    k: int,
    max_iters: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    medoids = _init_medoids_farthest(x, k)
    dist = _pairwise_sqdist(x, x)
    labels = np.argmin(dist[:, medoids], axis=1).astype(np.int32)

    for _ in range(max_iters):
        changed = False
        for j in range(len(medoids)):
            members = np.where(labels == j)[0]
            if members.size == 0:
                continue
            sub = dist[np.ix_(members, members)]
            costs = np.sum(sub, axis=1)
            best_local = int(np.argmin(costs))
            new_medoid = int(members[best_local])
            if new_medoid != int(medoids[j]):
                medoids[j] = new_medoid
                changed = True
        new_labels = np.argmin(dist[:, medoids], axis=1).astype(np.int32)
        if not changed and np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return medoids.astype(np.int32), labels


def _build_hsl_partition(
    roi_bgr: NDArray[np.uint8],
    candidate_mask: NDArray[np.bool_],
    axis_penalty: NDArray[np.float32],
    text_penalty: NDArray[np.float32],
    n_clusters: int,
) -> tuple[
    list[NDArray[np.float32]],
    NDArray[np.float32],
    list[tuple[float, float, float]],
    list[str],
]:
    """
    Partition foreground color with HSL + K-medoids and compute kNN consensus.

    Returns:
      cluster_likelihood_maps, consensus_map, cluster_lab_centers, warnings
    """
    warnings: list[str] = []
    h, w = candidate_mask.shape
    if n_clusters <= 0:
        return [], np.zeros((h, w), dtype=np.float32), [], warnings

    roi_hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h_chan = (roi_hls[:, :, 0] / 180.0).astype(np.float32)
    l_chan = (roi_hls[:, :, 1] / 255.0).astype(np.float32)
    s_chan = (roi_hls[:, :, 2] / 255.0).astype(np.float32)

    fg_mask = (
        candidate_mask
        & (axis_penalty < 0.70)
        & (text_penalty < 0.70)
        & ((s_chan >= HSL_BG_SATURATION_MIN) | (l_chan <= HSL_BG_LIGHTNESS_MAX))
    )
    ys, xs = np.where(fg_mask)
    n_fg = int(ys.size)
    if n_fg < max(120, n_clusters * 60):
        warnings.append(f"W_HSL_KMEDOIDS_SPARSE:{n_fg}")
        return [], np.zeros((h, w), dtype=np.float32), [], warnings

    feats = np.stack([h_chan[ys, xs], s_chan[ys, xs], l_chan[ys, xs]], axis=1).astype(np.float32)
    if n_fg > HSL_KMEDOIDS_MAX_POINTS:
        step = int(np.ceil(float(n_fg) / float(HSL_KMEDOIDS_MAX_POINTS)))
        sel = np.arange(0, n_fg, step, dtype=np.int32)
    else:
        sel = np.arange(0, n_fg, dtype=np.int32)

    ys_s = ys[sel]
    xs_s = xs[sel]
    x_s = feats[sel]
    mu = np.mean(x_s, axis=0, keepdims=True)
    sigma = np.std(x_s, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    x_norm = ((x_s - mu) / sigma) * HSL_CLUSTER_WEIGHT[None, :]

    k = max(1, min(n_clusters, x_norm.shape[0]))
    medoid_idx, labels_s = _pam_kmedoids(x_norm, k=k, max_iters=HSL_KMEDOIDS_MAX_ITERS)
    if medoid_idx.size == 0:
        warnings.append("W_HSL_KMEDOIDS_FAILED")
        return [], np.zeros((h, w), dtype=np.float32), [], warnings

    medoids = x_norm[medoid_idx]
    medoid_lab = [tuple(float(v) for v in roi_lab[ys_s[i], xs_s[i]]) for i in medoid_idx.tolist()]

    x_all = ((feats - mu) / sigma) * HSL_CLUSTER_WEIGHT[None, :]
    d_all = np.sqrt(_pairwise_sqdist(x_all.astype(np.float32), medoids.astype(np.float32)))
    sigma_d = np.percentile(d_all, 40, axis=0)
    sigma_d = np.clip(sigma_d, 0.08, 0.65).astype(np.float32)

    cluster_like_maps: list[NDArray[np.float32]] = []
    for j in range(k):
        like = np.exp(-((d_all[:, j] / sigma_d[j]) ** 2)).astype(np.float32)
        m = np.zeros((h, w), dtype=np.float32)
        m[ys, xs] = like
        cluster_like_maps.append(_normalize01(m))

    # kNN consensus on sampled foreground pixels.
    if x_norm.shape[0] >= max(30, HSL_KMEDOIDS_KNN_K + 1):
        d2 = _pairwise_sqdist(x_norm.astype(np.float32), x_norm.astype(np.float32))
        np.fill_diagonal(d2, np.inf)
        k_nn = min(HSL_KMEDOIDS_KNN_K, x_norm.shape[0] - 1)
        nn_idx = np.argpartition(d2, kth=k_nn, axis=1)[:, :k_nn]
        nn_d2 = np.take_along_axis(d2, nn_idx, axis=1)
        nn_w = 1.0 / (nn_d2 + 1e-10)
        same = np.where(labels_s[nn_idx] == labels_s[:, None], 1.0, -1.0).astype(np.float32)
        raw = np.sum(same * nn_w, axis=1) / np.maximum(1e-9, np.sum(nn_w, axis=1))
        raw01 = np.clip(0.5 * (raw + 1.0), 0.0, 1.0).astype(np.float32)
        cons = np.zeros((h, w), dtype=np.float32)
        cons[ys_s, xs_s] = raw01
        cons = cv2.GaussianBlur(cons, (7, 7), 0)
        fill = float(np.median(raw01))
        cons = np.where(cons > 0, cons, fill).astype(np.float32)
        consensus_map = _normalize01(cons)
    else:
        consensus_map = np.full((h, w), 0.5, dtype=np.float32)
        warnings.append("W_HSL_KNN_CONSENSUS_SPARSE")

    warnings.append(f"I_HSL_KMEDOIDS_USED:{k}:{x_norm.shape[0]}")
    return cluster_like_maps, consensus_map, medoid_lab, warnings


def _assign_clusters_to_arms(
    arm_names: list[str],
    color_models: dict[str, ArmColorModel],
    cluster_lab: list[tuple[float, float, float]],
) -> dict[str, int | None]:
    assigned: dict[str, int | None] = {name: None for name in arm_names}
    if not cluster_lab:
        return assigned

    pairs: list[tuple[float, str, int]] = []
    for name in arm_names:
        ref = color_models[name].reference_lab() if name in color_models else None
        if ref is None:
            continue
        ref_arr = np.asarray(ref, dtype=np.float32)
        for idx, center in enumerate(cluster_lab):
            dist = float(np.linalg.norm(ref_arr - np.asarray(center, dtype=np.float32)))
            pairs.append((dist, name, idx))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))

    used_names: set[str] = set()
    used_clusters: set[int] = set()
    for _, name, idx in pairs:
        if name in used_names or idx in used_clusters:
            continue
        assigned[name] = idx
        used_names.add(name)
        used_clusters.add(idx)

    leftovers = [idx for idx in range(len(cluster_lab)) if idx not in used_clusters]
    for name in arm_names:
        if assigned[name] is None and leftovers:
            assigned[name] = leftovers.pop(0)
    return assigned


@dataclass(frozen=True)
class EvidenceCube:
    ridge_map: NDArray[np.float32]
    edge_map: NDArray[np.float32]
    text_penalty_map: NDArray[np.float32]
    axis_penalty_map: NDArray[np.float32]
    structure_map: NDArray[np.float32]
    overlap_consensus_map: NDArray[np.float32]
    candidate_mask: NDArray[np.bool_]
    arm_score_maps: dict[str, NDArray[np.float32]]
    ambiguity_map: NDArray[np.float32]
    warning_codes: tuple[str, ...]


def build_evidence_cube(
    image: NDArray[np.uint8],
    plot_model: PlotModel,
    color_models: dict[str, ArmColorModel],
) -> EvidenceCube:
    """Compute shared evidence once, then derive per-arm score maps."""
    warnings: list[str] = []
    x0, y0, x1, y1 = plot_model.plot_region
    roi = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    ridge = _ridge_response(gray)
    edge = _edge_response(gray)
    text_pen = _text_penalty(gray)

    axis_pen = cv2.bitwise_or(plot_model.axis_mask, plot_model.tick_mask)
    axis_pen_f = (axis_pen.astype(np.float32) / 255.0).astype(np.float32)
    axis_pen_f = cv2.GaussianBlur(axis_pen_f, (5, 5), 0)
    axis_pen_f = _normalize01(axis_pen_f)

    direction = plot_model.curve_direction
    axis_weight = AXIS_PENALTY_WEIGHT
    candidate_axis_thresh = CANDIDATE_AXIS_THRESH
    axis_pen_for_structure = axis_pen_f
    if direction == "upward":
        h = axis_pen_f.shape[0]
        row_rel = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        # Upward (cumulative incidence) curves are expected near the x-axis early;
        # reduce axis suppression in the lower band while still discouraging true-axis capture.
        soften = 0.35 + 0.65 * (1.0 - row_rel)
        axis_pen_for_structure = (axis_pen_f * soften).astype(np.float32)
        axis_weight = AXIS_PENALTY_WEIGHT_UPWARD
        candidate_axis_thresh = CANDIDATE_AXIS_THRESH_UPWARD
        warnings.append("I_AXIS_PENALTY_SOFTENED_UPWARD")
    elif direction == "unknown":
        axis_weight = AXIS_PENALTY_WEIGHT_UNKNOWN
        candidate_axis_thresh = CANDIDATE_AXIS_THRESH_UNKNOWN

    structure_base = (
        RIDGE_WEIGHT * ridge
        + EDGE_WEIGHT * edge
        - axis_weight * axis_pen_for_structure
        - TEXT_PENALTY_WEIGHT * text_pen
    ).astype(np.float32)
    structure_map = _normalize01(structure_base)

    candidate_mask = (
        (ridge > 0.20)
        & (axis_pen_f < candidate_axis_thresh)
        & (text_pen < 0.55)
    )
    cand_density = float(np.mean(candidate_mask))
    if cand_density < 0.003:
        warnings.append(f"W_RIDGE_CANDIDATES_SPARSE:{cand_density:.4f}")
        candidate_mask = np.ones_like(candidate_mask, dtype=np.bool_)

    arm_names = sorted(color_models)
    cluster_like_maps, overlap_consensus_map, cluster_lab, cluster_warnings = _build_hsl_partition(
        roi_bgr=roi,
        candidate_mask=candidate_mask,
        axis_penalty=axis_pen_f,
        text_penalty=text_pen,
        n_clusters=max(1, len(arm_names)),
    )
    warnings.extend(cluster_warnings)
    cluster_assignment = _assign_clusters_to_arms(arm_names, color_models, cluster_lab)

    arm_maps: dict[str, NDArray[np.float32]] = {}
    for arm_name in arm_names:
        model = color_models[arm_name]
        color_like = _color_likelihood(
            roi_lab=roi_lab,
            reference_lab=model.reference_lab(),
            reliability=model.reliability,
        )
        cluster_like = np.zeros_like(color_like, dtype=np.float32)
        cluster_idx = cluster_assignment.get(arm_name)
        if isinstance(cluster_idx, int) and 0 <= cluster_idx < len(cluster_like_maps):
            cluster_like = cluster_like_maps[cluster_idx]
        color_mix = np.maximum(color_like, 0.85 * cluster_like).astype(np.float32)
        color_term = (
            COLOR_WEIGHT
            * np.clip(0.25 + 0.75 * model.reliability, 0.15, 1.0)
            * color_mix
        ).astype(np.float32)
        base = (structure_base + color_term).astype(np.float32)

        # Ridge-first candidates: attenuate non-candidates when dense enough.
        if cand_density >= 0.01:
            base = np.where(candidate_mask, base, base - 0.20)

        arm_maps[arm_name] = _normalize01(base.astype(np.float32))
        if model.reliability <= 0.05:
            warnings.append(f"W_ARM_COLOR_UNINFORMATIVE:{arm_name}")

    if not arm_maps:
        warnings.append("W_NO_ARM_SCORE_MAPS")
        ambiguity = np.zeros_like(ridge, dtype=np.float32)
    elif len(arm_maps) == 1:
        only = next(iter(arm_maps.values()))
        ambiguity = only.copy()
    else:
        stack = np.stack([arm_maps[name] for name in sorted(arm_maps)], axis=0)
        sorted_stack = np.sort(stack, axis=0)
        best = sorted_stack[-1]
        second = sorted_stack[-2]
        ambiguity = (best - second).astype(np.float32)
        ambiguity = _normalize01(ambiguity)
        if overlap_consensus_map.size and np.any(overlap_consensus_map > 0):
            ambiguity = (
                (1.0 - CONSENSUS_BLEND_WEIGHT) * ambiguity
                + CONSENSUS_BLEND_WEIGHT * overlap_consensus_map
            ).astype(np.float32)
            ambiguity = _normalize01(ambiguity)

    return EvidenceCube(
        ridge_map=ridge,
        edge_map=edge,
        text_penalty_map=text_pen,
        axis_penalty_map=axis_pen_f,
        structure_map=structure_map,
        overlap_consensus_map=overlap_consensus_map,
        candidate_mask=candidate_mask.astype(np.bool_),
        arm_score_maps=arm_maps,
        ambiguity_map=ambiguity,
        warning_codes=tuple(warnings),
    )
