"""Path tracing for digitization_v2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .axis_map import CurveDirection
from .probability_map import EvidenceCube

DirectionMode = Literal["downward", "upward", "unknown"]

MAX_CANDIDATES_PER_COLUMN = 10
MIN_COLUMN_SCORE = 0.16
JOINT_MAX_STATES = 72
COLLISION_DISTANCE_RATIO = 0.020
ORDER_LOCK_PENALTY = 0.26
SWAP_ALLOWED_PENALTY = 0.06
SMOOTHNESS_WEIGHT = 1.15
DIRECTION_WEIGHT = 1.0
AXIS_NEAR_WEIGHT = 0.8
LOW_CONF_COLUMN_THRESHOLD = 0.22
LOW_AMBIGUITY_THRESHOLD = 0.12
JUMP_THRESHOLD_RATIO = 0.035
MONOTONE_TOLERANCE_RATIO = 0.006
START_PRIOR_WEIGHT = 0.55
START_PRIOR_WINDOW_RATIO = 0.12

# Crossing detection caps and locality constraints
MAX_CROSSING_WINDOWS = 10
MAX_CROSSING_COL_RATIO = 0.20
CROSSING_MAX_WINDOW_RATIO = 0.08
CROSSING_WINDOW_RADIUS_RATIO = 0.02
CROSSING_SEARCH_RADIUS_RATIO = 0.03
CROSSING_MERGE_GAP = 3
APPROACH_DISTANCE_RATIO = 0.015
SEPARATION_DISTANCE_RATIO = 0.02
SWAP_GAIN_MARGIN = 0.08


@dataclass(frozen=True)
class TraceConfig:
    """Runtime weights for one tracing pass."""

    axis_multiplier: float = 1.0
    smoothness_multiplier: float = 1.0
    direction_multiplier: float = 1.0
    order_lock_multiplier: float = 1.0
    swap_multiplier: float = 1.0


@dataclass(frozen=True)
class TraceResult:
    pixel_curves: dict[str, list[tuple[int, int]]]
    confidence_by_arm: dict[str, float]
    arm_diagnostics: dict[str, dict[str, float]]
    plot_confidence: float
    warning_codes: tuple[str, ...]
    crossing_windows: tuple[tuple[int, int], ...]


def _direction_penalty(
    prev_y: int,
    cur_y: int,
    direction: DirectionMode,
    height: int,
    cfg: TraceConfig,
) -> float:
    tol = max(1, int(round(height * MONOTONE_TOLERANCE_RATIO)))
    scale = DIRECTION_WEIGHT * max(0.1, cfg.direction_multiplier)
    if direction == "downward":
        if cur_y + tol < prev_y:
            return scale * float(prev_y - cur_y) / float(max(1, height))
        return 0.0
    if direction == "upward":
        if cur_y > prev_y + tol:
            return scale * float(cur_y - prev_y) / float(max(1, height))
        return 0.0
    return 0.0


def _axis_penalty_value(
    axis_penalty_map: NDArray[np.float32],
    y: int,
    x: int,
    direction: DirectionMode,
    height: int,
) -> float:
    pen = float(axis_penalty_map[y, x])
    if direction == "upward":
        # Upward/cumulative-incidence curves legitimately hug the x-axis early.
        rel = float(y) / float(max(1, height - 1))
        factor = 0.35 + 0.65 * (1.0 - rel)
        return pen * factor
    return pen


def _start_anchor_penalty(
    y: int,
    x: int,
    width: int,
    height: int,
    direction: DirectionMode,
    cfg: TraceConfig,
) -> float:
    if direction == "unknown":
        return 0.0
    window = max(8, int(round(width * START_PRIOR_WINDOW_RATIO)))
    if x >= window:
        return 0.0
    target = (height - 1) if direction == "upward" else 0
    dist = abs(int(y) - int(target)) / float(max(1, height))
    decay = 1.0 - (float(x) / float(max(1, window)))
    return START_PRIOR_WEIGHT * cfg.direction_multiplier * decay * dist


def _column_candidates(
    column_scores: NDArray[np.float32],
    top_k: int,
    min_score: float,
    hard_mask: NDArray[np.bool_] | None,
) -> list[tuple[int, float]]:
    h = int(column_scores.shape[0])
    if h <= 0:
        return []
    source_idx = np.arange(h, dtype=np.int32)
    if hard_mask is not None and hard_mask.shape[0] == h:
        masked = source_idx[hard_mask]
        if masked.size >= 2:
            source_idx = masked
    source_scores = column_scores[source_idx]
    k = min(max(1, top_k), int(source_scores.shape[0]))
    idx_local = np.argpartition(source_scores, -k)[-k:]
    items = [
        (int(source_idx[i]), float(source_scores[i]))
        for i in idx_local
        if float(source_scores[i]) >= min_score
    ]
    if not items:
        best_local = int(np.argmax(source_scores))
        items = [(int(source_idx[best_local]), float(source_scores[best_local]))]
    items.sort(key=lambda p: (p[1], -p[0]), reverse=True)
    return items


def _extract_arm_diagnostics(
    ys: list[int],
    score_map: NDArray[np.float32],
    axis_penalty_map: NDArray[np.float32],
    ambiguity_map: NDArray[np.float32],
    direction: DirectionMode,
) -> dict[str, float]:
    if not ys:
        return {
            "mean_score": 0.0,
            "low_margin_frac": 1.0,
            "axis_capture_frac": 1.0,
            "jump_rate": 1.0,
            "monotone_violation_mass": 1.0,
            "low_conf_frac": 1.0,
            "overlap_conflict_frac": 0.0,
        }

    h, w = score_map.shape
    selected = np.asarray([score_map[ys[x], x] for x in range(w)], dtype=np.float32)
    low_conf_frac = float(np.mean(selected < LOW_CONF_COLUMN_THRESHOLD))
    axis_hits: list[float] = []
    for x in range(w):
        y = ys[x]
        pen = float(axis_penalty_map[y, x])
        if direction == "upward":
            # Near-bottom early columns are expected for cumulative-incidence starts;
            # don't count these as axis-capture failures.
            rel = float(y) / float(max(1, h - 1))
            if x < int(round(0.15 * w)) and rel > 0.88:
                axis_hits.append(0.0)
                continue
        axis_hits.append(1.0 if pen > 0.35 else 0.0)
    axis_capture_frac = float(np.mean(np.asarray(axis_hits, dtype=np.float32)))
    low_margin_frac = float(
        np.mean(np.asarray([ambiguity_map[ys[x], x] for x in range(w)], dtype=np.float32) < LOW_AMBIGUITY_THRESHOLD)
    )

    jump_thr = max(2, int(round(h * JUMP_THRESHOLD_RATIO)))
    jump_rate = 0.0
    monotone_mass = 0.0
    if len(ys) >= 2:
        jumps = 0
        tol = max(1, int(round(h * MONOTONE_TOLERANCE_RATIO)))
        for i in range(1, len(ys)):
            dy = int(ys[i] - ys[i - 1])
            if abs(dy) > jump_thr:
                jumps += 1
            if direction == "downward" and dy < -tol:
                monotone_mass += float(-dy) / float(max(1, h))
            elif direction == "upward" and dy > tol:
                monotone_mass += float(dy) / float(max(1, h))
        jump_rate = float(jumps) / float(len(ys) - 1)
        monotone_mass = float(monotone_mass) / float(len(ys) - 1)

    return {
        "mean_score": float(np.mean(selected)),
        "low_margin_frac": low_margin_frac,
        "axis_capture_frac": axis_capture_frac,
        "jump_rate": jump_rate,
        "monotone_violation_mass": monotone_mass,
        "low_conf_frac": low_conf_frac,
        "overlap_conflict_frac": 0.0,
    }


def _confidence_from_diagnostics(diag: dict[str, float], direction_confidence: float) -> float:
    conf = 1.0
    conf -= 0.70 * float(diag.get("axis_capture_frac", 0.0))
    conf -= 0.45 * float(diag.get("low_margin_frac", 0.0))
    conf -= 0.30 * float(diag.get("jump_rate", 0.0))
    conf -= 0.90 * float(diag.get("monotone_violation_mass", 0.0))
    conf -= 0.35 * float(diag.get("overlap_conflict_frac", 0.0))
    conf -= 0.20 * float(diag.get("low_conf_frac", 0.0))
    conf += 0.25 * float(diag.get("mean_score", 0.0))
    conf *= 0.85 + 0.15 * float(direction_confidence)
    return float(np.clip(conf, 0.0, 1.0))


def _mask_to_windows(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    if mask.size == 0:
        return windows
    start = -1
    for i, flag in enumerate(mask.tolist()):
        if flag and start < 0:
            start = i
        elif not flag and start >= 0:
            windows.append((start, i - 1))
            start = -1
    if start >= 0:
        windows.append((start, int(mask.size - 1)))
    return windows


def _merge_windows(windows: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = windows[0]
    for s, e in windows[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _detect_crossing_windows(
    map_a: NDArray[np.float32],
    map_b: NDArray[np.float32],
    ambiguity: NDArray[np.float32],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Detect true local crossing windows and close-parallel ambiguous windows.

    Crossing requires geometric inversion evidence:
    - delta sign change
    - approach near zero around sign change
    - separation before/after change
    """
    h, w = map_a.shape
    y_a = np.argmax(map_a, axis=0).astype(np.int32)
    y_b = np.argmax(map_b, axis=0).astype(np.int32)
    delta = (y_a - y_b).astype(np.int32)

    approach_px = max(2, int(round(h * APPROACH_DISTANCE_RATIO)))
    separation_px = max(3, int(round(h * SEPARATION_DISTANCE_RATIO)))
    win_radius = max(3, int(round(w * CROSSING_WINDOW_RADIUS_RATIO)))
    search_radius = max(3, int(round(w * CROSSING_SEARCH_RADIUS_RATIO)))
    max_win_len = max(6, int(round(w * CROSSING_MAX_WINDOW_RATIO)))

    # Resolve sign with zero propagation.
    sign = np.zeros_like(delta)
    prev = 1
    for i, d in enumerate(delta.tolist()):
        if d > 0:
            prev = 1
        elif d < 0:
            prev = -1
        sign[i] = prev

    cross_raw: list[tuple[int, int]] = []
    for i in range(1, w):
        if sign[i] == sign[i - 1]:
            continue
        local = delta[max(0, i - 1): min(w, i + 2)]
        if local.size == 0:
            continue
        if int(np.min(np.abs(local))) > approach_px:
            continue
        left = np.abs(delta[max(0, i - search_radius):i])
        right = np.abs(delta[i:min(w, i + search_radius)])
        left_sep = int(np.max(left)) if left.size else 0
        right_sep = int(np.max(right)) if right.size else 0
        if left_sep < separation_px or right_sep < separation_px:
            continue
        s = max(0, i - win_radius)
        e = min(w - 1, i + win_radius)
        if e - s + 1 > max_win_len:
            center = i
            half = max_win_len // 2
            s = max(0, center - half)
            e = min(w - 1, s + max_win_len - 1)
        cross_raw.append((s, e))

    crossing_windows = _merge_windows(cross_raw, CROSSING_MERGE_GAP)

    # Close + ambiguous windows that are not true crossings.
    mid = ((y_a + y_b) // 2).clip(0, h - 1)
    amb = ambiguity[mid, np.arange(w)]
    close_amb = (np.abs(delta) <= (approach_px * 2)) & (amb < LOW_AMBIGUITY_THRESHOLD)
    parallel_windows = _merge_windows(_mask_to_windows(close_amb.astype(np.bool_)), CROSSING_MERGE_GAP)

    # Remove overlaps from parallel set.
    cleaned_parallel: list[tuple[int, int]] = []
    for s, e in parallel_windows:
        overlaps = False
        for cs, ce in crossing_windows:
            if not (e < cs or s > ce):
                overlaps = True
                break
        if not overlaps:
            cleaned_parallel.append((s, e))
    return crossing_windows, cleaned_parallel


def _swap_necessity_mask(
    candidates_a: list[list[tuple[int, float]]],
    candidates_b: list[list[tuple[int, float]]],
    crossing_windows: list[tuple[int, int]],
    expected_sign: int,
    height: int,
) -> NDArray[np.bool_]:
    """Allow swaps only where swapped ordering has clear local evidence gain."""
    w = len(candidates_a)
    allow = np.zeros((w,), dtype=np.bool_)
    if not crossing_windows:
        return allow

    coll_scale = max(2, int(round(height * COLLISION_DISTANCE_RATIO)))
    for s, e in crossing_windows:
        for x in range(s, e + 1):
            best_same = float("inf")
            best_swap = float("inf")
            for ya, sa in candidates_a[x]:
                for yb, sb in candidates_b[x]:
                    coll = max(0.0, 1.0 - (abs(ya - yb) / float(coll_scale)))
                    unary = -float(sa + sb) + 0.25 * coll
                    sign = -1 if ya < yb else 1
                    if sign == expected_sign:
                        if unary < best_same:
                            best_same = unary
                    else:
                        if unary < best_swap:
                            best_swap = unary
            if best_swap + SWAP_GAIN_MARGIN < best_same:
                allow[x] = True
    return allow


def _trace_single(
    score_map: NDArray[np.float32],
    axis_penalty_map: NDArray[np.float32],
    ambiguity_map: NDArray[np.float32],
    direction: DirectionMode,
    occupied_penalty: NDArray[np.float32] | None,
    candidate_mask: NDArray[np.bool_] | None,
    cfg: TraceConfig,
) -> tuple[list[int], dict[str, float]]:
    """Trace one arm path with dynamic programming."""
    h, w = score_map.shape
    candidates_by_x: list[list[tuple[int, float]]] = []
    for x in range(w):
        hard = candidate_mask[:, x] if candidate_mask is not None else None
        cands = _column_candidates(
            score_map[:, x],
            top_k=MAX_CANDIDATES_PER_COLUMN,
            min_score=MIN_COLUMN_SCORE,
            hard_mask=hard,
        )
        candidates_by_x.append(cands)

    costs: list[NDArray[np.float32]] = []
    parents: list[NDArray[np.int32]] = []
    first = candidates_by_x[0]
    first_cost = np.asarray(
        [
            -score
            + (AXIS_NEAR_WEIGHT * cfg.axis_multiplier) * _axis_penalty_value(
                axis_penalty_map, y, 0, direction, h
            )
            + _start_anchor_penalty(y, 0, w, h, direction, cfg)
            + (float(occupied_penalty[y, 0]) if occupied_penalty is not None else 0.0)
            for y, score in first
        ],
        dtype=np.float32,
    )
    costs.append(first_cost)
    parents.append(np.full(first_cost.shape, -1, dtype=np.int32))

    for x in range(1, w):
        prev_cands = candidates_by_x[x - 1]
        curr_cands = candidates_by_x[x]
        prev_cost = costs[-1]
        curr_cost = np.full((len(curr_cands),), np.inf, dtype=np.float32)
        curr_parent = np.full((len(curr_cands),), -1, dtype=np.int32)
        for j, (y_cur, score_cur) in enumerate(curr_cands):
            unary = (
                -float(score_cur)
                + (AXIS_NEAR_WEIGHT * cfg.axis_multiplier) * _axis_penalty_value(
                    axis_penalty_map, y_cur, x, direction, h
                )
                + _start_anchor_penalty(y_cur, x, w, h, direction, cfg)
                + (float(occupied_penalty[y_cur, x]) if occupied_penalty is not None else 0.0)
            )
            best_value = np.inf
            best_idx = -1
            for i, (y_prev, _) in enumerate(prev_cands):
                smooth = (
                    SMOOTHNESS_WEIGHT
                    * cfg.smoothness_multiplier
                    * abs(y_cur - y_prev)
                    / float(max(1, h))
                )
                dir_pen = _direction_penalty(y_prev, y_cur, direction, h, cfg)
                total = float(prev_cost[i]) + unary + smooth + dir_pen
                if total < best_value:
                    best_value = total
                    best_idx = i
            curr_cost[j] = np.float32(best_value)
            curr_parent[j] = np.int32(best_idx)
        costs.append(curr_cost)
        parents.append(curr_parent)

    ys = [0] * w
    idx = int(np.argmin(costs[-1]))
    for x in range(w - 1, -1, -1):
        y, _ = candidates_by_x[x][idx]
        ys[x] = int(y)
        if x > 0:
            idx = int(parents[x][idx])
            if idx < 0:
                idx = 0

    return ys, _extract_arm_diagnostics(ys, score_map, axis_penalty_map, ambiguity_map, direction)


def _trace_joint_two(
    name_a: str,
    name_b: str,
    map_a: NDArray[np.float32],
    map_b: NDArray[np.float32],
    axis_penalty_map: NDArray[np.float32],
    ambiguity_map: NDArray[np.float32],
    direction: DirectionMode,
    candidate_mask: NDArray[np.bool_] | None,
    cfg: TraceConfig,
) -> tuple[dict[str, list[int]], dict[str, dict[str, float]], list[str], list[tuple[int, int]]]:
    """Joint DP for two curves with strict, localized swap handling."""
    h, w = map_a.shape
    warnings: list[str] = []

    candidates_a = [
        _column_candidates(
            map_a[:, x],
            top_k=MAX_CANDIDATES_PER_COLUMN,
            min_score=MIN_COLUMN_SCORE,
            hard_mask=(candidate_mask[:, x] if candidate_mask is not None else None),
        )
        for x in range(w)
    ]
    candidates_b = [
        _column_candidates(
            map_b[:, x],
            top_k=MAX_CANDIDATES_PER_COLUMN,
            min_score=MIN_COLUMN_SCORE,
            hard_mask=(candidate_mask[:, x] if candidate_mask is not None else None),
        )
        for x in range(w)
    ]

    # Initial deterministic expected ordering.
    y0a = candidates_a[0][0][0]
    y0b = candidates_b[0][0][0]
    if y0a == y0b:
        expected_sign = -1 if name_a < name_b else 1
    else:
        expected_sign = -1 if y0a < y0b else 1

    crossing_windows, parallel_windows = _detect_crossing_windows(map_a, map_b, ambiguity_map)
    total_cross_cols = sum(e - s + 1 for s, e in crossing_windows)
    cross_cap_cols = int(round(MAX_CROSSING_COL_RATIO * w))
    crossing_disabled = False
    if len(crossing_windows) > MAX_CROSSING_WINDOWS or total_cross_cols > cross_cap_cols:
        crossing_disabled = True
        warnings.append(
            f"W_CROSSING_DISABLED_CAP:{len(crossing_windows)}:{total_cross_cols}:{w}"
        )
        crossing_windows = []

    for s, e in parallel_windows:
        warnings.append(f"W_CLOSE_PARALLEL_AMBIGUOUS:{s}:{e}")
    for s, e in crossing_windows:
        warnings.append(f"W_CROSSING_AMBIGUOUS:{s}:{e}")

    allow_swap = (
        _swap_necessity_mask(candidates_a, candidates_b, crossing_windows, expected_sign, h)
        if not crossing_disabled
        else np.zeros((w,), dtype=np.bool_)
    )

    states_by_x: list[list[tuple[int, int, float]]] = []
    for x in range(w):
        raw_states: list[tuple[int, int, float]] = []
        coll_scale = max(2, int(round(h * COLLISION_DISTANCE_RATIO)))
        for (ya, sa), (yb, sb) in product(candidates_a[x], candidates_b[x]):
            col_dist = abs(ya - yb)
            collision_penalty = max(0.0, 1.0 - (float(col_dist) / float(coll_scale)))
            unary = (
                -float(sa + sb)
                + (AXIS_NEAR_WEIGHT * cfg.axis_multiplier) * _axis_penalty_value(
                    axis_penalty_map, ya, x, direction, h
                )
                + (AXIS_NEAR_WEIGHT * cfg.axis_multiplier) * _axis_penalty_value(
                    axis_penalty_map, yb, x, direction, h
                )
                + _start_anchor_penalty(ya, x, w, h, direction, cfg)
                + _start_anchor_penalty(yb, x, w, h, direction, cfg)
                + 0.25 * collision_penalty
            )
            raw_states.append((ya, yb, unary))
        raw_states.sort(key=lambda s: (s[2], s[0], s[1]))
        states_by_x.append(raw_states[:JOINT_MAX_STATES])

    costs: list[NDArray[np.float32]] = []
    parents: list[NDArray[np.int32]] = []
    costs.append(np.asarray([s[2] for s in states_by_x[0]], dtype=np.float32))
    parents.append(np.full((len(states_by_x[0]),), -1, dtype=np.int32))

    for x in range(1, w):
        prev_states = states_by_x[x - 1]
        curr_states = states_by_x[x]
        prev_cost = costs[-1]
        curr_cost = np.full((len(curr_states),), np.inf, dtype=np.float32)
        curr_parent = np.full((len(curr_states),), -1, dtype=np.int32)
        local_allow_swap = bool(allow_swap[x] or allow_swap[x - 1])
        for j, (ya, yb, unary) in enumerate(curr_states):
            best_value = np.inf
            best_idx = -1
            cur_sign = -1 if ya < yb else 1
            for i, (pya, pyb, _) in enumerate(prev_states):
                smooth = (
                    SMOOTHNESS_WEIGHT
                    * cfg.smoothness_multiplier
                    * (abs(ya - pya) + abs(yb - pyb))
                    / float(max(1, h))
                )
                dpen = _direction_penalty(pya, ya, direction, h, cfg) + _direction_penalty(
                    pyb, yb, direction, h, cfg
                )
                order_pen = 0.0
                if cur_sign != expected_sign:
                    if local_allow_swap:
                        order_pen = SWAP_ALLOWED_PENALTY * cfg.swap_multiplier
                    else:
                        order_pen = ORDER_LOCK_PENALTY * cfg.order_lock_multiplier
                total = float(prev_cost[i]) + unary + smooth + dpen + order_pen
                if total < best_value:
                    best_value = total
                    best_idx = i
            curr_cost[j] = np.float32(best_value)
            curr_parent[j] = np.int32(best_idx)
        costs.append(curr_cost)
        parents.append(curr_parent)

    idx = int(np.argmin(costs[-1]))
    ys_a: list[int] = [0] * w
    ys_b: list[int] = [0] * w
    for x in range(w - 1, -1, -1):
        ya, yb, _ = states_by_x[x][idx]
        ys_a[x] = int(ya)
        ys_b[x] = int(yb)
        if x > 0:
            idx = int(parents[x][idx])
            if idx < 0:
                idx = 0

    diag_a = _extract_arm_diagnostics(ys_a, map_a, axis_penalty_map, ambiguity_map, direction)
    diag_b = _extract_arm_diagnostics(ys_b, map_b, axis_penalty_map, ambiguity_map, direction)

    # Overlap conflict diagnostic.
    coll_px = max(2, int(round(h * COLLISION_DISTANCE_RATIO)))
    conflict = 0
    for x in range(w):
        if abs(ys_a[x] - ys_b[x]) <= coll_px:
            y_mid = int(np.clip((ys_a[x] + ys_b[x]) // 2, 0, h - 1))
            if float(ambiguity_map[y_mid, x]) < LOW_AMBIGUITY_THRESHOLD:
                conflict += 1
    overlap_conflict_frac = float(conflict) / float(max(1, w))
    diag_a["overlap_conflict_frac"] = overlap_conflict_frac
    diag_b["overlap_conflict_frac"] = overlap_conflict_frac

    return (
        {name_a: ys_a, name_b: ys_b},
        {name_a: diag_a, name_b: diag_b},
        warnings,
        crossing_windows,
    )


def trace_curves(
    arm_score_maps: dict[str, NDArray[np.float32]],
    evidence: EvidenceCube,
    direction: CurveDirection,
    direction_confidence: float,
    x0: int,
    y0: int,
    trace_config: TraceConfig | None = None,
    candidate_mask: NDArray[np.bool_] | None = None,
) -> TraceResult:
    """Trace all arms from score maps."""
    cfg = trace_config or TraceConfig()
    warnings: list[str] = []
    names = sorted(arm_score_maps)
    if not names:
        return TraceResult(
            pixel_curves={},
            confidence_by_arm={},
            arm_diagnostics={},
            plot_confidence=0.0,
            warning_codes=("W_NO_ARMS_FOR_TRACING",),
            crossing_windows=(),
        )

    first_map = arm_score_maps[names[0]]
    h, w = first_map.shape
    axis_pen = evidence.axis_penalty_map
    ambiguity = evidence.ambiguity_map

    y_paths: dict[str, list[int]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    crossing_windows: list[tuple[int, int]] = []

    if len(names) == 2:
        joint_paths, diag_map, joint_warnings, windows = _trace_joint_two(
            names[0],
            names[1],
            arm_score_maps[names[0]],
            arm_score_maps[names[1]],
            axis_pen,
            ambiguity,
            direction,
            candidate_mask=candidate_mask,
            cfg=cfg,
        )
        y_paths.update(joint_paths)
        diagnostics.update(diag_map)
        warnings.extend(joint_warnings)
        crossing_windows.extend(windows)
    else:
        occupied = np.zeros((h, w), dtype=np.float32)
        occ_band = max(1, int(round(h * 0.015)))
        single_arm = len(names) == 1
        single_cfg = (
            TraceConfig(
                axis_multiplier=cfg.axis_multiplier * 1.4,
                smoothness_multiplier=cfg.smoothness_multiplier * 1.2,
                direction_multiplier=cfg.direction_multiplier * 1.4,
                order_lock_multiplier=cfg.order_lock_multiplier,
                swap_multiplier=cfg.swap_multiplier,
            )
            if single_arm
            else cfg
        )
        for name in names:
            ys, diag = _trace_single(
                arm_score_maps[name],
                axis_pen,
                ambiguity,
                direction,
                occupied_penalty=occupied,
                candidate_mask=candidate_mask,
                cfg=single_cfg,
            )
            y_paths[name] = ys
            diagnostics[name] = diag
            for x, y in enumerate(ys):
                y_lo = max(0, y - occ_band)
                y_hi = min(h, y + occ_band + 1)
                occupied[y_lo:y_hi, x] += 0.18

    pixel_curves: dict[str, list[tuple[int, int]]] = {}
    confidence: dict[str, float] = {}
    for name in names:
        ys = y_paths.get(name, [])
        pts = [(int(x0 + x), int(y0 + y)) for x, y in enumerate(ys)]
        pixel_curves[name] = pts
        diag = diagnostics.get(name, {})
        conf = _confidence_from_diagnostics(diag, direction_confidence=direction_confidence)
        confidence[name] = conf
        if conf < 0.45:
            warnings.append(f"W_LOW_ARM_CONFIDENCE:{name}:{conf:.3f}")
        if float(diag.get("axis_capture_frac", 0.0)) > 0.03:
            warnings.append(f"W_AXIS_CAPTURE_HIGH:{name}:{diag['axis_capture_frac']:.3f}")
        if float(diag.get("low_margin_frac", 0.0)) > 0.35:
            warnings.append(f"W_LOW_MARGIN_HIGH:{name}:{diag['low_margin_frac']:.3f}")
        if float(diag.get("monotone_violation_mass", 0.0)) > 0.08:
            warnings.append(
                f"W_MONOTONE_VIOLATION_MASS:{name}:{diag['monotone_violation_mass']:.3f}"
            )

    plot_confidence = (
        float(np.mean(np.asarray(list(confidence.values()), dtype=np.float32)))
        if confidence
        else 0.0
    )
    return TraceResult(
        pixel_curves=pixel_curves,
        confidence_by_arm=confidence,
        arm_diagnostics=diagnostics,
        plot_confidence=plot_confidence,
        warning_codes=tuple(warnings),
        crossing_windows=tuple(crossing_windows),
    )
