"""Pipeline nodes for KM curve processing.

This module intentionally uses lazy imports to avoid loading heavy optional
dependencies during package import.
"""

from __future__ import annotations

from km_estimator.models import PipelineState


def preprocess(state: PipelineState) -> PipelineState:
    from km_estimator.nodes.preprocessing import preprocess as _preprocess

    return _preprocess(state)


def input_guard(state: PipelineState) -> PipelineState:
    from km_estimator.nodes.input_guard import input_guard as _input_guard

    return _input_guard(state)


def mmpu(state: PipelineState) -> PipelineState:
    from km_estimator.nodes.mmpu import mmpu as _mmpu

    return _mmpu(state)


def digitize(state: PipelineState) -> PipelineState:
    from km_estimator.nodes.digitization_5 import digitize_v5 as _digitize

    return _digitize(state)


def reconstruct(state: PipelineState) -> PipelineState:
    from km_estimator.nodes.reconstruction import reconstruct as _reconstruct

    return _reconstruct(state)


def validate(state: PipelineState) -> PipelineState:
    from km_estimator.nodes.reconstruction import validate as _validate

    return _validate(state)


__all__ = [
    "digitize",
    "input_guard",
    "mmpu",
    "preprocess",
    "reconstruct",
    "validate",
]
