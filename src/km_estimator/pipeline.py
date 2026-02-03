"""LangGraph pipeline for KM curve digitization."""

from langgraph.graph import END, StateGraph

from km_estimator.models import PipelineConfig, PipelineState, ProcessingStage
from km_estimator.nodes.digitization import digitize
from km_estimator.nodes.input_guard import input_guard
from km_estimator.nodes.mmpu import mmpu
from km_estimator.nodes.preprocessing import preprocess
from km_estimator.nodes.reconstruction import reconstruct, validate


def _route_input_guard(state: PipelineState) -> str:
    for err in state.errors:
        if err.stage == ProcessingStage.INPUT_GUARD and not err.recoverable:
            return END
    if state.validation_result and state.validation_result.valid:
        return "preprocess"
    return END


def _route_preprocess(state: PipelineState) -> str:
    for err in state.errors:
        if err.stage == ProcessingStage.PREPROCESS and not err.recoverable:
            return END
    if state.preprocessed_image_path:
        return "mmpu"
    return END


def _route_mmpu(state: PipelineState) -> str:
    for err in state.errors:
        if err.stage == ProcessingStage.MMPU and not err.recoverable:
            return END
    if state.plot_metadata and len(state.plot_metadata.curves) > 0:
        return "digitize"
    return END


def _route_digitize(state: PipelineState) -> str:
    for err in state.errors:
        if err.stage == ProcessingStage.DIGITIZE and not err.recoverable:
            return END
    if state.digitized_curves and len(state.digitized_curves) > 0:
        return "reconstruct"
    return END


def _route_reconstruct(state: PipelineState) -> str:
    for err in state.errors:
        if err.stage == ProcessingStage.RECONSTRUCT and not err.recoverable:
            return END
    if state.output:
        return "validate"
    return END


def _route_validate(state: PipelineState) -> str:
    cfg = state.config
    if state.output:
        passed = all(
            c.validation_mae is not None and c.validation_mae <= cfg.validation_mae_threshold
            for c in state.output.curves
        )
        if passed:
            return END
    if state.validation_retries < cfg.max_validation_retries:
        return "digitize"
    return END


def create_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("input_guard", input_guard)
    graph.add_node("preprocess", preprocess)
    graph.add_node("mmpu", mmpu)
    graph.add_node("digitize", digitize)
    graph.add_node("reconstruct", reconstruct)
    graph.add_node("validate", validate)

    graph.set_entry_point("input_guard")

    graph.add_conditional_edges(
        "input_guard", _route_input_guard, {"preprocess": "preprocess", END: END}
    )
    graph.add_conditional_edges(
        "preprocess", _route_preprocess, {"mmpu": "mmpu", END: END}
    )
    graph.add_conditional_edges("mmpu", _route_mmpu, {"digitize": "digitize", END: END})
    graph.add_conditional_edges(
        "digitize", _route_digitize, {"reconstruct": "reconstruct", END: END}
    )
    graph.add_conditional_edges(
        "reconstruct", _route_reconstruct, {"validate": "validate", END: END}
    )
    graph.add_conditional_edges(
        "validate", _route_validate, {"digitize": "digitize", END: END}
    )

    return graph.compile()


def run_pipeline(image_path: str, config: PipelineConfig | None = None) -> PipelineState:
    initial = PipelineState(image_path=image_path, config=config or PipelineConfig())
    result = create_pipeline().invoke(initial)
    return result if isinstance(result, PipelineState) else PipelineState(**result)


pipeline = create_pipeline()
