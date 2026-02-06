"""LangGraph pipeline for KM curve digitization."""

from langgraph.graph import END, StateGraph

from km_estimator.models import PipelineConfig, PipelineState, ProcessingError, ProcessingStage
from km_estimator.nodes.digitization import digitize
from km_estimator.nodes.input_guard import input_guard, input_guard_async
from km_estimator.nodes.mmpu import mmpu, mmpu_async
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
    # Max retries exhausted with validation still failing - add warning
    if state.output:
        failed_curves = [
            c.group_name for c in state.output.curves
            if c.validation_mae is None or c.validation_mae > cfg.validation_mae_threshold
        ]
        state.errors.append(ProcessingError(
            stage=ProcessingStage.VALIDATE,
            error_type="validation_threshold_exceeded",
            recoverable=True,
            message=(
                f"Validation MAE threshold ({cfg.validation_mae_threshold}) "
                f"exceeded after {cfg.max_validation_retries} retries"
            ),
            details={"failed_curves": failed_curves},
        ))
    return END


def create_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("preprocess", preprocess)
    graph.add_node("input_guard", input_guard)
    graph.add_node("mmpu", mmpu)
    graph.add_node("digitize", digitize)
    graph.add_node("reconstruct", reconstruct)
    graph.add_node("validate", validate)

    # Preprocess first to improve input_guard accuracy on noisy/low-res images
    graph.set_entry_point("preprocess")

    graph.add_conditional_edges(
        "preprocess", _route_preprocess, {"mmpu": "input_guard", END: END}
    )
    graph.add_conditional_edges(
        "input_guard", _route_input_guard, {"preprocess": "mmpu", END: END}
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
    result = pipeline.invoke(initial)
    return result if isinstance(result, PipelineState) else PipelineState(**result)


def create_async_pipeline():
    """Create async pipeline with async nodes for concurrent processing."""
    graph = StateGraph(PipelineState)

    graph.add_node("preprocess", preprocess)
    graph.add_node("input_guard", input_guard_async)
    graph.add_node("mmpu", mmpu_async)
    graph.add_node("digitize", digitize)
    graph.add_node("reconstruct", reconstruct)
    graph.add_node("validate", validate)

    # Preprocess first to improve input_guard accuracy on noisy/low-res images
    graph.set_entry_point("preprocess")

    graph.add_conditional_edges(
        "preprocess", _route_preprocess, {"mmpu": "input_guard", END: END}
    )
    graph.add_conditional_edges(
        "input_guard", _route_input_guard, {"preprocess": "mmpu", END: END}
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


async def run_pipeline_async(
    image_path: str, config: PipelineConfig | None = None
) -> PipelineState:
    """Async pipeline runner for concurrent image processing."""
    initial = PipelineState(image_path=image_path, config=config or PipelineConfig())
    result = await async_pipeline.ainvoke(initial)
    return result if isinstance(result, PipelineState) else PipelineState(**result)


pipeline = create_pipeline()
async_pipeline = create_async_pipeline()
