"""IPD reconstruction and validation nodes."""

from km_estimator.models import (
    CurveIPD,
    IPDOutput,
    PipelineState,
    ProcessingError,
    ProcessingStage,
    ReconstructionMode,
)


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

    mode = (
        ReconstructionMode.FULL
        if state.plot_metadata.risk_table
        else ReconstructionMode.ESTIMATED
    )

    curves = []
    for name, coords in state.digitized_curves.items():
        censoring = state.censoring_marks.get(name, []) if state.censoring_marks else []
        # TODO: Guyot iKM or estimation
        curves.append(CurveIPD(group_name=name, patients=[], censoring_times=censoring))

    output = IPDOutput(
        metadata=state.plot_metadata,
        curves=curves,
        reconstruction_mode=mode,
        warnings=state.mmpu_warnings,
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

    # TODO: Re-plot IPD, calculate MAE
    # Stub: mark all curves as passing validation
    validated_curves = [
        c.model_copy(update={"validation_mae": 0.0}) for c in state.output.curves
    ]
    updated_output = state.output.model_copy(update={"curves": validated_curves})

    return state.model_copy(update={"output": updated_output})
