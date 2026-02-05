"""CLI for KM curve extraction."""

import json
import sys
from pathlib import Path

import click

from km_estimator.models import PipelineConfig
from km_estimator.pipeline import run_pipeline


@click.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output JSON file (single image)")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Output directory (batch mode)")
@click.option("--max-retries", type=int, default=3, help="Max validation retries")
@click.option("--convergence-threshold", type=float, default=0.9, help="Model convergence threshold")
@click.option("--target-resolution", type=int, default=2000, help="Target image resolution")
@click.option("--single-model", is_flag=True, help="Use single model mode (Pro only)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    images: tuple[Path, ...],
    output: Path | None,
    output_dir: Path | None,
    max_retries: int,
    convergence_threshold: float,
    target_resolution: int,
    single_model: bool,
    verbose: bool,
) -> None:
    """Extract IPD from Kaplan-Meier curve images."""
    if not images:
        click.echo("Error: No input images provided", err=True)
        sys.exit(1)

    batch = len(images) > 1 or output_dir is not None

    if batch and output:
        click.echo("Error: Use --output-dir for batch processing", err=True)
        sys.exit(1)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        max_validation_retries=max_retries,
        convergence_threshold=convergence_threshold,
        target_resolution=target_resolution,
        single_model_mode=single_model,
    )

    success_count = 0
    fail_count = 0

    for img_path in images:
        if verbose:
            click.echo(f"Processing: {img_path}")

        try:
            state = run_pipeline(str(img_path), config)
        except Exception as e:
            fail_count += 1
            click.echo(f"Error processing {img_path}: {e}", err=True)
            continue

        # Determine output path
        if batch:
            out_path = (output_dir or img_path.parent) / f"{img_path.stem}.json"
        else:
            out_path = output or Path(f"{img_path.stem}.json")

        if state.output:
            result = state.output.model_dump(mode="json")
            out_path.write_text(json.dumps(result, indent=2))

            if verbose:
                n_patients = sum(len(c.patients) for c in state.output.curves)
                click.echo(f"  Output: {out_path} ({n_patients} patients)")

            if state.output.warnings:
                for w in state.output.warnings:
                    click.echo(f"  Warning: {w}", err=True)

            success_count += 1
        else:
            fail_count += 1
            click.echo(f"Error processing {img_path}:", err=True)
            for e in state.errors:
                click.echo(f"  [{e.stage.value}] {e.message}", err=True)

    if batch:
        click.echo(f"Processed {success_count + fail_count} images: {success_count} success, {fail_count} failed")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
