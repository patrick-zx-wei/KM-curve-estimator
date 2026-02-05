"""CLI for KM curve extraction with concurrent processing."""

import asyncio
import json
import sys
from pathlib import Path

import click

from km_estimator import config
from km_estimator.models import PipelineConfig, PipelineState
from km_estimator.pipeline import run_pipeline_async


async def process_single_image(
    img_path: Path,
    pipeline_config: PipelineConfig,
    semaphore: asyncio.Semaphore,
    verbose: bool,
) -> tuple[Path, PipelineState | None, Exception | None]:
    """Process a single image with semaphore control."""
    async with semaphore:
        if verbose:
            click.echo(f"Processing: {img_path}")
        try:
            state = await run_pipeline_async(str(img_path), pipeline_config)
            return (img_path, state, None)
        except Exception as e:
            return (img_path, None, e)


async def process_images_concurrent(
    images: tuple[Path, ...],
    pipeline_config: PipelineConfig,
    max_concurrency: int,
    verbose: bool,
) -> list[tuple[Path, PipelineState | None, Exception | None]]:
    """Process all images concurrently with semaphore control."""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        process_single_image(img_path, pipeline_config, semaphore, verbose)
        for img_path in images
    ]
    return await asyncio.gather(*tasks)


@click.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output JSON file (single image)")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Output directory (batch mode)")
@click.option(
    "--max-concurrency",
    type=int,
    default=config.DEFAULT_MAX_CONCURRENCY,
    help="Max concurrent image processing",
)
@click.option(
    "--max-retries",
    type=int,
    default=config.MAX_VALIDATION_RETRIES,
    help="Max validation retries",
)
@click.option(
    "--convergence-threshold",
    type=float,
    default=config.CONVERGENCE_THRESHOLD,
    help="Model convergence threshold",
)
@click.option(
    "--target-resolution",
    type=int,
    default=config.TARGET_RESOLUTION,
    help="Target image resolution",
)
@click.option("--single-model", is_flag=True, help="Use single model mode (Pro only)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    images: tuple[Path, ...],
    output: Path | None,
    output_dir: Path | None,
    max_concurrency: int,
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

    pipeline_config = PipelineConfig(
        max_validation_retries=max_retries,
        convergence_threshold=convergence_threshold,
        target_resolution=target_resolution,
        single_model_mode=single_model,
    )

    # Run concurrent processing
    results = asyncio.run(
        process_images_concurrent(images, pipeline_config, max_concurrency, verbose)
    )

    success_count = 0
    fail_count = 0

    for img_path, state, error in results:
        if error:
            fail_count += 1
            click.echo(f"Error processing {img_path}: {error}", err=True)
            continue

        # Determine output path
        if batch:
            out_path = (output_dir or img_path.parent) / f"{img_path.stem}.json"
        else:
            out_path = output or Path(f"{img_path.stem}.json")

        if state and state.output:
            # Check for validation failures (output exists but validation threshold exceeded)
            has_validation_failure = any(
                e.error_type == "validation_threshold_exceeded" for e in state.errors
            )

            # Write output (still useful for inspection even if validation failed)
            result = state.output.model_dump(mode="json")
            out_path.write_text(json.dumps(result, indent=2))

            if verbose:
                n_patients = sum(len(c.patients) for c in state.output.curves)
                click.echo(f"  Output: {out_path} ({n_patients} patients)")

            if state.output.warnings:
                for w in state.output.warnings:
                    click.echo(f"  Warning: {w}", err=True)

            # Print any errors (including validation failures)
            for e in state.errors:
                click.echo(f"  [{e.stage.value}] {e.message}", err=True)

            if has_validation_failure:
                fail_count += 1  # Count as failure if validation threshold exceeded
            else:
                success_count += 1
        else:
            fail_count += 1
            click.echo(f"Error processing {img_path}:", err=True)
            if state:
                for e in state.errors:
                    click.echo(f"  [{e.stage.value}] {e.message}", err=True)

    if batch:
        click.echo(
            f"Processed {success_count + fail_count} images: "
            f"{success_count} success, {fail_count} failed"
        )

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
