"""Command-line interface for Vertex AI Gemini CLI."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from pydantic import ValidationError

from .batch import BatchProcessor
from .config import Settings, get_settings
from .exceptions import VertexGeminiError
from .logger import get_logger, setup_logging
from .realtime import RealtimeInference

logger = get_logger(__name__)


def load_config() -> Optional[Settings]:
    """Load configuration and handle errors gracefully.

    Returns:
        Settings instance or None if configuration is invalid
    """
    try:
        return get_settings()
    except ValidationError as e:
        click.secho("Configuration Error:", fg="red", bold=True)
        click.secho(str(e), fg="red")
        click.echo("\nPlease check your .env file or environment variables.")
        return None


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set logging level",
)
@click.option("--json-logs", is_flag=True, help="Output logs in JSON format")
@click.pass_context
def cli(ctx: click.Context, log_level: str, json_logs: bool) -> None:
    """Vertex AI Gemini CLI - Enterprise-grade client for Google's Gemini models.

    \b
    Features:
    - Real-time inference with automatic retries
    - Batch processing with GCS integration
    - Comprehensive error handling and logging
    - Configuration via environment variables or .env file

    \b
    Examples:
        # Real-time inference
        vertex-gemini-cli realtime "Explain quantum computing"

        # Batch processing
        vertex-gemini-cli batch process prompts.txt --output gs://bucket/output/

    Get started by creating a .env file with your GCP configuration.
    """
    setup_logging(log_level=log_level, json_logs=json_logs)
    ctx.ensure_object(dict)


@cli.group()
def realtime() -> None:
    """Real-time inference commands."""
    pass


@realtime.command(name="predict")
@click.argument("prompt", required=False)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True, path_type=Path),
    help="Read prompt from file",
)
@click.option(
    "--temperature", "-t",
    type=float,
    help="Model temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum output tokens",
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output response in JSON format with metadata",
)
def realtime_predict(
    prompt: Optional[str],
    file: Optional[Path],
    temperature: Optional[float],
    max_tokens: Optional[int],
    json_output: bool,
) -> None:
    """Make a real-time prediction with Gemini.

    \b
    Examples:
        vertex-gemini-cli realtime predict "What is AI?"
        vertex-gemini-cli realtime predict -f prompt.txt
        vertex-gemini-cli realtime predict "Summarize this" -t 0.3 --json-output
    """
    try:
        # Load configuration
        settings = load_config()
        if not settings:
            sys.exit(1)

        # Get prompt from argument or file
        if file:
            prompt_text = file.read_text().strip()
        elif prompt:
            prompt_text = prompt
        else:
            click.secho("Error: Either PROMPT or --file must be provided", fg="red")
            sys.exit(1)

        # Create inference handler
        inference = RealtimeInference(settings)

        # Make prediction
        with click.progressbar(
            length=1,
            label="Generating response",
            show_eta=False,
        ) as bar:
            if json_output:
                result = inference.predict_with_metadata(
                    prompt=prompt_text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                bar.update(1)
                click.echo(json.dumps(result, indent=2))
            else:
                response = inference.predict(
                    prompt=prompt_text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                bar.update(1)
                click.echo("\n" + "=" * 80)
                click.secho("Response:", fg="green", bold=True)
                click.echo("=" * 80)
                click.echo(response)
                click.echo("=" * 80)

    except VertexGeminiError as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in realtime prediction")
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@cli.group()
def batch() -> None:
    """Batch processing commands."""
    pass


@batch.command(name="prepare")
@click.argument("prompts_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default="batch_input.jsonl",
    help="Output JSONL file path",
)
def batch_prepare(prompts_file: Path, output: Path) -> None:
    """Prepare batch input file from a list of prompts.

    PROMPTS_FILE should contain one prompt per line.

    \b
    Example:
        vertex-gemini-cli batch prepare prompts.txt -o batch_input.jsonl
    """
    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        # Read prompts
        prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]

        if not prompts:
            click.secho("Error: No prompts found in file", fg="red")
            sys.exit(1)

        # Create batch processor
        processor = BatchProcessor(settings)

        # Prepare input file
        processor.prepare_batch_input(prompts, local_path=output)

        click.secho(f"✓ Batch input prepared: {output}", fg="green")
        click.echo(f"  Total prompts: {len(prompts)}")

    except Exception as e:
        logger.exception("Error preparing batch input")
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@batch.command(name="submit")
@click.argument("input_uri")
@click.argument("output_uri")
@click.option(
    "--job-name",
    help="Display name for the batch job",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for job completion",
)
def batch_submit(
    input_uri: str,
    output_uri: str,
    job_name: Optional[str],
    wait: bool,
) -> None:
    """Submit a batch prediction job.

    \b
    INPUT_URI: GCS URI of input JSONL file (gs://bucket/input.jsonl)
    OUTPUT_URI: GCS URI for output directory (gs://bucket/output/)

    \b
    Example:
        vertex-gemini-cli batch submit gs://bucket/input.jsonl gs://bucket/output/
    """
    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        processor = BatchProcessor(settings)

        # Submit job
        click.echo("Submitting batch job...")
        batch_job = processor.submit_batch_job(
            input_uri=input_uri,
            output_uri=output_uri,
            job_display_name=job_name,
        )

        click.secho("✓ Batch job submitted", fg="green")
        click.echo(f"  Job ID: {batch_job.resource_name}")
        click.echo(f"  State: {batch_job.state.name}")

        # Wait for completion if requested
        if wait:
            click.echo("\nWaiting for job completion...")
            with click.progressbar(
                length=100,
                label="Processing",
                show_eta=False,
            ) as bar:
                completed_job = processor.wait_for_completion(batch_job)
                bar.update(100)

            click.secho("\n✓ Batch job completed", fg="green")
            click.echo(f"  Final state: {completed_job.state.name}")
            click.echo(f"  Output: {output_uri}")

    except VertexGeminiError as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        logger.exception("Error submitting batch job")
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@batch.command(name="process")
@click.argument("prompts_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--input-uri",
    required=True,
    help="GCS URI for input file (gs://bucket/input.jsonl)",
)
@click.option(
    "--output-uri",
    required=True,
    help="GCS URI for output directory (gs://bucket/output/)",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for job completion",
)
def batch_process(
    prompts_file: Path,
    input_uri: str,
    output_uri: str,
    wait: bool,
) -> None:
    """Process a batch of prompts end-to-end.

    This command handles the entire workflow: prepare input, upload to GCS,
    submit job, and optionally wait for completion.

    \b
    Example:
        vertex-gemini-cli batch process prompts.txt \\
            --input-uri gs://bucket/input.jsonl \\
            --output-uri gs://bucket/output/
    """
    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        # Read prompts
        prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]

        if not prompts:
            click.secho("Error: No prompts found in file", fg="red")
            sys.exit(1)

        click.echo(f"Processing {len(prompts)} prompts...")

        # Create processor and run batch
        processor = BatchProcessor(settings)
        result = processor.process_batch(
            prompts=prompts,
            input_uri=input_uri,
            output_uri=output_uri,
            wait_for_completion=wait,
        )

        click.secho("\n✓ Batch processing completed", fg="green")
        click.echo(f"  Job ID: {result['job_id']}")
        click.echo(f"  State: {result['state']}")
        click.echo(f"  Output: {result['output_uri']}")

    except VertexGeminiError as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        logger.exception("Error processing batch")
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@cli.group()
def loadtest() -> None:
    """Load testing commands for QPS benchmarking."""
    pass


@loadtest.command(name="realtime")
@click.option(
    "--rpm",
    type=float,
    default=10.0,
    show_default=True,
    help="Target requests per minute (matches Vertex AI quota units)",
)
@click.option(
    "--duration",
    type=float,
    default=0,
    help="Test duration in seconds (use --duration or --max-requests)",
)
@click.option(
    "--max-requests",
    type=int,
    default=0,
    help="Total number of requests to send (use --duration or --max-requests)",
)
@click.option(
    "--concurrency",
    type=int,
    default=10,
    show_default=True,
    help="Number of concurrent worker threads",
)
@click.option(
    "--prompts-file",
    type=click.Path(exists=True, path_type=Path),
    help="File with prompts (one per line). Uses built-in test prompts if not provided.",
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results as JSON",
)
def loadtest_realtime(
    rpm: float,
    duration: float,
    max_requests: int,
    concurrency: int,
    prompts_file: Optional[Path],
    json_output: bool,
) -> None:
    """Run a real-time load test at a target RPM.

    Issues queries at the specified rate (requests per minute) using
    concurrent workers and reports latency percentiles, throughput,
    and error rates. RPM matches Vertex AI's quota units.

    \b
    Examples:
        # 10 RPM for 120 seconds
        vertex-gemini-cli loadtest realtime --rpm 10 --duration 120

        # 15 RPM, 10 total requests
        vertex-gemini-cli loadtest realtime --rpm 15 --max-requests 10

        # Custom prompts at 5 RPM
        vertex-gemini-cli loadtest realtime --rpm 5 --max-requests 10 --prompts-file prompts.txt
    """
    from .loadtest import DEFAULT_TEST_PROMPTS, LoadGenerator, format_metrics_report

    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        if duration <= 0 and max_requests <= 0:
            click.secho(
                "Error: Specify --duration (seconds) or --max-requests", fg="red"
            )
            sys.exit(1)

        # Load prompts
        if prompts_file:
            prompts = [
                line.strip()
                for line in prompts_file.read_text().splitlines()
                if line.strip()
            ]
            if not prompts:
                click.secho("Error: No prompts found in file", fg="red")
                sys.exit(1)
        else:
            prompts = DEFAULT_TEST_PROMPTS

        total = max_requests if max_requests > 0 else int((rpm / 60.0) * duration)
        click.secho(f"\nStarting load test: {total} requests @ {rpm} RPM", fg="cyan", bold=True)
        click.echo(f"  Model:       {settings.gemini_model}")
        click.echo(f"  Concurrency: {concurrency}")
        click.echo(f"  Prompts:     {len(prompts)} {'(custom)' if prompts_file else '(built-in)'}")
        click.echo("")

        generator = LoadGenerator(settings)
        metrics = generator.run_realtime_load_test(
            prompts=prompts,
            rpm=rpm,
            duration_seconds=duration,
            max_requests=max_requests,
            concurrency=concurrency,
        )

        if json_output:
            click.echo(json.dumps(metrics, indent=2))
        else:
            click.echo(format_metrics_report(metrics))

    except KeyboardInterrupt:
        click.echo("\nLoad test interrupted.")
        sys.exit(130)
    except VertexGeminiError as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in load test")
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@loadtest.command(name="batch")
@click.option(
    "--prompts-file",
    type=click.Path(exists=True, path_type=Path),
    help="File with prompts (one per line). Uses built-in test prompts if not provided.",
)
@click.option(
    "--num-prompts",
    type=int,
    default=0,
    help="Number of prompts to include (0 = all from file or all built-in)",
)
@click.option(
    "--input-uri",
    help="GCS URI for input (overrides config.yaml)",
)
@click.option(
    "--output-uri",
    help="GCS URI for output (overrides config.yaml)",
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results as JSON",
)
def loadtest_batch(
    prompts_file: Optional[Path],
    num_prompts: int,
    input_uri: Optional[str],
    output_uri: Optional[str],
    json_output: bool,
) -> None:
    """Run a batch load test by submitting prompts as a batch job.

    Prepares prompts, uploads to GCS, and submits a batch prediction
    job. Does not wait for completion (batch jobs are async).

    \b
    Examples:
        # Batch test with built-in prompts
        vertex-gemini-cli loadtest batch

        # Batch test with custom prompts
        vertex-gemini-cli loadtest batch --prompts-file prompts.txt

        # Override GCS URIs
        vertex-gemini-cli loadtest batch --input-uri gs://bucket/in/ --output-uri gs://bucket/out/
    """
    from .loadtest import DEFAULT_TEST_PROMPTS, LoadGenerator

    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        # Load prompts
        if prompts_file:
            prompts = [
                line.strip()
                for line in prompts_file.read_text().splitlines()
                if line.strip()
            ]
            if not prompts:
                click.secho("Error: No prompts found in file", fg="red")
                sys.exit(1)
        else:
            prompts = DEFAULT_TEST_PROMPTS

        if num_prompts > 0:
            # Repeat/cycle prompts to reach num_prompts
            full_prompts: List[str] = []
            for i in range(num_prompts):
                full_prompts.append(prompts[i % len(prompts)])
            prompts = full_prompts

        click.secho(f"\nStarting batch load test: {len(prompts)} prompts", fg="cyan", bold=True)
        click.echo(f"  Model:  {settings.gemini_model}")
        click.echo("")

        generator = LoadGenerator(settings)
        result = generator.run_batch_load_test(
            prompts=prompts,
            input_uri=input_uri,
            output_uri=output_uri,
        )

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("=" * 72)
            click.secho("BATCH LOAD TEST RESULTS", fg="cyan", bold=True)
            click.echo("=" * 72)
            click.echo(f"  Status:       {result['status']}")
            click.echo(f"  Job Name:     {result['job_name']}")
            click.echo(f"  Job Resource: {result['job_resource']}")
            click.echo(f"  Model:        {result['model']}")
            click.echo(f"  Prompts:      {result['num_prompts']}")
            click.echo(f"  Input URI:    {result['input_uri']}")
            click.echo(f"  Output URI:   {result['output_uri']}")
            click.echo("")
            click.secho("  Timing:", fg="yellow")
            timing = result["timing"]
            click.echo(f"    Prepare:  {timing['prepare_seconds']}s")
            click.echo(f"    Upload:   {timing['upload_seconds']}s")
            click.echo(f"    Submit:   {timing['submit_seconds']}s")
            click.echo(f"    Total:    {timing['total_seconds']}s")
            click.echo("=" * 72)
            click.echo("")
            click.echo("Batch job submitted. Check status with:")
            click.echo(f"  gcloud ai batch-prediction-jobs list --region={settings.gcp_location}")

    except VertexGeminiError as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in batch load test")
        click.secho(f"Unexpected error: {e}", fg="red")
        sys.exit(1)


@cli.group()
def quota() -> None:
    """Rate limit quota management commands."""
    pass


@quota.command(name="fetch")
def quota_fetch() -> None:
    """Fetch rate limits from the Cloud Quotas API and save to rate_limits.yaml.

    Queries the Cloud Quotas API for per-model GenerateContentRequestsPerMinute
    quotas, merges with conservative defaults, and writes the result to
    rate_limits.yaml for use by the rate limiter.

    \b
    Requires:
        - Application Default Credentials (run `gcloud auth application-default login`)
        - cloudquotas.googleapis.com API enabled on the project

    \b
    Example:
        vertex-gemini-cli quota fetch
    """
    from .quota import fetch_and_save_rate_limits

    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        click.echo(f"Fetching quota info for project: {settings.gcp_project_id}")
        click.echo(f"Location: {settings.gcp_location}")
        click.echo("")

        limits = fetch_and_save_rate_limits(
            settings,
            rate_limits_file=settings.rate_limits_file,
        )

        click.secho("Rate limits saved to rate_limits.yaml", fg="green", bold=True)
        click.echo("")
        _display_rate_limits(limits)

    except Exception as e:
        logger.exception("Error fetching quota")
        click.secho(f"Error: {e}", fg="red")
        click.echo("\nMake sure you have:")
        click.echo("  1. Run `gcloud auth application-default login`")
        click.echo("  2. Enabled cloudquotas.googleapis.com API")
        sys.exit(1)


@quota.command(name="show")
def quota_show() -> None:
    """Display current rate limits (from file or defaults).

    \b
    Example:
        vertex-gemini-cli quota show
    """
    from .quota import load_rate_limits

    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        limits = load_rate_limits(rate_limits_file=settings.rate_limits_file)

        click.secho("\nCurrent Rate Limits", fg="cyan", bold=True)
        click.echo("=" * 50)
        click.echo(f"  Rate limiting: {'enabled' if settings.rate_limit_enabled else 'disabled'}")
        click.echo(f"  Default RPM:   {settings.default_rpm}")
        click.echo(f"  Limits file:   {settings.rate_limits_file}")
        click.echo("")
        _display_rate_limits(limits)

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


def _display_rate_limits(limits: dict) -> None:
    """Helper to display rate limits in a formatted table."""
    click.secho("  Per-Model RPM Limits:", fg="yellow")
    click.echo(f"  {'Model':<30} {'RPM':>5}")
    click.echo(f"  {'-' * 30} {'-' * 5}")
    for model, config in sorted(limits.items()):
        rpm = config.get("rpm", "?") if isinstance(config, dict) else config
        click.echo(f"  {model:<30} {rpm:>5}")
    click.echo("")


@cli.command()
def info() -> None:
    """Show configuration and model information."""
    try:
        settings = load_config()
        if not settings:
            sys.exit(1)

        click.secho("\nVertex AI Gemini CLI Configuration", fg="cyan", bold=True)
        click.echo("=" * 80)

        click.secho("\nGoogle Cloud:", fg="yellow")
        click.echo(f"  Project ID: {settings.gcp_project_id}")
        click.echo(f"  Location:   {settings.gcp_location}")

        click.secho("\nModel Configuration:", fg="yellow")
        click.echo(f"  Model:            {settings.gemini_model}")
        click.echo(f"  Temperature:      {settings.temperature}")
        click.echo(f"  Max Tokens:       {settings.max_output_tokens}")
        click.echo(f"  Top-p:            {settings.top_p}")
        click.echo(f"  Top-k:            {settings.top_k}")

        click.secho("\nRetry Configuration:", fg="yellow")
        click.echo(f"  Max Retries:      {settings.max_retries}")
        click.echo(f"  Retry Delay:      {settings.retry_delay}s")

        click.secho("\nRate Limiting:", fg="yellow")
        click.echo(f"  Enabled:          {settings.rate_limit_enabled}")
        click.echo(f"  Default RPM:      {settings.default_rpm}")
        click.echo(f"  Limits File:      {settings.rate_limits_file}")

        if settings.rate_limit_enabled:
            from .quota import load_rate_limits
            model_config = load_rate_limits(
                model_name=settings.gemini_model,
                rate_limits_file=settings.rate_limits_file,
            )
            rpm = model_config.get("rpm", settings.default_rpm) if isinstance(model_config, dict) else settings.default_rpm
            click.echo(f"  Current Model RPM: {rpm} ({settings.gemini_model})")

        click.echo("=" * 80 + "\n")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
