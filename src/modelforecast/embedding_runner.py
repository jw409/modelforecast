"""CLI runner for embedding probes (E0, E1, E2).

Supports both OpenRouter API and local TalentOS service (localhost:8765).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from modelforecast import __version__
from modelforecast.clients.local_embedding import LocalEmbeddingClient, LocalRerankClient
from modelforecast.probes.base import EmbeddingResult
from modelforecast.probes.e0_invoke import E0InvokeProbe
from modelforecast.probes.e1_retrieval import E1RetrievalProbe
from modelforecast.probes.e2_rerank import E2RerankProbe, RerankResult
from modelforecast.stats.confidence import wilson_interval
from modelforecast.verification.provenance import ProvenanceTracker


# Available OpenRouter embedding models (OpenAI-compatible API)
EMBEDDING_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "google/gemini-embedding-001",
    "mistralai/mistral-embed-2312",
]

# Local TalentOS models (localhost:8765)
LOCAL_MODELS = {
    "embedding": ["qwen3_4b"],
    "reranker": ["qwen3_reranker"],
}


class EmbeddingRunner:
    """Runner for embedding capability probes."""

    def __init__(
        self,
        output_dir: Path,
        model: str,
        contributor: str | None = None,
        local: bool = False,
        local_url: str = "http://localhost:8765",
    ):
        """Initialize embedding runner.

        Args:
            output_dir: Directory to write results to
            model: Embedding model identifier
            contributor: GitHub username for provenance (defaults to env var or "unknown")
            local: If True, use local TalentOS service instead of OpenRouter
            local_url: Base URL for local service (default: http://localhost:8765)
        """
        self.output_dir = Path(output_dir)
        self.model = model
        self.contributor = contributor or os.getenv("GITHUB_USERNAME", "unknown")
        self.console = Console()
        self.local = local
        self.local_url = local_url

        if local:
            # Initialize local clients
            self.client = LocalEmbeddingClient(base_url=local_url)
            self.rerank_client = LocalRerankClient(base_url=local_url)

            # Check health
            if not self.client.health_check():
                raise ValueError(f"Local embedding service not available at {local_url}")
        else:
            # Initialize OpenAI client with OpenRouter
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.rerank_client = None  # No reranker for OpenRouter

        # Initialize probes
        self.probes = {
            0: E0InvokeProbe(),
            1: E1RetrievalProbe(),
            2: E2RerankProbe(),  # E2 reranking (local only)
        }

    def run_level(
        self,
        level: int,
        trials: int = 10,
    ) -> dict:
        """Run all trials for a specific embedding level.

        Args:
            level: Probe level (0, 1, or 2)
            trials: Number of trials to run

        Returns:
            Complete result dictionary with provenance
        """
        if level not in self.probes:
            self.console.print(f"[yellow]Level {level} probe not implemented[/yellow]")
            return None

        # E2 (reranking) requires local mode
        if level == 2 and not self.local:
            self.console.print(
                "[yellow]Level 2 (Reranking) requires --local flag "
                "(uses localhost:8765 reranker)[/yellow]"
            )
            return None

        probe = self.probes[level]
        tracker = ProvenanceTracker(contributor=self.contributor)

        mode_label = "local" if self.local else "openrouter"
        self.console.print(f"\n[bold]Testing {self.model} ({mode_label}) - {probe}[/bold]")

        results = []
        trial_records = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {trials} trials...", total=trials)

            for trial_num in range(trials):
                # E2 uses rerank client, others use embedding client
                if level == 2:
                    result = probe.run(self.model, self.rerank_client)
                else:
                    result = probe.run(self.model, self.client)
                results.append(result)

                # Create trial record for provenance
                if level == 2:
                    # RerankResult structure
                    trial_record = {
                        "ranking_correct": result.ranking_correct,
                        "scores": result.scores,
                        "success": result.success,
                        "latency_ms": result.latency_ms,
                        "margin": result.margin,
                        "error": result.error,
                    }
                else:
                    # EmbeddingResult structure
                    trial_record = {
                        "embedding_returned": result.embedding_returned,
                        "dimensions": result.dimensions,
                        "success": result.success,
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                    }

                    # Add similarity score for E1
                    if result.similarity_score is not None:
                        trial_record["similarity_score"] = result.similarity_score

                # Add raw_response metadata (without full embeddings to save space)
                if result.raw_response:
                    trial_record["raw_response"] = result.raw_response

                trial_records.append(trial_record)

                # Update progress
                status = "✓" if result.success else "✗"
                progress.update(
                    task,
                    advance=1,
                    description=f"Trial {trial_num + 1}/{trials} {status}",
                )

        # Calculate statistics
        successes = sum(1 for r in results if r.success)
        ci = wilson_interval(successes, trials)

        # Create complete result with provenance
        complete_result = tracker.create_result(
            model=self.model,
            level=level,
            trials=trial_records,
            successes=successes,
            wilson_ci=ci,
        )

        # Add embedding-specific metadata
        if level == 2:
            # Reranking metadata
            complete_result["embedding_metadata"] = {
                "probe_type": "E2",
                "probe_name": probe.name,
                "reranker_model": self.model,
                "local_url": self.local_url,
            }
        else:
            complete_result["embedding_metadata"] = {
                "probe_type": "E" + str(level),
                "probe_name": probe.name,
                "dimensions": results[0].dimensions if results and hasattr(results[0], 'dimensions') and results[0].dimensions else None,
                "mode": "local" if self.local else "openrouter",
            }

        # Print summary
        rate_pct = int(complete_result["summary"]["rate"] * 100)
        ci_lower = int(ci[0] * 100)
        ci_upper = int(ci[1] * 100)
        self.console.print(
            f"[bold green]Result: {successes}/{trials} ({rate_pct}%) "
            f"CI=[{ci_lower},{ci_upper}][/bold green]"
        )

        # For E1, show average similarity margin
        if level == 1 and results:
            margins = [
                r.raw_response.get("margin_relevant_vs_distractor", 0)
                for r in results
                if r.success and r.raw_response
            ]
            if margins:
                avg_margin = sum(margins) / len(margins)
                self.console.print(
                    f"[bold]Average discrimination margin: {avg_margin:.4f}[/bold]"
                )

        # For E2, show average reranking margin
        if level == 2 and results:
            margins = [r.margin for r in results if r.success and r.margin is not None]
            if margins:
                avg_margin = sum(margins) / len(margins)
                self.console.print(
                    f"[bold]Average reranking margin: {avg_margin:.4f}[/bold]"
                )

        return complete_result

    def write_result(self, level: int, result: dict) -> Path:
        """Write result to JSON file.

        Args:
            level: Probe level
            result: Complete result dictionary

        Returns:
            Path to written JSON file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from model name
        model_slug = self.model.replace("/", "_").replace(":", "_").replace(".", "-")
        filename = f"{model_slug}__embedding_level_{level}.json"

        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        return output_file


def main():
    """Main entry point for embedding runner."""
    parser = argparse.ArgumentParser(
        description="ModelForecast - Embedding capability benchmarks"
    )
    parser.add_argument("--version", action="version", version=f"modelforecast {__version__}")
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model to test (e.g., openai/text-embedding-3-small or qwen3_4b for local)",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[0, 1, 2],
        help="Embedding probe level: 0=Basic Invoke, 1=Retrieval Quality, 2=Reranking (local only)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per probe (default: 10)",
    )
    parser.add_argument(
        "--contributor",
        type=str,
        help="GitHub username for provenance (default: GITHUB_USERNAME env var)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available embedding models and exit",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local TalentOS service (localhost:8765) instead of OpenRouter",
    )
    parser.add_argument(
        "--local-url",
        type=str,
        default="http://localhost:8765",
        help="Base URL for local embedding service (default: http://localhost:8765)",
    )

    args = parser.parse_args()

    print(f"ModelForecast Embedding Runner v{__version__}")

    # Handle --list-models (doesn't need API key)
    if args.list_models:
        print("\nOpenRouter embedding models (via API):\n")
        for model in EMBEDDING_MODELS:
            print(f"  {model}")
        print("\nLocal TalentOS models (--local flag):\n")
        print("  Embedding:")
        for model in LOCAL_MODELS["embedding"]:
            print(f"    {model}")
        print("  Reranking (E2 probe):")
        for model in LOCAL_MODELS["reranker"]:
            print(f"    {model}")
        print(
            "\nNote: Use --local flag for local models at localhost:8765"
        )
        return 0

    # Check for API key or local mode
    if not args.local and not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key from: https://openrouter.ai/keys")
        print("Or use --local flag for local TalentOS service")
        return 1

    # Validate required arguments when not listing models
    if not args.model:
        print("ERROR: --model is required")
        parser.print_usage()
        return 1
    if args.level is None:
        print("ERROR: --level is required")
        parser.print_usage()
        return 1

    # Initialize output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize runner
    try:
        runner = EmbeddingRunner(
            output_dir=output_dir,
            model=args.model,
            contributor=args.contributor,
            local=args.local,
            local_url=args.local_url,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    try:
        # Run the specified level
        result = runner.run_level(args.level, args.trials)

        if result:
            # Write result to file
            output_file = runner.write_result(args.level, result)
            print(f"\n[✓] Results written to: {output_file}")
            return 0
        else:
            print(f"\nERROR: Level {args.level} probe not implemented")
            return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
