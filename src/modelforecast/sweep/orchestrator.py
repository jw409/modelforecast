"""SweepOrchestrator — coordinates a full probe sweep with checkpoint-resume."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from modelforecast.runner import ProbeRunner


class SweepOrchestrator:
    """Coordinates a full sweep run across all configured models.

    Responsibilities:
    - Generate and manage sweep_id (sweep_YYYYMMDD or sweep_YYYYMMDD_N for same-day runs)
    - Create output directory: results/{sweep_id}/
    - Read checkpoint.json on resume to skip completed models
    - Write checkpoint.json after each model completes
    - Write sweep_manifest.json when all models finish
    - Inject sweep_id into ProbeRunner's output_dir

    Args:
        base_results_dir: Root results directory (default: Path("results"))
        sweep_id: Explicit sweep ID (default: auto-generated from date)
    """

    def __init__(
        self,
        base_results_dir: Path = Path("results"),
        sweep_id: str | None = None,
    ) -> None:
        self.base_results_dir = Path(base_results_dir)
        self.sweep_id = sweep_id or self._generate_sweep_id()
        self.sweep_dir = self.base_results_dir / self.sweep_id
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def _generate_sweep_id(self) -> str:
        """Generate a sweep ID from today's date. Appends _N if directory exists."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        base = f"sweep_{today}"
        if not (self.base_results_dir / base).exists():
            return base
        # Find next available suffix
        n = 2
        while (self.base_results_dir / f"{base}_{n}").exists():
            n += 1
        return f"{base}_{n}"

    @property
    def checkpoint_path(self) -> Path:
        return self.sweep_dir / "checkpoint.json"

    @property
    def manifest_path(self) -> Path:
        return self.sweep_dir / "sweep_manifest.json"

    def read_checkpoint(self) -> list[str]:
        """Read completed model list from checkpoint. Returns [] if no checkpoint."""
        if not self.checkpoint_path.exists():
            return []
        with open(self.checkpoint_path) as f:
            data = json.load(f)
        completed = data.get("completed_models", [])
        self.console.print(
            f"[cyan]Resuming sweep {self.sweep_id}: "
            f"{len(completed)} models already completed[/cyan]"
        )
        return completed

    def write_checkpoint(self, completed_models: list[str]) -> None:
        """Write checkpoint after each model completes."""
        data = {
            "sweep_id": self.sweep_id,
            "completed_models": completed_models,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def write_manifest(
        self,
        models_attempted: int,
        models_completed: int,
        trials_per_level: int,
        max_level: int,
        started_at: str,
    ) -> None:
        """Write sweep manifest on completion."""
        manifest = {
            "sweep_id": self.sweep_id,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "models_attempted": models_attempted,
            "models_completed": models_completed,
            "trials_per_level": trials_per_level,
            "max_level": max_level,
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.console.print(f"[green]✓ Sweep manifest: {self.manifest_path}[/green]")

    def run(
        self,
        runner: ProbeRunner,
        trials: int = 10,
        max_level: int = 4,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Execute the sweep with optional checkpoint-resume.

        Args:
            runner: Configured ProbeRunner (output_dir will be overridden to sweep_dir)
            trials: Trials per (model, level)
            max_level: Max probe level to run (0-4)
            resume: If True, skip models listed in checkpoint.json

        Returns:
            Dict mapping result_key -> result data for all completed results
        """
        # Override runner output_dir to sweep-stamped directory
        runner.output_dir = self.sweep_dir

        started_at = datetime.now(timezone.utc).isoformat()
        completed_models: list[str] = []

        if resume:
            completed_models = self.read_checkpoint()

        pending_models = [m for m in runner.models if m not in completed_models]
        total_models = len(runner.models)

        self.console.print(
            f"[bold blue]Sweep {self.sweep_id}[/bold blue]: "
            f"{len(pending_models)} models to run "
            f"({len(completed_models)} already completed)"
        )

        all_results: dict[str, Any] = {}

        for idx, model in enumerate(pending_models, start=len(completed_models) + 1):
            self.console.print(
                f"\n[bold cyan]Model {idx}/{total_models}: {model}[/bold cyan]"
            )
            model_results = runner.run_model(model, trials=trials, max_level=max_level)
            for level, result in model_results.items():
                all_results[f"{model}__level_{level}"] = result

            completed_models.append(model)
            self.write_checkpoint(completed_models)

        self.write_manifest(
            models_attempted=total_models,
            models_completed=len(completed_models),
            trials_per_level=trials,
            max_level=max_level,
            started_at=started_at,
        )

        return all_results
