# ModelForecast Agent Bootloader v1.1

## Project Identity

**ModelForecast**: Empirical tool-calling benchmarks for free OpenRouter LLM models.

**Stack**: Pure Python 3.11+ | OpenRouter API | Probe sweep runner | Wilson CI statistics | No local GPU inference

**Repository**: github.com/jw409/modelforecast

---

## Architecture Contract

### What This Project IS
- OpenRouter API client for LLM tool-calling probe evaluation
- Probe dimensions: T (tool calling), R (restraint), A (agency)
- SweepOrchestrator with checkpoint-resume and timestamped output directories
- ProbeRunner with SDK-managed retry and per-model rate limiting

### What This Project IS NOT
- No local GPU inference (no 8765/8888 ports)
- No MCP server integration
- No multi-agent infrastructure
- No room coordination or agent spawning

### Execution Model
```
uv run python -m modelforecast sweep → SweepOrchestrator → ProbeRunner → OpenRouter API → results/sweep_YYYYMMDD/
```

---

## Resource Manifest

### Directories (Read/Write Boundaries)
| Path | Purpose | Access |
|------|---------|--------|
| `src/modelforecast/` | Core package | Read/Write |
| `tests/` | Test suite | Read/Write |
| `scripts/` | Utility scripts | Read/Write |
| `results/` | Sweep outputs (timestamped) | Write |
| `var/` | Runtime data | Write |
| `archive/` | Historical data | Read |
| `docs/` | Documentation | Read/Write |

### Environment
- **Virtual env**: `.venv/` (project-local)
- **Python**: `uv run python` (always, never `python` or `python3`)
- **Config**: `.mcp.json` (MCP disabled for this project)

---

## Tool Selection Decision Tree

```
Task Type                                    → Tool Choice
──────────────────────────────────────────────────────────
Run full sweep                               → uv run python -m modelforecast sweep
Check live free model roster                 → uv run python -m modelforecast sweep --validate-roster
Resume interrupted sweep                     → uv run python -m modelforecast sweep --resume
Run tests                                    → uv run pytest tests/
Lint code                                    → uv run ruff check src/
```

### OpenRouter API Usage
```python
# All LLM calls go through OpenRouter
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

**Free models** (examples — actual sweep fetches live roster dynamically):
- `qwen/qwen3-32b:free`
- `meta-llama/llama-4-maverick:free`
- `x-ai/grok-4.1-fast:free`

---

## Code Style & Patterns

### PEP 8 Enforced
- Line length: 100 (per pyproject.toml)
- Ruff linting: `uv run ruff check src/`
- Format: `uv run ruff format src/`

### Error Handling
```python
# GOOD: Explicit, actionable errors
raise ValueError(f"Model {model_id} not found in OpenRouter registry")

# BAD: Silent failures
result = api_call() or default_value  # Don't do this
```

### Logging
```python
from rich.console import Console
console = Console()
console.print("[green]Success:[/] Sweep completed")
console.print("[red]Error:[/] API rate limited", style="bold")
```

---

## Session Context Protocol

### On Session Start
Before proposing changes, verify:
1. Working directory is `/home/jw/dev/modelforecast`
2. Virtual env exists: `.venv/`
3. OpenRouter API key available: `$OPENROUTER_API_KEY`

### Context Boundaries
**DO reference**:
- This CLAUDE.md
- README.md (project overview)
- pyproject.toml (dependencies)
- src/modelforecast/ (implementation)

**DO NOT reference**:
- Other projects' CLAUDE.md files
- External infrastructure not in this repo
- Port 8765/8888 services (not running)

---

## Baseline Capabilities (Regression Prevention)

These features must always work:

1. **Sweep execution**: `uv run python -m modelforecast sweep --trials 10` runs without errors
2. **Roster validation**: `uv run python -m modelforecast sweep --validate-roster` shows live free models
3. **Test suite passes**: `uv run pytest tests/` all green
4. **OpenRouter connectivity**: API calls succeed with valid key

### Pre-Commit Verification
Before committing changes:
```bash
uv run ruff check src/ tests/
uv run pytest tests/ -x
```

---

## Reasoning Budget

### Use Sequential Thinking For:
- Multi-model sweep strategy design
- Probe dimension taxonomy decisions
- Performance regression root cause analysis

### DO NOT Use Sequential Thinking For:
- Simple file edits
- Running existing scripts
- Reading documentation
- Status checks

### Cost Awareness
- Development: Use free models (Qwen3-32b, Llama-4-Maverick)
- Production benchmarks: Use paid models only when necessary
- Log all API calls to `var/api_calls.jsonl` for cost tracking

---

## Common Tasks

### Run a Sweep
```bash
cd /home/jw/dev/modelforecast
uv run python -m modelforecast sweep --trials 10
```

### Check Live Model Roster
```bash
uv run python -m modelforecast sweep --validate-roster
```

### Resume Interrupted Sweep
```bash
uv run python -m modelforecast sweep --resume
```

### Add a New Probe
1. Create `src/modelforecast/probes/my_probe.py`
2. Implement `ProbeProtocol` interface
3. Add test in `tests/test_probes.py`
4. Register in `src/modelforecast/probes/__init__.py`

### Debug API Issues
```bash
# Check OpenRouter status
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models | jq '.data[:3]'
```

---

## Git Conventions

- **Commits**: Author as github handles jw409/jw408
- **Branch naming**: `feat/description`, `fix/description`, `docs/description`
- **PR target**: `main` branch

---

## Anti-Patterns

### DO NOT:
1. **Import external infrastructure** - This project is standalone
2. **Reference 8765/8888 ports** - No GPU services here
3. **Create multi-agent coordination** - Single-agent only
4. **Skip preflight checks** - Validate before running sweeps

### DO:
1. **Use uv run python** - Always, for reproducibility
2. **Check API key** - Fail fast if missing
3. **Log costs** - Track API spend
4. **Run tests** - Before and after changes

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `uv run python -m modelforecast sweep` | Run full sweep |
| `uv run python -m modelforecast sweep --validate-roster` | Check live free model roster |
| `uv run python -m modelforecast sweep --resume` | Resume interrupted sweep |
| `uv run pytest tests/` | Run tests |
| `uv run ruff check src/` | Lint code |
| `cat README.md` | Project overview |

---

*Last updated: 2026-03-19*
