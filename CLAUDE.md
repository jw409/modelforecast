# ModelForecast Agent Bootloader v1.0

## Project Identity

**ModelForecast**: Competitive LLM evaluation platform using CoreWars battles. Tool-calling benchmarks for free LLM models.

**Stack**: Pure Python 3.11+ | OpenRouter API | Dagster DAGs | No local GPU inference

**Repository**: github.com/jw409/modelforecast

---

## Architecture Contract

### What This Project IS
- OpenRouter API client for LLM tool-calling evaluation
- Dagster-based DAG orchestration for tournament pipelines
- CoreWars MARS simulator integration (external binaries)
- Playwright for browser automation testing

### What This Project IS NOT
- No local GPU inference (no 8765/8888 ports)
- No MCP server integration
- No TalentOS/talent-os infrastructure
- No room coordination or agent spawning

### Execution Model
```
User Request → Claude Code → Python scripts → OpenRouter API → Results
                   ↓
              Dagster DAGs (optional, for tournaments)
```

---

## Resource Manifest

### Directories (Read/Write Boundaries)
| Path | Purpose | Access |
|------|---------|--------|
| `src/modelforecast/` | Core package | Read/Write |
| `tests/` | Test suite | Read/Write |
| `scripts/` | Utility scripts | Read/Write |
| `games/` | Game definitions (CoreWars warriors) | Read/Write |
| `results/` | Tournament outputs | Write |
| `charts/` | Generated visualizations | Write |
| `var/` | Runtime data | Write |
| `archive/` | Historical data | Read |
| `docs/` | Documentation | Read/Write |

### External Dependencies
| Dependency | Purpose | Location |
|------------|---------|----------|
| PMARS | CoreWars simulator | `/home/jw/dev/game1/external/corewars-sandbox/pmars` |
| GPU MARS | Fast GPU simulator | `/home/jw/dev/game1/external/corewars-sandbox/gpu_mars/build/gpu_mars_interleaved` |

### Environment
- **Virtual env**: `.venv/` (project-local, NOT game1/.venv)
- **Python**: `uv run python` (always, never `python` or `python3`)
- **Config**: `.mcp.json` (MCP disabled for this project)

---

## Tool Selection Decision Tree

```
Task Type                          → Tool Choice
─────────────────────────────────────────────────
Run tournament                     → uv run python -m modelforecast
Run single probe                   → uv run python src/modelforecast/probes/<probe>.py
Generate charts                    → uv run python scripts/generate_waffle.py
Run tests                          → uv run pytest tests/
Dagster pipeline                   → uv run dagster dev (port 3000)
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

**Free models** (prioritize these for development):
- `google/gemini-2.5-flash-lite-preview-09-2025:free`
- `qwen/qwen3-32b:free`
- `meta-llama/llama-4-maverick:free`

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
console.print("[green]Success:[/] Tournament completed")
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
- game1/CLAUDE.md (different project)
- talent-os/ (not used here)
- ZMCPTools/ (not used here)
- Port 8765/8888 services (not running)

---

## Baseline Capabilities (Regression Prevention)

These features must always work:

1. **Tournament execution**: `uv run python -m modelforecast` runs without errors
2. **Test suite passes**: `uv run pytest tests/` all green
3. **Chart generation**: `scripts/generate_waffle.py` produces valid PNG
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
- Multi-model tournament strategy design
- CoreWars warrior optimization algorithms
- DAG dependency conflict resolution
- Performance regression root cause analysis

### DO NOT Use Sequential Thinking For:
- Simple file edits
- Running existing scripts
- Reading documentation
- Status checks

### Cost Awareness
- Development: Use free models (Gemini Flash Lite, Qwen3-32b)
- Production benchmarks: Use paid models only when necessary
- Log all API calls to `var/api_calls.jsonl` for cost tracking

---

## Common Tasks

### Run a Tournament
```bash
cd /home/jw/dev/modelforecast
uv run python -m modelforecast --models "gpt-4o-mini,gemini-flash" --rounds 10
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

- **Commits**: Use "jw" not "Jeff" in author
- **Branch naming**: `feat/description`, `fix/description`, `docs/description`
- **PR target**: `main` branch

---

## Anti-Patterns (From TeacherV3 Analysis)

### DO NOT:
1. **Import game1 infrastructure** - This project is standalone
2. **Reference 8765/8888 ports** - No GPU services here
3. **Use MCP tools** - Direct Python + OpenRouter only
4. **Create talent-os style rooms** - No agent coordination
5. **Skip preflight checks** - Validate before running tournaments

### DO:
1. **Use uv run python** - Always, for reproducibility
2. **Check API key** - Fail fast if missing
3. **Log costs** - Track API spend
4. **Run tests** - Before and after changes

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `uv run python -m modelforecast` | Run tournament |
| `uv run pytest tests/` | Run tests |
| `uv run ruff check src/` | Lint code |
| `uv run dagster dev` | Start Dagster UI |
| `cat README.md` | Project overview |
| `cat CONTRIBUTING.md` | Contribution guide |

---

*Generated from TeacherV3 pattern analysis (12,650 patterns, 3,297 sessions)*
*Confidence: 0.89 | Last updated: 2025-12-10*
