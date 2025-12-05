# Contributing to ModelForecast

We welcome community contributions! This guide explains how to submit results and contribute to the project.

## Submitting Results

### Quick Version

1. Fork this repo
2. Run `uv run python -m modelforecast`
3. Commit your `results/` folder
4. Open a PR

### Detailed Steps

#### 1. Set Up Your Environment

```bash
git clone https://github.com/YOUR_USERNAME/modelforecast
cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

#### 2. Configure API Key

```bash
export OPENROUTER_API_KEY=your_key_here
```

#### 3. Run Probes

```bash
# Run all probes (recommended)
uv run python -m modelforecast

# Run specific model only
uv run python -m modelforecast --model "x-ai/grok-4.1-fast:free"

# Run specific level only
uv run python -m modelforecast --level 0
```

#### 4. Verify Your Results

```bash
uv run python -m modelforecast.verify --local
```

#### 5. Commit and Push

```bash
git add results/
git commit -m "Add results for [model names]"
git push origin main
```

#### 6. Open a Pull Request

Use the PR template. Automated CI will verify your results.

## What Happens After You Submit

1. **Automated Verification**: CI re-runs your probes to verify results
2. **Tolerance Check**: Results must match within 15% of reproduced values
3. **Agreement Check**: At least 70% of probes must agree
4. **If Passed**: Bot comments "Verification Passed", maintainer merges
5. **If Failed**: Bot explains discrepancy, you can investigate and update

## Verification Failures

Common causes of verification failures:

- **Rate limiting**: OpenRouter may rate limit differently
- **Model updates**: Free models can change without notice
- **Environment**: Different Python/SDK versions

If verification fails but you believe your results are correct:
1. Check workflow logs for details
2. Add explanation to PR description
3. A maintainer will review manually

## Code Contributions

### Bug Fixes

1. Open an issue describing the bug
2. Fork and create a branch: `git checkout -b fix/your-fix-name`
3. Make your changes
4. Run tests: `uv run pytest`
5. Open a PR referencing the issue

### Methodology Improvements

Methodology changes require discussion first:

1. Open an issue with the `methodology` label
2. Describe the proposed change and rationale
3. Wait for maintainer feedback
4. If approved, implement and PR

### New Probe Dimensions

Adding new probe dimensions is a significant change:

1. Open an issue with detailed specification
2. Include: prompt, tools, pass criteria, fail modes
3. Explain what capability it tests that existing probes don't
4. Wait for approval before implementing

**Example**: See [2025-12-04-capability-dimensions-migration.md](docs/plans/2025-12-04-capability-dimensions-migration.md) for a well-structured plan that:
- Explains the "why" (L0-L4 implied difficulty progression, but L4 < L3 empirically)
- Defines the taxonomy clearly (T = Tool Calling, R = Restraint, A = Agency)
- Maps old → new naming with rationale
- Includes falsification tests (R0 must abstain AND be helpful, preventing "dumb silence" false positives)
- Specifies backwards compatibility requirements
- Outlines phased execution with verification steps

## Code Style

- Python 3.11+
- Use `ruff` for linting: `uv run ruff check .`
- Type hints required for public functions
- Docstrings for modules and public functions

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_probes.py

# Run with coverage
uv run pytest --cov=modelforecast
```

## Questions?

- **Methodology questions**: Open an issue with `methodology` label
- **Bug reports**: Open an issue with `bug` label
- **Model requests**: Open an issue with `model-request` label

We aim to respond to issues within 48 hours.

---

## Contributor Badges

When your PR is merged, you earn a badge that's displayed in the README.

| Badge | Earned By | Example |
|-------|-----------|---------|
| 🔬 Model Contributor | Added results for 1+ new model | "Add results for mistral-large" |
| ⚙️ Methodology Contributor | Improved test harness, scoring, or CI | "Fix Wilson interval edge case" |
| 🐛 Bug Hunter | Found and fixed reproducibility issue | "Fix race condition in probe runner" |

**Process**: PR merged → maintainer adds you to README Contributors section.

No point systems. No tiers. Just recognition for real contributions.
