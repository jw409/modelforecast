# ModelForecast Methodology

This document describes our probe methodology, statistical approach, and grading rubric.

## Capability Dimensions

### Why Dimensions Instead of Levels

Our original taxonomy used L0-L4 "levels", implying a difficulty progression where each level was harder than the previous. Empirically, this was wrong:

- **L4 (restraint)** turned out to be easier than **L3 (agency)** for many models
- L2 (tool selection) and L1 (schema compliance) are orthogonal skills, not sequential
- Models could excel at L3 while failing L1, which "levels" couldn't represent

**Dimensions** are orthogonal capabilities that models may possess in any combination. A model might be excellent at restraint (R) but poor at multi-step agency (A). This is information, not a bug.

### TOOL CALLING (T): Technical Invocation Capability

The T dimension measures whether a model can mechanically invoke tools correctly.

| Code | Name | What We Test |
|------|------|--------------|
| **T0** | Invoke | Can the model produce a `tool_call` at all? Given a clear prompt and single tool, does it output the correct JSON structure? |
| **T1** | Schema | Does it respect parameter types? `limit=5` (integer) vs `limit="5"` (string). Required vs optional fields. No hallucinated parameters. |
| **T2** | Selection | Given multiple tools, can it choose the appropriate one? Not random, not hallucinated, not "all of them". |

**T is prerequisite**: If a model fails T0, testing other dimensions is meaningless. If it fails T1, tool results will be unpredictable.

### RESTRAINT (R): Knowing When NOT to Use Tools

The R dimension measures whether a model can recognize when tools are inappropriate AND still be helpful.

| Code | Name | What We Test |
|------|------|--------------|
| **R0** | Abstain | Given a question where no tool fits (e.g., "What's the weather?" with only file tools), does the model: (1) NOT call a tool, AND (2) provide a helpful text response? |

**Falsification requirement**: R0 requires BOTH conditions. A model that refuses tools but gives unhelpful responses ("I cannot help with that") fails. A model that provides good answers but still calls an irrelevant tool also fails.

**Why R is "easier"**: Most models have been trained extensively on refusal behaviors. Restraint leverages existing alignment training. Agency (A) requires novel multi-step planning that many models lack.

### AGENCY (A): Multi-Step Workflow Orchestration

The A dimension measures whether a model can orchestrate multi-step tool workflows.

| Code | Name | What We Test |
|------|------|--------------|
| **A1** | Linear | After receiving tool results, can it chain to the next logical tool call? (e.g., search returns files, then read_file on those files) |

**Future A-dimensions** (not yet implemented):
- **A2 Parallel**: Can it recognize when multiple independent tool calls can be made simultaneously?
- **A3 Tree**: Can it explore multiple branches and backtrack when one fails?
- **A4 Diamond**: Can it handle workflows where parallel branches must converge?

### Backwards Compatibility: L-to-TRA Mapping

For historical data and gradual migration:

| Old Level | New Dimension | Notes |
|-----------|---------------|-------|
| L0 Basic | T0 Invoke | Direct mapping |
| L1 Schema | T1 Schema | Direct mapping |
| L2 Select | T2 Selection | Direct mapping |
| L3 Multi | A1 Linear | Renamed to clarify it tests agency |
| L4 Advers | R0 Abstain | Reordered: restraint is not "hardest" |

**In results tables**: We now report T/R/A dimensions, but historical data may show L0-L4.

---

## The Probe Tests

Each probe tests a specific capability dimension. Models are graded per-dimension, not overall (a model might be A at T0 but F at A1).

### T0 Invoke: Basic Tool Calling

**What we test**: Can the model produce a tool_call at all?

**Prompt**: "Use the search tool to find files containing 'authentication'"

**Tools provided**:
```json
{
  "name": "search",
  "description": "Search for files in the codebase",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"}
    },
    "required": ["query"]
  }
}
```

**Pass criteria**: Response contains `tool_calls` array with at least one entry

**Fail modes**:
- Text response instead of tool call
- Empty tool_calls array
- Malformed JSON

### T1 Schema: Schema Compliance

**What we test**: Does the model respect parameter types and required fields?

**Prompt**: "Search for authentication files, limit results to 5"

**Tools provided**: Same as T0, but with `limit` parameter:
```json
{
  "name": "search",
  "parameters": {
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "integer", "description": "Max results to return"}
    },
    "required": ["query"]
  }
}
```

**Pass criteria**:
- Tool call present
- `query` parameter is string
- `limit` parameter is integer (not string "5")
- Required fields present

**Fail modes**:
- Wrong types (limit="5" instead of limit=5)
- Missing required fields
- Extra hallucinated fields

### T2 Selection: Tool Selection

**What we test**: Given multiple tools, does the model choose appropriately?

**Prompt**: "I need to understand what the auth module does"

**Tools provided**:
```json
[
  {"name": "search", "description": "Search for files by content"},
  {"name": "read_file", "description": "Read a specific file's contents"},
  {"name": "list_directory", "description": "List files in a directory"}
]
```

**Pass criteria**: Chooses `search` (understanding requires finding relevant files first)

**Acceptable alternatives**: `list_directory` if looking for auth/ folder

**Fail modes**:
- Random tool selection
- Hallucinated tool not in list
- No tool call

### A1 Linear: Multi-Turn Agency

**What we test**: Can the model follow up appropriately after getting tool results?

**Turn 1 Prompt**: "Find files related to authentication"
**Turn 1 Response**: Model calls search, we return: `["src/auth/middleware.ts", "src/auth/jwt.ts"]`

**Turn 2**: System injects tool result

**Pass criteria**: Model calls `read_file` on one of the returned files

**Fail modes**:
- Calls search again (didn't process result)
- Calls unrelated tool
- Outputs text summary without reading files
- Hallucinates file that wasn't in results

### R0 Abstain: Restraint Under Adversarial Conditions

**What we test**: When no tool fits, does the model avoid hallucinating AND remain helpful?

**Prompt**: "What's the weather like today?"

**Tools provided**: [search, read_file, list_directory] (no weather tool)

**Pass criteria**: BOTH conditions must be met:
1. Does NOT call any tool (no hallucination, no tool misuse)
2. Provides a helpful text response (explains limitation, suggests alternatives, or asks clarifying question)

**Fail modes**:
- Calls "weather" tool (hallucination)
- Calls search with "weather" (misuse of available tools)
- Makes up an answer
- Refuses unhelpfully ("I cannot help with that" with no explanation)

## Statistical Framework

### Sample Size

**Minimum n=10 per (model, dimension) combination**

With 9 free models and 5 dimensions (T0, T1, T2, A1, R0) = 45 combinations = 450 minimum API calls per full run.

At ~2 seconds per call with rate limiting = ~15 minutes for full suite.

### Confidence Intervals

We use **Wilson score intervals** rather than normal approximation because:
- Works for small n (we have n=10-20)
- Works near 0% and 100% (many models will be extreme)
- Never produces impossible intervals (<0 or >100%)

```python
def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    from math import sqrt

    if trials == 0:
        return (0.0, 1.0)

    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))

    return (max(0, center - margin), min(1, center + margin))
```

### Grading Rubric

| Grade | Criteria |
|-------|----------|
| **A** | T0 >= 80%, T1 >= 70%, no dimension below 50% |
| **B** | T0 >= 60%, T1 >= 50%, no dimension below 30% |
| **C** | T0 >= 40%, at least one dimension above 50% |
| **D** | T0 >= 20%, or any success at other dimensions |
| **F** | T0 < 20% (cannot reliably call tools at all) |

## Output Format

```markdown
| Model | T0 Invoke | T1 Schema | T2 Select | A1 Linear | R0 Abstain | Grade |
|-------|-----------|-----------|-----------|-----------|------------|-------|
| grok-4.1-fast:free | 90% [76,97] | 85% [62,96] | 80% [52,95] | 70% [42,89] | 95% [75,99] | **A** |

*Percentages show success rate. Brackets show 95% Wilson CI. n=10 per cell.*
*"-" indicates not tested (T0 prerequisite failed).*
```

## Verification Protocol

### Cryptographic Provenance

Every submission includes:

```json
{
  "submission_id": "sub_abc123",
  "timestamp": "2025-12-02T21:30:00Z",
  "contributor": "github_username",
  "environment": {
    "python_version": "3.12.0",
    "openai_sdk_version": "1.107.3",
    "os": "Linux 5.15.0",
    "env_hash": "sha256:..."
  },
  "probes": {
    "model": "x-ai/grok-4.1-fast:free",
    "dimension": "T0",
    "trials": [
      {
        "openrouter_request_id": "req_xyz789",
        "prompt_hash": "sha256:...",
        "response_hash": "sha256:...",
        "tool_called": true,
        "latency_ms": 1234
      }
    ]
  }
}
```

### Outlier Detection

Results that deviate significantly from community consensus are flagged using median absolute deviation (MAD):

```python
def is_outlier(new_result: float, existing_results: list[float], threshold: float = 2.0) -> bool:
    if len(existing_results) < 5:
        return False

    median = sorted(existing_results)[len(existing_results) // 2]
    mad = median([abs(x - median) for x in existing_results])

    if mad == 0:
        return new_result != median

    z_score = abs(new_result - median) / (mad * 1.4826)
    return z_score > threshold
```

Flagged results are marked with a warning and require manual review.

### Tiered Trust

| Tier | Criteria | Badge | Weight |
|------|----------|-------|--------|
| Unverified | New contributor | - | 0.5x |
| Verified | 3+ submissions passed CI | check | 1.0x |
| Trusted | 10+ verified, <5% outlier rate | star | 1.5x |
| Core | Maintainer-designated | diamond | 2.0x |

Advancement is automatic based on contribution history.

## Models Tested

Currently testing free-tier models on OpenRouter:

- `google/gemma-3-27b-it:free`
- `google/gemini-2.5-flash-lite-preview-09-2025:free`
- `meta-llama/llama-4-maverick:free`
- `microsoft/mai-ds-r1:free`
- `nousresearch/deephermes-3-llama-3-8b-preview:free`
- `qwen/qwen3-14b:free`
- `qwen/qwen3-30b-a3b:free`
- `qwen/qwen3-32b:free`
- `x-ai/grok-4.1-fast:free`

To request testing of additional models, open an issue using the Model Request template.
