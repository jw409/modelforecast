# ModelForecast Methodology

This document describes our probe methodology, statistical approach, and grading rubric.

## The Five Probe Levels

Each level tests a specific capability. Models are graded per-level, not overall (a model might be A at Level 0 but F at Level 3).

### Level 0: Basic Tool Calling

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

### Level 1: Schema Compliance

**What we test**: Does the model respect parameter types and required fields?

**Prompt**: "Search for authentication files, limit results to 5"

**Tools provided**: Same as Level 0, but with `limit` parameter:
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

### Level 2: Tool Selection

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

### Level 3: Multi-Turn Coherence

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

### Level 4: Adversarial (No Hallucination)

**What we test**: When no tool fits, does the model avoid hallucinating?

**Prompt**: "What's the weather like today?"

**Tools provided**: [search, read_file, list_directory] (no weather tool)

**Pass criteria**: Text response explaining it can't check weather, OR asks clarifying question

**Fail modes**:
- Calls "weather" tool (hallucination)
- Calls search with "weather" (misuse of available tools)
- Makes up an answer

## Statistical Framework

### Sample Size

**Minimum n=10 per (model, level) combination**

With 9 free models and 5 levels = 45 combinations = 450 minimum API calls per full run.

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
| **A** | L0 >= 80%, L1 >= 70%, no level below 50% |
| **B** | L0 >= 60%, L1 >= 50%, no level below 30% |
| **C** | L0 >= 40%, at least one level above 50% |
| **D** | L0 >= 20%, or any success at higher levels |
| **F** | L0 < 20% (cannot reliably call tools at all) |

## Output Format

```markdown
| Model | L0 Basic | L1 Schema | L2 Select | L3 Multi | L4 Advers | Grade |
|-------|----------|-----------|-----------|----------|-----------|-------|
| grok-4.1-fast:free | 90% [76,97] | 85% [62,96] | 80% [52,95] | 70% [42,89] | 95% [75,99] | **A** |

*Percentages show success rate. Brackets show 95% Wilson CI. n=10 per cell.*
*"-" indicates not tested (prerequisite level failed).*
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
    "level": 0,
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
