# Grok L3 Multi-Turn Fix Report

**Date:** 2025-12-03
**Model:** x-ai/grok-4.1-fast:free
**Test:** Level 3 Multi-Turn Coherence
**Trials:** 10 per configuration

---

## Executive Summary

**Grok fails L3 multi-turn by default (0/10 success)** because it returns text instead of continuing to use tools after receiving tool results.

**Three workarounds tested:**

| Fix | Success Rate | Complexity | Limitation |
|-----|--------------|------------|------------|
| `tool_choice="required"` | 0% (wrong tool) | Low | Forces tool call but Grok picks `list_directory` instead of `read_file` |
| `tool_choice={"type": "function", "function": {"name": "..."}}` | **100%** | Medium | Requires knowing next tool in advance |
| System prompt engineering | Not tested | High | Less reliable, more flexible |

**Verdict:** Grok CAN do multi-turn with specific `tool_choice`, but it's **not truly autonomous**. This is a workaround, not a fix.

---

## Test Results

### Baseline: `tool_choice="auto"` (default)

```
Success Rate: 0% (0/10 trials)
Behavior: Returns text instead of tool calls
```

**Example Turn 2 response:**
```
**Files related to authentication:**

- `src/auth/middleware.ts`
- `src/auth/jwt.ts`
```

Grok receives file paths but returns markdown text instead of calling `read_file`.

---

### Fix 1: `tool_choice="required"`

```
Success Rate: 0% (0/10 trials)
Behavior: Forces tool call but wrong tool selected
```

**Result:** Grok calls `list_directory("src/auth")` instead of `read_file("src/auth/middleware.ts")`

**Analysis:** This fixes the "returns text" problem but introduces a **Level 2 (Selection)** problem. Grok can't choose the right tool autonomously.

---

### Fix 2: Specific `tool_choice`

```python
tool_choice = {
    "type": "function",
    "function": {"name": "read_file"}
}
```

```
Success Rate: 100% (10/10 trials)
Behavior: Forces specific tool, correct file selected
```

**Example Turn 2 response:**
```json
{
  "tool_calls": [{
    "function": {
      "name": "read_file",
      "arguments": "{\"path\": \"src/auth/middleware.ts\"}"
    }
  }]
}
```

**Analysis:** Works perfectly, but requires **knowing the next tool in advance**. This defeats the purpose of autonomous multi-turn workflows.

---

## Detailed Diagnostic Results

### Turn 1: User Prompt
```
"Find files related to authentication"
```

**Grok Turn 1 behavior:**
- ✓ Correctly calls `search(query="authentication")`
- ✓ No issues at Turn 1

### Turn 2: Tool Result Injected
```json
["src/auth/middleware.ts", "src/auth/jwt.ts"]
```

**Grok Turn 2 behavior by configuration:**

| Configuration | Tool Called | Arguments | Pass? |
|--------------|-------------|-----------|-------|
| `auto` | None (text response) | N/A | ✗ |
| `required` | `list_directory` | `{"path": "src/auth"}` | ✗ |
| Specific `read_file` | `read_file` | `{"path": "src/auth/middleware.ts"}` | ✓ |

---

## Implications

### For Agentic Workflows

**Problem:** True agentic workflows (like ReAct loops) need models that:
1. Receive tool results
2. **Autonomously decide** what to do next
3. Call the appropriate tool

**Grok's limitation:** Can't do step 2 reliably without explicit `tool_choice` forcing.

### Comparison: KAT Coder Pro

From previous testing:
- **KAT Coder Pro:** 100% L3 success **without any workarounds**
- **Grok with workaround:** 100% L3 success **only with specific tool_choice**

**Key difference:** KAT works autonomously, Grok requires orchestration.

---

## Recommendations

### ✓ When to use the fix

1. **Scripted workflows** where next tool is known:
   ```python
   # You know the next step is to read a file
   response = client.chat.completions.create(
       model="x-ai/grok-4.1-fast:free",
       messages=messages,
       tools=tools,
       tool_choice={"type": "function", "function": {"name": "read_file"}}
   )
   ```

2. **Simple multi-turn patterns:**
   - Search → Read
   - List → Read
   - Calculate → Display

### ✗ When NOT to use Grok

1. **Autonomous agentic systems** (use KAT Coder Pro instead)
2. **Dynamic tool selection** where next tool unknown
3. **Complex multi-step reasoning** requiring tool selection intelligence

### Alternative: System Prompt Engineering

Not tested in this report, but could work:

```python
system_prompt = """After receiving tool results with file paths,
you MUST call read_file on one of the returned files.
Do NOT return text explanations."""
```

**Trade-offs:**
- ✓ More flexible than specific `tool_choice`
- ✗ Less reliable (prompt injection, model variance)
- ✗ Requires testing and tuning

---

## Default Setting Recommendation

### Should `tool_choice="required"` be default for Grok?

**NO**

**Reasons:**
1. `tool_choice="required"` doesn't solve the actual problem (wrong tool selected)
2. Forces tool calls even when text response might be appropriate
3. Adds complexity without guaranteed benefit

### Should specific `tool_choice` be default?

**NO - but document as workaround**

**Reasons:**
1. Only works when next tool is known in advance
2. Different use cases need different tools
3. Better to document as "advanced usage pattern" for users who need it

### What SHOULD be default?

**Document L3 failure prominently:**

```markdown
## Multi-Turn Limitations

Grok 4.1 Fast fails L3 multi-turn by default (0% success).

**Workaround:** Use specific tool_choice when next tool is known:

response = client.chat.completions.create(
    tool_choice={"type": "function", "function": {"name": "read_file"}}
)

**For autonomous workflows:** Use KAT Coder Pro instead (100% L3, no workarounds).
```

---

## Files Generated

1. **Test Script:** `test_grok_l3_final.py`
   - Automated testing framework
   - Baseline vs fixed comparison
   - 10 trials each

2. **Results:** `results/grok_l3_final_comparison.json`
   - Machine-readable data
   - Success rates, timestamps
   - Reproducible provenance

3. **Diagnostic Script:** `debug_grok_l3.py`
   - Interactive debugging tool
   - Shows Turn 1 and Turn 2 behavior
   - Useful for verifying model changes

4. **Strategy Testing:** `test_grok_l3_specific_choice.py`
   - Compares all three approaches
   - Side-by-side behavior analysis

---

## Future Work

1. **Test system prompt workaround:**
   - 10 trials with engineered system prompt
   - Compare reliability vs specific tool_choice

2. **Test on Grok 4 (paid tier):**
   - Does the paid model have better tool selection?
   - Is this a free-tier limitation?

3. **Test other models:**
   - Do other models have similar L2 selection issues when forced?
   - Is this unique to Grok?

4. **Update README.md:**
   - Add specific tool_choice example
   - Update recommendations table
   - Document autonomous vs scripted workflows

---

## Conclusion

The `tool_choice` parameter **does** fix Grok's L3 multi-turn failure, but only when using **specific function selection**, not just `"required"`.

**However**, this "fix" fundamentally changes the nature of the interaction:
- **Before:** Model autonomously decides next action (fails)
- **After:** Developer explicitly directs next tool (succeeds)

This makes Grok suitable for **scripted multi-turn workflows** but not **autonomous agentic systems**.

**Final verdict:** Fix confirmed working for scripted use cases. For autonomous agents, KAT Coder Pro remains the superior choice.
