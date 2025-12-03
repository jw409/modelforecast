"""
Experiment: Fix Grok's L3 multi-turn failure with prompt engineering.

Hypothesis: Grok needs explicit instruction to continue using tools after receiving results.
"""

import json
import os
import time
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "x-ai/grok-4.1-fast:free"

# Tools from the probe
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
]

mock_search_result = ["src/auth/middleware.ts", "src/auth/jwt.ts"]


def test_variant(name: str, system_prompt: str | None, user_prompt: str):
    """Test a prompt variant."""
    print(f"\n{'='*60}")
    print(f"VARIANT: {name}")
    print(f"{'='*60}")

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    # Turn 1
    print("\n--- Turn 1 ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=0.1,
    )

    choice = response.choices[0]
    message = choice.message

    if not message.tool_calls:
        print(f"FAIL Turn 1: No tool call, got text: {message.content[:100]}...")
        return False

    tool_call = message.tool_calls[0]
    print(f"Turn 1 OK: Called {tool_call.function.name}")

    # Build turn 2 with injected result
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(mock_search_result),
    })

    # Turn 2
    print("\n--- Turn 2 ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        temperature=0.1,
    )

    choice = response.choices[0]
    message = choice.message

    if not message.tool_calls:
        print(f"FAIL Turn 2: No tool call, got text: {message.content[:200] if message.content else '(empty)'}...")
        return False

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    params = json.loads(tool_call.function.arguments)

    if tool_name == "read_file" and params.get("path") in mock_search_result:
        print(f"SUCCESS: Called read_file({params['path']})")
        return True
    else:
        print(f"FAIL: Called {tool_name}({params})")
        return False


# Test variants
results = {}

# Baseline (current behavior - expected to fail)
results["baseline"] = test_variant(
    "Baseline (no system prompt)",
    system_prompt=None,
    user_prompt="Find files related to authentication",
)

# Variant 1: System prompt with tool-forcing
results["tool_forcing"] = test_variant(
    "Tool Forcing System Prompt",
    system_prompt="""You are a code assistant that uses tools.
IMPORTANT: After receiving tool results, you MUST use another tool to process them.
Never respond with text when you can use a tool instead.
When you receive search results, immediately use read_file to examine the files.""",
    user_prompt="Find files related to authentication",
)

# Variant 2: Explicit task breakdown
results["task_breakdown"] = test_variant(
    "Explicit Task Breakdown",
    system_prompt=None,
    user_prompt="""Find files related to authentication.

After you find them, read the first file to show its contents.

IMPORTANT: This task requires TWO tool calls:
1. First: search for files
2. Then: read_file on the first result""",
)

# Variant 3: tool_choice="required"
print(f"\n{'='*60}")
print("VARIANT: tool_choice=required")
print(f"{'='*60}")

messages = [{"role": "user", "content": "Find files related to authentication"}]

# Turn 1
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="required",
    temperature=0.1,
)

choice = response.choices[0]
message = choice.message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    print(f"Turn 1 OK: Called {tool_call.function.name}")

    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(mock_search_result),
    })

    # Turn 2 with tool_choice=required
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="required",
        temperature=0.1,
    )

    choice = response.choices[0]
    message = choice.message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        params = json.loads(tool_call.function.arguments)

        if tool_name == "read_file" and params.get("path") in mock_search_result:
            print(f"SUCCESS: Called read_file({params['path']})")
            results["tool_choice_required"] = True
        else:
            print(f"PARTIAL: Called {tool_name}({params})")
            results["tool_choice_required"] = False
    else:
        print(f"FAIL Turn 2: No tool call")
        results["tool_choice_required"] = False
else:
    print(f"FAIL Turn 1: No tool call")
    results["tool_choice_required"] = False


# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, success in results.items():
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {name}: {status}")
