"""OpenRouter client with full cost and usage tracking.

Implements the witness pattern from schemaless: every call tracked,
every token counted, every dollar visible.

Usage:
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    # Chat completions (OpenAI-compatible)
    response = client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    # Embeddings
    response = client.embeddings.create(
        model="qwen/qwen3-embedding-4b",
        input=["Hello world", "Test text"],
    )

    # Cost report
    client.print_report()
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from modelforecast.models import get_available_models

# Fallback pricing per 1M tokens for embedding endpoints that may be absent
# from the chat-oriented model catalog.
PRICING = {
    # Embeddings
    "qwen/qwen3-embedding-4b": {"input": 0.02, "output": 0.0},
    "qwen/qwen3-embedding-8b": {"input": 0.05, "output": 0.0},
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "openai/text-embedding-3-large": {"input": 0.13, "output": 0.0},
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


@dataclass
class APICall:
    """Record of a single API call."""

    timestamp: str
    endpoint: str  # "embeddings" or "chat/completions"
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float  # USD
    output_cost: float  # USD
    total_cost: float  # USD
    latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class WitnessReport:
    """Full observability report for API usage."""

    calls: list[APICall] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def successful_calls(self) -> int:
        return sum(1 for c in self.calls if c.success)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(c.total_cost for c in self.calls)

    @property
    def total_latency_ms(self) -> int:
        return sum(c.latency_ms for c in self.calls)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Group stats by model."""
        models: dict[str, dict[str, Any]] = {}
        for call in self.calls:
            if call.model not in models:
                models[call.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            models[call.model]["calls"] += 1
            models[call.model]["input_tokens"] += call.input_tokens
            models[call.model]["output_tokens"] += call.output_tokens
            models[call.model]["cost"] += call.total_cost
        return models

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# OpenRouter Usage Report",
            "",
            "## Summary",
            f"- **Total Calls**: {self.total_calls} ({self.successful_calls} successful)",
            f"- **Input Tokens**: {self.total_input_tokens:,}",
            f"- **Output Tokens**: {self.total_output_tokens:,}",
            f"- **Total Cost**: ${self.total_cost:.6f}",
            f"- **Total Latency**: {self.total_latency_ms:,}ms",
            "",
            "## By Model",
            "",
            "| Model | Calls | Input | Output | Cost |",
            "|-------|-------|-------|--------|------|",
        ]

        for model, stats in self.by_model().items():
            short_model = model.split("/")[-1] if "/" in model else model
            lines.append(
                f"| {short_model} | {stats['calls']} | "
                f"{stats['input_tokens']:,} | {stats['output_tokens']:,} | "
                f"${stats['cost']:.6f} |"
            )

        if self.calls:
            lines.extend([
                "",
                "## Recent Calls",
                "",
                "| Time | Model | Tokens | Cost | Latency |",
                "|------|-------|--------|------|---------|",
            ])

            for call in self.calls[-20:]:  # Last 20 calls
                short_model = call.model.split("/")[-1][:15]
                status = "OK" if call.success else "ERR"
                lines.append(
                    f"| {call.timestamp[11:19]} | {short_model} | "
                    f"{call.input_tokens}+{call.output_tokens} | "
                    f"${call.total_cost:.6f} | {call.latency_ms}ms {status} |"
                )

        return "\n".join(lines)


# Response dataclasses for OpenAI compatibility
@dataclass
class FunctionCall:
    """Function call details."""
    name: str
    arguments: str


@dataclass
class ToolCall:
    """Tool call in a message."""
    id: str
    type: str
    function: FunctionCall


@dataclass
class Message:
    """Chat completion message."""
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None


@dataclass
class Choice:
    """Chat completion choice."""
    index: int
    message: Message
    finish_reason: str


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dict (OpenAI compatibility)."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (c.message.tool_calls or [])
                        ] if c.message.tool_calls else None,
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


@dataclass
class EmbeddingData:
    """Single embedding result."""
    embedding: list[float]
    index: int = 0
    object: str = "embedding"


@dataclass
class EmbeddingResponse:
    """OpenAI-compatible embedding response."""
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dict (OpenAI compatibility)."""
        return {
            "data": [{"embedding": d.embedding, "index": d.index, "object": d.object} for d in self.data],
            "model": self.model,
            "usage": self.usage,
        }


class OpenRouterClient:
    """OpenRouter client with full cost tracking via witness pattern.

    Provides OpenAI-compatible interface for chat completions and embeddings
    while tracking all API calls, tokens, and costs.

    Usage:
        client = OpenRouterClient()

        # Chat (same interface as openai.OpenAI)
        response = client.chat.completions.create(
            model="meta-llama/llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Get cost report
        print(client.report.total_cost)
        client.print_report()
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_API_URL,
        timeout: float = 60.0,
        app_name: str = "modelforecast",
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (default: OPENROUTER_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
            app_name: Application name for OpenRouter tracking
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required (pass api_key or set env var)")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.app_name = app_name

        # Initialize sub-APIs (OpenAI interface pattern)
        self.chat = ChatAPI(self)
        self.embeddings = EmbeddingsAPI(self)

        # Witness tracking
        self.report = WitnessReport()

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Get current per-million-token pricing, with a conservative fallback."""
        try:
            model_info = get_available_models(self.api_key).get(model)
            if model_info:
                pricing = model_info.get("pricing", {})
                return {
                    "input": float(pricing.get("prompt", 0.0)) * 1_000_000,
                    "output": float(pricing.get("completion", 0.0)) * 1_000_000,
                }
        except Exception:
            pass

        if model in PRICING:
            return PRICING[model]
        return {"input": 5.00, "output": 30.00}

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, float, float]:
        """Calculate costs in USD."""
        pricing = self._get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost, output_cost, input_cost + output_cost

    def _make_request(
        self,
        endpoint: str,
        payload: dict,
        model: str,
    ) -> tuple[dict, APICall]:
        """Make API request with full tracking."""
        url = f"{self.base_url}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": f"https://github.com/{self.app_name}",
            "X-Title": self.app_name,
        }

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                latency_ms = int((time.time() - start_time) * 1000)

                # Extract token usage
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                input_cost, output_cost, total_cost = self._calculate_cost(
                    model, input_tokens, output_tokens
                )

                call = APICall(
                    timestamp=timestamp,
                    endpoint=endpoint,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    latency_ms=latency_ms,
                    success=True,
                )

                self.report.calls.append(call)
                return result, call

        except urllib.error.HTTPError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_body = e.read().decode("utf-8", errors="replace")

            call = APICall(
                timestamp=timestamp,
                endpoint=endpoint,
                model=model,
                input_tokens=0,
                output_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                latency_ms=latency_ms,
                success=False,
                error=f"{e.code}: {error_body[:200]}",
            )

            self.report.calls.append(call)
            raise RuntimeError(f"OpenRouter API error: {error_body}") from e

    def get_report(self) -> WitnessReport:
        """Get the current usage report."""
        return self.report

    def print_report(self) -> None:
        """Print markdown report to stdout."""
        print(self.report.to_markdown())

    def reset(self) -> None:
        """Reset usage tracking (new session)."""
        self.report = WitnessReport()

    def health_check(self) -> bool:
        """Check if OpenRouter API is reachable."""
        try:
            request = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(request, timeout=5.0) as response:
                return response.status == 200
        except Exception:
            return False


class ChatAPI:
    """Chat completions API (client.chat.completions interface)."""

    def __init__(self, client: OpenRouterClient):
        self._client = client
        self.completions = ChatCompletionsAPI(client)


class ChatCompletionsAPI:
    """Chat completions endpoint (client.chat.completions.create interface)."""

    def __init__(self, client: OpenRouterClient):
        self._client = client

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,  # Accept extra params for compatibility
    ) -> ChatCompletionResponse:
        """Create a chat completion.

        Args:
            model: Model ID (e.g., "meta-llama/llama-3.2-3b-instruct")
            messages: List of message dicts with role and content
            tools: Optional tool definitions for function calling
            tool_choice: How to select tools ("auto", "required", or specific)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ChatCompletionResponse (OpenAI-compatible)
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if max_tokens:
            payload["max_tokens"] = max_tokens

        result, _call = self._client._make_request("chat/completions", payload, model)

        # Parse response into dataclasses
        choices = []
        for c in result.get("choices", []):
            msg = c.get("message", {})
            tool_calls = None

            if msg.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get("arguments", "{}"),
                        ),
                    )
                    for tc in msg["tool_calls"]
                ]

            choices.append(Choice(
                index=c.get("index", 0),
                message=Message(
                    role=msg.get("role", "assistant"),
                    content=msg.get("content"),
                    tool_calls=tool_calls,
                ),
                finish_reason=c.get("finish_reason", "stop"),
            ))

        usage = result.get("usage", {})

        return ChatCompletionResponse(
            id=result.get("id", ""),
            object=result.get("object", "chat.completion"),
            created=result.get("created", int(time.time())),
            model=result.get("model", model),
            choices=choices,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


class EmbeddingsAPI:
    """Embeddings API (client.embeddings.create interface)."""

    def __init__(self, client: OpenRouterClient):
        self._client = client

    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs,  # Accept extra params for compatibility
    ) -> EmbeddingResponse:
        """Create embeddings.

        Args:
            model: Embedding model ID (e.g., "qwen/qwen3-embedding-4b")
            input: Text or list of texts to embed

        Returns:
            EmbeddingResponse (OpenAI-compatible)
        """
        # Normalize input to list
        texts = [input] if isinstance(input, str) else input

        payload = {
            "input": texts,
            "model": model,
            "encoding_format": "float",
        }

        result, _call = self._client._make_request("embeddings", payload, model)

        # Parse response
        embeddings = [None] * len(texts)
        for item in result.get("data", []):
            idx = item.get("index", 0)
            embeddings[idx] = item.get("embedding", [])

        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=emb, index=i)
                for i, emb in enumerate(embeddings)
                if emb is not None
            ],
            model=model,
            usage=result.get("usage", {"prompt_tokens": 0, "total_tokens": 0}),
        )


def test_openrouter_client() -> WitnessReport:
    """Run a quick test and return the report."""
    print("Testing OpenRouter Client...")
    print("=" * 50)

    client = OpenRouterClient()

    # Test chat completion
    print("\n1. Testing chat completion...")
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.7-flash",
            messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            max_tokens=10,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        print(f"   Response: {content}")
        print(f"   Tokens: {response.usage.prompt_tokens}+{response.usage.completion_tokens}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test tool calling
    print("\n2. Testing tool calling...")
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }]

        response = client.chat.completions.create(
            model="google/gemini-3.6-flash",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
            temperature=0.1,
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            print(f"   Tool called: {tc.function.name}")
            print(f"   Arguments: {tc.function.arguments}")
        else:
            print(f"   No tool called. Content: {choice.message.content}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test embeddings
    print("\n3. Testing embeddings...")
    try:
        response = client.embeddings.create(
            model="qwen/qwen3-embedding-4b",
            input=["Hello world", "Test embedding"],
        )
        print(f"   Got {len(response.data)} embeddings")
        print(f"   Dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"   Error: {e}")

    # Print report
    print("\n" + "=" * 50)
    client.print_report()

    return client.get_report()


if __name__ == "__main__":
    test_openrouter_client()
