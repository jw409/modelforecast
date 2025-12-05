"""Model validation and discovery for OpenRouter models."""

import os
from functools import lru_cache
from typing import Any

import httpx


@lru_cache(maxsize=1)
def get_available_models(api_key: str | None = None) -> dict[str, Any]:
    """Fetch available models from OpenRouter API.

    Returns:
        Dict mapping model ID -> model info
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return {model["id"]: model for model in data.get("data", [])}


def supports_tool_calling(model_id: str, api_key: str | None = None) -> bool:
    """Check if a model supports tool calling on OpenRouter.

    Args:
        model_id: Full model ID (e.g., "google/gemma-3-12b-it:free")
        api_key: Optional API key

    Returns:
        True if model supports tools, False otherwise
    """
    models = get_available_models(api_key)
    model_info = models.get(model_id, {})
    supported_params = model_info.get("supported_parameters", [])
    return "tools" in supported_params


def get_free_models(api_key: str | None = None, tools_only: bool = False) -> list[str]:
    """Get list of currently available free models.

    Args:
        api_key: Optional API key
        tools_only: If True, only return models that support tool calling

    Returns:
        List of model IDs ending with :free
    """
    models = get_available_models(api_key)
    free_models = [
        model_id for model_id in models.keys()
        if model_id.endswith(":free")
    ]

    if tools_only:
        free_models = [
            m for m in free_models
            if supports_tool_calling(m, api_key)
        ]

    return sorted(free_models)


def get_tool_support_matrix(api_key: str | None = None) -> dict[str, bool]:
    """Get tool support status for all free models.

    Returns:
        Dict mapping model_id -> supports_tools (bool)
    """
    models = get_available_models(api_key)
    return {
        model_id: "tools" in model_info.get("supported_parameters", [])
        for model_id, model_info in models.items()
        if model_id.endswith(":free")
    }


def validate_model(model_id: str, api_key: str | None = None) -> tuple[bool, str]:
    """Validate that a model ID exists on OpenRouter.

    Args:
        model_id: Full model ID (e.g., "google/gemini-2.0-flash-exp:free")
        api_key: Optional API key

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        models = get_available_models(api_key)

        if model_id in models:
            return True, f"Model {model_id} is available"

        # Check for similar models (helpful suggestions)
        base_name = model_id.split(":")[0]
        similar = [m for m in models.keys() if base_name.lower() in m.lower()]

        if similar:
            return False, f"Model {model_id} not found. Similar models: {similar[:5]}"

        return False, f"Model {model_id} not found on OpenRouter"

    except Exception as e:
        return False, f"Error validating model: {e}"


def validate_models(model_ids: list[str], api_key: str | None = None) -> dict[str, tuple[bool, str]]:
    """Validate multiple model IDs.

    Args:
        model_ids: List of model IDs to validate
        api_key: Optional API key

    Returns:
        Dict mapping model_id -> (is_valid, message)
    """
    results = {}
    for model_id in model_ids:
        results[model_id] = validate_model(model_id, api_key)
    return results


def filter_valid_models(model_ids: list[str], api_key: str | None = None) -> list[str]:
    """Filter list to only valid models, with warnings for invalid ones.

    Args:
        model_ids: List of model IDs to check
        api_key: Optional API key

    Returns:
        List of valid model IDs
    """
    validation = validate_models(model_ids, api_key)
    valid = []

    for model_id, (is_valid, message) in validation.items():
        if is_valid:
            valid.append(model_id)
        else:
            print(f"WARNING: {message}")

    return valid


def clear_model_cache() -> None:
    """Clear the cached model list (useful for testing)."""
    get_available_models.cache_clear()
