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


def get_free_models(api_key: str | None = None) -> list[str]:
    """Get list of currently available free models.

    Returns:
        List of model IDs ending with :free
    """
    models = get_available_models(api_key)
    return sorted([
        model_id for model_id in models.keys()
        if model_id.endswith(":free")
    ])


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
