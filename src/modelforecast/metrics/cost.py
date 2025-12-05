from typing import Dict, Any
from modelforecast.models import get_available_models

class CostTracker:
    def __init__(self):
        self.pricing_cache = {}

    def get_pricing(self, model_id: str) -> Dict[str, float]:
        if model_id not in self.pricing_cache:
            try:
                models = get_available_models()
                if model_id in models:
                    pricing = models[model_id].get("pricing", {})
                    self.pricing_cache[model_id] = {
                        "prompt": float(pricing.get("prompt", 0)),
                        "completion": float(pricing.get("completion", 0))
                    }
                else:
                    # Default to 0 if not found (e.g. local or unknown)
                    self.pricing_cache[model_id] = {"prompt": 0.0, "completion": 0.0}
            except Exception:
                 self.pricing_cache[model_id] = {"prompt": 0.0, "completion": 0.0}
        return self.pricing_cache[model_id]

    def calculate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self.get_pricing(model_id)
        # OpenRouter pricing is per token (usually string decimal)
        cost = (prompt_tokens * pricing["prompt"]) + (completion_tokens * pricing["completion"])
        return cost
