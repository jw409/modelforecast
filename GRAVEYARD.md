# Model Graveyard

OpenRouter model endpoints that were previously tested or tracked but are no longer available.
Maintained to preserve historical result context and prevent re-testing removed models.

Run `uv run python scripts/update_graveyard.py --known <model_id> [<model_id>...]` to check
a list of previously-known models and append any that have disappeared.

## Format

| Model ID | Last Known Available | Date Removed | Notes |
|----------|----------------------|--------------|-------|
| `vendor/model-name` | YYYY-MM-DD | YYYY-MM-DD | Reason if known |

## Graveyard

| Model ID | Last Known Available | Date Removed | Notes |
|----------|----------------------|--------------|-------|
| `google/gemma-3-12b-it:free` | 2026-03-01 | 2026-03-18 | Removed from OpenRouter free tier |
| `google/gemma-3-27b-it:free` | 2026-03-01 | 2026-03-18 | Removed from OpenRouter free tier |
| `deepseek/deepseek-r1:free` | 2026-03-01 | 2026-03-18 | Removed from OpenRouter free tier |
| `meta-llama/llama-3.2-3b-instruct:free` | 2026-03-01 | 2026-03-18 | Removed from OpenRouter free tier |
