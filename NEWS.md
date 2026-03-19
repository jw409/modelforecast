# ModelForecast News

---

**March 19, 2026** — "The Great Re-Sweep"

## NVIDIA Nemotron Sweeps All Five Dimensions

**nvidia/nemotron-3-nano-30b-a3b** scores 100% across T0 (invoke), T1 (schema), T2 (selection), A1 (multi-turn), and R0 (restraint). The only free model with a perfect score on the March 2026 sweep.

## Five Grade-A Free Models Found

Out of 16 free models on OpenRouter that claim tool support:

| Model | T0 | T1 | T2 | A1 | R0 | Grade |
|-------|:--:|:--:|:--:|:--:|:--:|:-----:|
| nemotron-3-nano-30b-a3b | 100% | 100% | 100% | 100% | 100% | **A** |
| nemotron-3-super-120b-a12b | 100% | 100% | 100% | 100% | 70% | **A** |
| gpt-oss-120b | 80% | 100% | 90% | 90% | 100% | **A** |
| step-3.5-flash | 100% | 100% | 100% | 100% | 80% | **A** |
| glm-4.5-air | 100% | 90% | 100% | 80% | 80% | **A** |

## Half of "Tool-Capable" Models Can't Call Tools

7 of 16 free models that advertise tool support scored 0% on T0 — they respond with text instead of tool calls. Qwen (3 models), Meta, Mistral, and OpenAI's smaller gpt-oss-20b all fail completely.

## Privacy Settings Gotcha

OpenRouter's account privacy settings can silently block free model endpoints. gpt-oss-120b initially scored 0% — turned out the endpoint was blocked by default privacy config. After enabling "free endpoints that may train on inputs," it scored Grade A. **Check your settings before trusting zero scores.**

## 5 Models Still Rate-Limited

llama-3.3-70b, mistral-small-3.1, qwen3-4b, qwen3-coder, and qwen3-next-80b are globally throttled by OpenRouter ("temporarily rate-limited"). Results pending when demand drops.

---

**December 29, 2025** — "The Metacognitive Era"

## DeepSeek V3 Dominates CoreWars

In the latest Round-Robin, DeepSeek V3 secured a dominant 62-38 victory over Claude Sonnet 4.5 in CoreWars battles. The Endless Workshop metacognitive self-play framework produced Obsidian Breaker v4, which reached Rank #3 by beating GPT 5.1.

---

**December 28, 2025** — "Hard Times Recovery"

## Zero-Win Warrior Evolves to Rank #3

The warrior "Hard Times" (zero wins) was transformed via 12 rounds of metacognitive self-play into Obsidian Breaker v4, securing Rank #3 and neutralizing GPT 5.1 in the Massive Tournament.
