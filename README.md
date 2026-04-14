# ModelForecast

[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)

<!-- MODELFORECAST:QUICK-ANSWER:START -->
### Best free model right now: nemotron-3-nano-30b-a3b

![Grade A](https://img.shields.io/badge/Grade-A-brightgreen) — scores A across all tool-calling dimensions. Free on OpenRouter.

> Use [`nvidia/nemotron-3-nano-30b-a3b:free`](https://openrouter.ai/models/nvidia/nemotron-3-nano-30b-a3b) for tool calling in your agent pipeline.

Full results: [results/RESULTS.md](results/RESULTS.md) · [Methodology](METHODOLOGY.md)
<!-- MODELFORECAST:QUICK-ANSWER:END -->

<!-- MODELFORECAST:GRADE-BADGES:START -->
### Top performers

[![nemotron-3-nano-30b-a3b: Grade A](https://img.shields.io/badge/nemotron--3--nano--30b--a3b-Grade_A-brightgreen)](https://openrouter.ai/models/nvidia/nemotron-3-nano-30b-a3b) [![nemotron-3-super-120b-a12b: Grade A](https://img.shields.io/badge/nemotron--3--super--120b--a12b-Grade_A-brightgreen)](https://openrouter.ai/models/nvidia/nemotron-3-super-120b-a12b) [![gpt-oss-120b: Grade A](https://img.shields.io/badge/gpt--oss--120b-Grade_A-brightgreen)](https://openrouter.ai/models/openai/gpt-oss-120b) [![step-3.5-flash: Grade A](https://img.shields.io/badge/step--3.5--flash-Grade_A-brightgreen)](https://openrouter.ai/models/stepfun/step-3.5-flash) [![glm-4.5-air: Grade A](https://img.shields.io/badge/glm--4.5--air-Grade_A-brightgreen)](https://openrouter.ai/models/z-ai/glm-4.5-air)

*Click a badge to view the model on OpenRouter.*
<!-- MODELFORECAST:GRADE-BADGES:END -->

<!-- MODELFORECAST:CATEGORY-WINNERS:START -->
### Category winners

- **Best for tool calling (T0+T1)**: [nemotron-3-nano-30b-a3b](https://openrouter.ai/models/nvidia/nemotron-3-nano-30b-a3b) — 100% T0+T1
- **Best for schema compliance (T1)**: [nemotron-3-nano-30b-a3b](https://openrouter.ai/models/nvidia/nemotron-3-nano-30b-a3b) — 100% T1
- **Best for restraint (R0)**: [trinity-large-preview](https://openrouter.ai/models/arcee-ai/trinity-large-preview) — 100% R0
- **Best for multi-turn agency (A1)**: [trinity-large-preview](https://openrouter.ai/models/arcee-ai/trinity-large-preview) — 100% A1
<!-- MODELFORECAST:CATEGORY-WINNERS:END -->

<!-- MODELFORECAST:AVOID:START -->
### Avoid these models for tool calling

These models fail the basic tool invocation test (T0 < 20%). They will silently fail your agent pipeline.

- **llama-3.3-70b-instruct**: text-instead-of-tool (T0=0%)
- **minimax-m2.5**: text-instead-of-tool (T0=0%)
- **mistral-small-3.1-24b-instruct**: text-instead-of-tool (T0=0%)
- **gpt-oss-20b**: text-instead-of-tool (T0=0%)
- **qwen3-4b**: text-instead-of-tool (T0=0%)
- **qwen3-coder**: text-instead-of-tool (T0=0%)
- **qwen3-next-80b-a3b-instruct**: text-instead-of-tool (T0=0%)
<!-- MODELFORECAST:AVOID:END -->

---

## How We Test

Five probes across orthogonal capability dimensions:

| Probe | Dimension | Question |
|:-----:|-----------|----------|
| T0 | **Invoke** | Can it call a tool at all? |
| T1 | **Schema** | Does it respect parameter types? |
| T2 | **Selection** | Can it choose the right tool from many? |
| A1 | **Linear** | Can it chain tool calls across turns? |
| R0 | **Abstain** | Does it know when NOT to use tools? |

Wilson score intervals. 10 trials per test. Grades based on lowest dimension score. [Full methodology](docs/METHODOLOGY.md).

---

## The Colosseum

We also run GPU-accelerated CoreWars battles where LLMs write assembly to fight for shared memory.

```mermaid
flowchart LR
    subgraph Turn["Each Turn (x10)"]
        A[LLM] -->|writes| B[Redcode]
        B -->|battles| C[GPU MARS]
        C -->|10K fights| D[Results]
        D -->|feedback| A
    end

    subgraph Surprise["Turn 6-7"]
        E[Champion]
        E -->|boss fight| C
    end

    style A fill:#4a9eff
    style C fill:#ff6b6b
    style E fill:#ffd93d
```

Each model starts with a basic IMP (`MOV 0, 1`). They watch 10,000 battles. They write improved code. They repeat for 10 turns. At Turn 6, a surprise champion appears.

<div align="center">

**[WATCH THE BATTLES LIVE](https://jw409.github.io/modelforecast/corewars/)**

8,192 battles. Real-time visualization. 27,845 battles/sec on GPU (RTX 5090).

</div>

---

## Run It Yourself

Test your own models on your own hardware.

```bash
git clone https://github.com/jw409/modelforecast && cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
export OPENROUTER_API_KEY=your_key

# Full sweep with canary test
uv run python scripts/run_sweep.py

# Resume interrupted sweep
uv run python scripts/run_sweep.py --resume

# Regenerate results
uv run python scripts/generate_readme_results.py
```

---

## What We Learned

**Price doesn't predict performance.** NVIDIA's free nemotron-3-nano-30b scores 100% across all dimensions — matching or beating most paid models.

**Half of "tool-capable" free models can't actually call tools.** 8 of 16 free models that advertise tool support fail the basic T0 invocation test at 0%. Don't trust the label — test it.

**Small samples lie.** Wilson score intervals or you're fooling yourself.

---

## Known Caveats

### OpenRouter privacy settings silently block free endpoints

If a model scores 0% on T0 (invoke) despite being listed as tool-capable, check your OpenRouter account settings before concluding the model is broken.

**How to reproduce the trap:**
1. Log in to OpenRouter → Settings → Privacy
2. Check whether "Allow free endpoints that may train on inputs" is enabled
3. If disabled, free-tier endpoints return errors that resemble model failures — no clear error message distinguishes them

**What happened during this sweep:** `gpt-oss-120b` initially scored 0% on every probe. After enabling free endpoint access in account settings, it scored Grade A. The API returns no meaningful error — the model simply appears unresponsive.

**Mitigation before trusting a zero score:** verify the endpoint responds to a simple non-tool prompt. If that also fails with an auth-adjacent error, it is a settings issue, not a model issue.

### Rate-limited models (March 2026 sweep)

Five models were globally throttled by OpenRouter during the sweep and could not be tested:

- `meta-llama/llama-3.3-70b-instruct:free`
- `mistral/mistral-small-3.1-24b-instruct:free`
- `qwen/qwen3-4b:free`
- `qwen/qwen3-coder:free`
- `qwen/qwen3-next-80b-a3b-instruct:free`

Results for these models are pending. Re-run with `uv run python scripts/run_sweep.py --resume` when OpenRouter demand drops. Scores shown as "—" in the results table are rate-limit gaps, not failures.

---

## Contributors

Thanks to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jw409"><img src="https://avatars.githubusercontent.com/u/218849921?v=4?s=100" width="100px;" alt="jw"/><br /><sub><b>jw</b></sub></a><br /><a href="https://github.com/jw409/modelforecast/commits?author=jw409" title="Code">💻</a> <a href="https://github.com/jw409/modelforecast/commits?author=jw409" title="Documentation">📖</a> <a href="#ideas-jw409" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-jw409" title="Maintenance">🚧</a> <a href="#infra-jw409" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jw408"><img src="https://avatars.githubusercontent.com/u/218849921?v=4?s=100" width="100px;" alt="jw409"/><br /><sub><b>jw409</b></sub></a><br /><a href="https://github.com/jw409/modelforecast/commits?author=jw408" title="Code">💻</a> <a href="#ideas-jw408" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

---

MIT License · *Not affiliated with OpenRouter*
