# Reward functions (MO-GRPO-Med)

This folder contains the LLM-based reward functions used during GRPO training. They evaluate generated Discharge Instructions (DI) along four dimensions and can be combined into a single reward.

Dimensions

- Structure (r_struct): alignment to a 9-section canonical format.
- Coverage (r_cover): completeness with respect to gold and source anchors.
- Medical factuality & safety (r_medfact): correctness/contraindications per extracted items and source.
- Style (r_style): readability, clarity, tone, and conciseness.

High-level flow

1) From a generated DI, extract the 9-section map using an LLM prompt.
2) From that map, run information extraction (IE) to produce anchors per section.
3) Compare against gold/source anchors for coverage; compute structure similarity; evaluate medfact.
4) Optionally compute a style score in parallel.
5) Aggregate into a weighted total reward.

Key modules

- struct.py: LLM-based structure extraction and final r_struct computation.
- cover.py: IE extraction from generated map; coverage/precision metrics.
- medfact.py: medical factuality/safety scoring from IE results and source row.
- style.py: style scoring from the final DI text.
- aggregate.py: orchestrates the above and returns a unified reward (compute_total_reward).
- openai_io.py: wraps the OpenAI-compatible helper with retries and streaming.
- cache.py / logs.py / utils.py: caching, logging, utilities.

Inputs and outputs

- Inputs (per sample):
  - prompts: chat messages used for generation (only the user content is used for logs).
  - completions: model outputs (the DI may include a <think> section; the code extracts <answer>).
  - answer: optional gold DI (not required by rewards).
  - note_id: identifier for logging.
  - formated_source_note: the formatted source note string for medfact checks.
  - gold_map_obj: dict of 9-section gold text.
  - t_source_anchors_obj: dict of anchors derived from source text.
  - t_gold_anchors_obj: dict of anchors from gold DI.
  - t_all_anchors_obj: dict of combined anchors (optional, used for precision).

- Outputs:
  - Per-sample float score in [0, 1] (0 if extraction fails or DI is empty).
  - Intermediate artifacts logged to JSONL when GRPO_OUTPUT_DIR is set.

Environment variables

- OPENAI_API_BASE / OPENAI_API_KEY: endpoint and key for reward LLM calls.
- REWARD_DS_V3_MODEL: model for DeepSeek-V3-style prompts (default: deepseek-v3-250324).
- REWARD_DS_R1_MODEL: model for DeepSeek-R1-style prompts (default: deepseek-r1-250528).
- LLM_STREAM: "1" to enable streaming (default); "0" to disable.
- GRPO_DEBUG: "1" to print raw outputs if JSON parsing fails.
- GRPO_OUTPUT_DIR: if set, logs go under <dir>/logs/.
- GRPO_CACHE_DIR: base dir for caches; reward_cache.csv will be placed here by default.
- REWARD_CACHE_CSV: explicit path to reward_cache.csv (overrides GRPO_CACHE_DIR).

Caching

- The cache de-duplicates LLM calls by [type, key_sha1, model]. Use it to accelerate repeated training runs.
- Default path: `${GRPO_CACHE_DIR}/reward_cache.csv`. You can use a shared path across machines if needed.

Logging

- logs/llm_calls.jsonl: each LLM call with prompts, raw output, and parsed JSON.
- logs/reward_steps.jsonl: per-generation reward components and intermediate values.
- logs/<reward>_zeros.jsonl: samples where a reward ended up as 0.

Reliability notes

- JSON schemas are enforced by scripts/openai_helper.py. For DeepSeek-R1 outputs, `<think>...</think>` is stripped before parsing.
- The retry wrapper backs off exponentially (`base_wait ** attempt + jitter`). Increase retries if your endpoint is flaky.

Usage

- In the GRPO trainer we recommend using `compute_total_reward`, which internally orchestrates struct/cover/medfact/style.
- If you want to ablate dimensions, pass a custom `reward_funcs=[...]` to GRPOTrainer (e.g., only `compute_r_struct_reward`).

Weights

- Default aggregation weights in aggregate.py:
  - struct: 0.15, cover: 0.35, medfact: 0.40, style: 0.10
- Adjust inside `compute_total_reward` if you need a different balance.

Troubleshooting

- Empty or malformed DI: the code extracts content between `<answer>...</answer>`; missing tags lead to 0 score.
- Very long outputs: set `LLM_STREAM=0` to disable streaming or reduce generation length.
- Cache not used: ensure deterministic prompts; the cache key includes `model` and an SHA1 of the inputs.
