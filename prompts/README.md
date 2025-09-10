# Prompts

Prompt templates used by MO-GRPO-Med for evaluation, rewards, structure extraction, and DI generation.

## Layout

- `di_judge/`: rubric-based DI grader
	- `system.txt`: senior clinician–educator rubric with JSON schema (scores, rationales, improvements, overall stub)
	- `user.txt`: input wrapper; replace `{{DI_TEXT}}` with the Discharge Instruction to grade

- `reward_function/`: prompts for reward modeling in RL
	- `r_cover/`: coverage/completeness (system/user pair)
	- `r_medfact/`: medical factual consistency and safety
	- `r_struct/`: structure/layout adherence
	- `r_style/`: language clarity/readability/tone

- `R_struct_extract.txt`: extractor prompt to parse a DI into 9 canonical sections and return a fixed-key JSON

- `reasoning_datset/system.txt`: DI generation system prompt (physician letter-style) used for reasoning SFT/data

## Usage (minimal)

1) DI Grader

```python
from pathlib import Path
di_text = Path("examples/sample_di.txt").read_text(encoding="utf-8")
system = Path("prompts/di_judge/system.txt").read_text(encoding="utf-8")
user_t = Path("prompts/di_judge/user.txt").read_text(encoding="utf-8")
user = user_t.replace("{{DI_TEXT}}", di_text)
# Send [system, user] to a chat model; expect a single JSON object per the schema.
```

2) Structure Extractor

```python
from pathlib import Path
extractor = Path("prompts/R_struct_extract.txt").read_text(encoding="utf-8")
note_text = Path("examples/sample_di.txt").read_text(encoding="utf-8")
# Use as the sole system prompt; pass the note as user content. Expect fixed-key JSON with 9 sections.
```

3) Reward Prompts

- Each subfolder contains `system.txt` and `user.txt`. In your RM/evaluator, compose messages and parse outputs
	into numeric scores or pass/fail signals. Ensure consistent scaling/normalization across rewards.

4) Reasoning Dataset Prompt

- `reasoning_datset/system.txt` defines the letter-style DI generation policy for data synthesis and reasoning SFT.
- It enforces section order, de-identification, and strict “no invention” rules.

## Notes

- Prefer JSON-only completions where the system message requires it; avoid Markdown fences.
- Preserve exact key names for extractors to simplify downstream parsing.
- You can customize these prompts, but keep schema compatibility if code depends on specific fields.
