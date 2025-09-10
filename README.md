# MO-GRPO-Med
Official implementation of MO-GRPO-MED: A MULTI-OBJECTIVE FRAMEWORK FOR GENERATING SAFE AND HIGH-QUALITY DISCHARGE INSTRUCTIONS.

## Environment Setup

```bash
# create and activate a dedicated conda environment
conda create -n mo-grpo-med python=3.12
conda activate mo-grpo-med

# install PyTorch (CUDA 12.6)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# install Unsloth
pip install unsloth==2025.8.9

# install remaining dependencies
pip install -r requirements.txt
```

## Environment Variables

This project reads configuration from environment variables. Use a `.env` file in the project root to store secrets for local development.

- Copy `.env.example` to `.env` and fill in the values you need.
- Never commit your `.env` file to version control.

Variables:

- OPENAI_API_BASE: Base URL for the OpenAI (or OpenAI-compatible) API. Example: https://api.openai.com/v1
- OPENAI_API_KEY: Your API key for the provider specified by OPENAI_API_BASE.
- WANDB_API_KEY (optional): API key for Weights & Biases if you use experiment tracking.
- HF_TOKEN (optional): Hugging Face access token for private models/datasets.

Tips:

- If you use an OpenAI-compatible endpoint (e.g., a proxy), set both OPENAI_API_BASE and OPENAI_API_KEY to the values provided by that service.
- For production deployments, set these variables via your runtime’s secret manager or environment, not via a `.env` file.

## Third-Party Libraries (Patched)

This project modifies specific versions of two third-party libraries to add research features from the paper:

- TRL 0.19.1: patched to support expectile baseline and Huber-style advantage handling.
- Unsloth-Zoo 2025.8.9: patched to wire the same features into GRPO training.

What changed

- Expectile baseline: toggled by --use_expectile_baseline and controlled by --expectile_tau; computes a baseline as an expectile of group returns, which can be more robust to outliers.
- Huberized advantages: optional clamping via --use_advantage_delta and --advantage_delta for stable optimization when large advantage magnitudes appear.
- Optional z-score: --normalize_by_std divides by group standard deviation to normalize advantages.

How to install the patched packages locally

1) Download the original packages that match the versions above.
2) Overwrite their code with our improved implementations placed under third_party/:
	- third_party/trl-0.19.1/
	- third_party/unsloth-zoo/
3) From the project root, run the reinstall script to install them in editable mode:

```bash
bash scripts/reinstall_trl_unsloth_zoo.sh
```

Training flags (CLI)

Add these arguments to your training script to control the new behavior:

- --use_expectile_baseline (flag): Use τ-expectile baseline instead of mean.
- --expectile_tau (float, default=0.7): Tau parameter for expectile baseline calculation.
- --normalize_by_std (flag): Divide by group std (z-score) when computing advantages.
- --use_advantage_delta (flag): Enable advantage clamp via advantage_delta in GRPO loss.
- --advantage_delta (float, default=0.65): Clamp magnitude for advantages when enabled.
- --tau (float, default=0.7): Legacy tau parameter (use --expectile_tau instead).

Example snippet in argparse

```python
p.add_argument("--use_expectile_baseline", action="store_true", help="Use τ-expectile baseline instead of mean.")
p.add_argument("--expectile_tau", type=float, default=0.7, help="Tau parameter for expectile baseline calculation.")
p.add_argument("--normalize_by_std", action="store_true", help="Divide by group std (z-score).")
p.add_argument("--use_advantage_delta", action="store_true", help="Enable advantage clamp via advantage_delta in GRPO loss.")
p.add_argument("--advantage_delta", type=float, default=0.65, help="Clamp magnitude for advantages when enabled.")
```

## Dataset Preprocessing (MIMIC-IV-Ext-iDS → SFT/RL)

We provide a helper script to transform MIMIC-IV-Ext-iDS into SFT and RL training splits with subject-level isolation.

- Script: `datasets/preprocess.py`
- Default input: `datasets/raw/MIMIC-IV-Ext_iDS_with_PR.csv`
- Outputs (under `datasets/processed/`):
	- BCH: `BCH_train_sft.csv`, `BCH_train_rl.csv`, `BCH_dev.csv`, `BCH_test.csv`
	- DI:  `DI_train_sft.csv`,  `DI_train_rl.csv`,  `DI_dev.csv`,  `DI_test.csv`

Tasks

- BCH: predict `brief_hospital_course` from structured inputs
- DI:  predict `discharge_instructions` from structured inputs plus hospital course and discharge info

Important

- Run the script from the `datasets/` directory so relative paths resolve correctly.
- Splits are subject-level (patients do not overlap across train/dev/test).
- If your file name differs, either rename it to `MIMIC-IV-Ext_iDS_with_PR.csv` or modify `RAW_CSV` inside `preprocess.py`.

Usage (Ubuntu/Linux)

```bash
# Navigate to the datasets folder
cd ./datasets

# BCH task
python ./preprocess.py --mode BCH --split 9,0.5,0.5 --rl_ratio 0.2 --seed 42

# DI task
python ./preprocess.py --mode DI  --split 9,0.5,0.5 --rl_ratio 0.2 --seed 42

# Optional caps for quick experiments (row and patient limits)
python ./preprocess.py --mode DI --max_dev 1500 --max_test 1500 --max_dev_patient 0 --max_test_patient 0
```

Arguments

- `--mode`: `BCH` or `DI`; selects input/target columns for the task.
- `--split`: train,dev,test patient ratios (default `9,0.5,0.5` → normalized to 90/5/5).
- `--rl_ratio`: fraction of `train_sft` reused as `train_rl` (default `0.2`).
- `--max_dev`/`--max_test`: row caps for dev/test (0 = unlimited).
- `--max_dev_patient`/`--max_test_patient`: patient caps for dev/test (0 = unlimited).
- `--seed`: random seed for reproducibility.

## SFT Training (Quick Start)

After preprocessing the dataset, start SFT with the Unsloth-based trainer:

```bash
# DI task
python ./models/train_sft.py --task DI --data_dir datasets/processed --batch_size 4 --grad_accum 4 --epochs 2 --lr 2e-5

# BCH task
python ./models/train_sft.py --task BCH --data_dir datasets/processed --batch_size 4 --grad_accum 4 --epochs 2 --lr 2e-5

# Reasoning SFT
python ./models/train_sft.py --reasoning_sft \
	--reasoning_dataset_path data/related_datasets/MO-GRPO-Med-Reasoning-Dataset.csv \
	--batch_size 2 --grad_accum 8 --epochs 3 --lr 2e-4
```

Notes

- Ensure patched TRL/Unsloth-Zoo are installed via `scripts/reinstall_trl_unsloth_zoo.sh` if you use the expectile/HAL features.
- More options and details are in `models/README.md`.

### Multi-GPU (torchrun)

```bash
# 4 GPUs example
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
	./models/train_sft_multigpu.py \
	--task DI \
	--data_dir datasets/processed \
	--batch_size 2 \
	--grad_accum 8 \
	--epochs 2 \
	--lr 2e-5
```

Guidance for Unsloth multi-GPU

- You can use Accelerate or DeepSpeed with Unsloth to run DDP/FSDP today.
- Ensure `ddp_find_unused_parameters = False` in SFTConfig/TrainingArguments (already set in this repo).
- Launch options:
	- `accelerate launch ./models/train_sft_multigpu.py --task DI ...`
	- `torchrun --nproc_per_node N ./models/train_sft_multigpu.py --task DI ...`
- If VRAM is insufficient per GPU, enable pipeline/model sharding by loading with `device_map="balanced"`:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
		"unsloth/Llama-3.3-70B-Instruct",
		load_in_4bit=True,
		device_map="balanced",
)
```

Community repos that improve multi-GPU with Unsloth:

- unsloth-5090-multiple
- opensloth

## Prompts (Templates for Evaluation, Rewards, Extraction, and Generation)

This repo ships reusable prompt assets for judging, reward shaping, structure extraction, and DI generation.

Locations

- `prompts/di_judge/`: clinician-style rubric to grade a Discharge Instruction (DI)
	- `system.txt`: scoring rubric, JSON schema, and instructions
	- `user.txt`: input template; replace `{{DI_TEXT}}` with the DI to grade
- `prompts/reward_function/`: LLM prompts used as reward sources in RL
	- `r_cover/`: coverage/completeness
	- `r_medfact/`: medical factual consistency/safety
	- `r_struct/`: structure/format alignment
	- `r_style/`: language clarity/readability/tone
	- Each subfolder contains `system.txt` and `user.txt`, to be wired into your RM harness
- `prompts/R_struct_extract.txt`: extractor prompt to parse a DI into 9 canonical sections (fixed JSON keys)
- `prompts/reasoning_datset/system.txt`: system prompt to generate DI in physician letter-style (for reasoning SFT/data)

Using with the OpenAI helper

`scripts/openai_helper.py` provides two utilities:

- `load_client()`: loads `OPENAI_API_BASE` and `OPENAI_API_KEY` from environment or `.env` and returns a client.
- `chat_json_extract(...)`: calls chat completions and returns parsed JSON, with smart handling for DeepSeek-R1 models.

DI grading example (JSON parsed for you)

```python
from pathlib import Path
import sys
sys.path.append("scripts")  # ensure scripts/ is importable
from openai_helper import load_client, chat_json_extract

client = load_client()
model = "gpt-4o-mini"  # or a compatible model; for DeepSeek-R1 see notes below

di_text = Path("examples/sample_di.txt").read_text(encoding="utf-8")
system = Path("prompts/di_judge/system.txt").read_text(encoding="utf-8")
user_t = Path("prompts/di_judge/user.txt").read_text(encoding="utf-8")
user = user_t.replace("{{DI_TEXT}}", di_text)

result = chat_json_extract(client, model, system, user, temperature=0.2)
print(result)  # a Python dict parsed from the model's JSON output
```

Structure extraction example

```python
from pathlib import Path
import sys
sys.path.append("scripts")
from openai_helper import load_client, chat_json_extract

client = load_client()
model = "gpt-4o-mini"

note_text = Path("examples/sample_di.txt").read_text(encoding="utf-8")
system = Path("prompts/R_struct_extract.txt").read_text(encoding="utf-8")
user = note_text  # extractor prompt expects the note as user content

sections = chat_json_extract(client, model, system, user)
print(sections.keys())  # 9 canonical section keys
```

Notes

- The helper forces `response_format={"type":"json_object"}` for non‑R1 models, and automatically strips `<think>...</think>` for DeepSeek‑R1 before parsing JSON. Keep the prompts' JSON schema unchanged.
- Set `GRPO_DEBUG=1` to print raw outputs to stderr when JSON parsing fails.
- You can stream incrementally via `chat_json_extract(..., stream=True)`; final JSON is parsed after streaming completes.
- Use `return_raw=True` to also get the original text: `parsed, raw = chat_json_extract(..., return_raw=True)`.
