# MO-GRPO-Med

Official implementation of **MO-GRPO-Med: A Multi-Objective Framework for Generating Safe and High-Quality Discharge Instructions**.

**Paper Accepted at IEEE ICASSP 2026!** 🎉

---

## 📑 Table of Contents

- [Environment Setup](#-environment-setup)
- [Environment Variables](#-environment-variables)
- [Third-Party Libraries](#-third-party-libraries-patched)
- [Dataset Preprocessing](#-dataset-preprocessing)
- [SFT Training](#-sft-training)
- [Multi-GPU Training](#multi-gpu-training)
- [Prompts and Templates](#-prompts-and-templates)
- [GRPO Training](#-grpo-training-reinforcement-learning)
- [Citation](#-citation)

---

## 🚀 Environment Setup

### Step 1: Create Conda Environment

```bash
# Create and activate a dedicated conda environment
conda create -n mo-grpo-med python=3.12
conda activate mo-grpo-med
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (CUDA 12.6)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install Unsloth
pip install unsloth==2025.8.9

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

This project reads configuration from environment variables. Use a `.env` file in the project root to store secrets for local development.

### Quick Setup

1. Copy `.env.example` to `.env` and fill in the values you need
2. Never commit your `.env` file to version control

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_BASE` | Base URL for OpenAI (or OpenAI-compatible) API | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | Your API key for the provider | `sk-...` |

### Optional Variables

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | API key for Weights & Biases experiment tracking |
| `HF_TOKEN` | Hugging Face access token for private models/datasets |

### Tips

- **OpenAI-compatible endpoints**: Set both `OPENAI_API_BASE` and `OPENAI_API_KEY` to the values provided by your service (e.g., proxy)
- **Production deployments**: Use your runtime's secret manager or environment variables instead of a `.env` file

---

## 🔧 Third-Party Libraries (Patched)

This project modifies specific versions of two third-party libraries to add research features from the paper:

- **TRL 0.19.1**: Patched to support expectile baseline and Huber-style advantage handling
- **Unsloth-Zoo 2025.8.9**: Patched to wire the same features into GRPO training

### What Changed

- **Expectile baseline**: Toggled by `--use_expectile_baseline` and controlled by `--expectile_tau`; computes a baseline as an expectile of group returns, which can be more robust to outliers
- **Huberized advantages**: Optional clamping via `--use_advantage_delta` and `--advantage_delta` for stable optimization when large advantage magnitudes appear
- **Optional z-score**: `--normalize_by_std` divides by group standard deviation to normalize advantages

### Installation

1. Download the original packages that match the versions above
2. Overwrite their code with our improved implementations placed under `third_party/`:
   - `third_party/trl-0.19.1/`
   - `third_party/unsloth-zoo/`
3. From the project root, run the reinstall script to install them in editable mode:

```bash
bash scripts/reinstall_trl_unsloth_zoo.sh
```

### Training Flags (CLI)

Add these arguments to your training script to control the new behavior:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_expectile_baseline` | flag | - | Use τ-expectile baseline instead of mean |
| `--expectile_tau` | float | 0.7 | Tau parameter for expectile baseline calculation |
| `--normalize_by_std` | flag | - | Divide by group std (z-score) when computing advantages |
| `--use_advantage_delta` | flag | - | Enable advantage clamp via advantage_delta in GRPO loss |
| `--advantage_delta` | float | 0.65 | Clamp magnitude for advantages when enabled |
| `--tau` | float | 0.7 | Legacy tau parameter (use `--expectile_tau` instead) |

### Example Snippet in Argparse

```python
p.add_argument("--use_expectile_baseline", action="store_true", help="Use τ-expectile baseline instead of mean.")
p.add_argument("--expectile_tau", type=float, default=0.7, help="Tau parameter for expectile baseline calculation.")
p.add_argument("--normalize_by_std", action="store_true", help="Divide by group std (z-score).")
p.add_argument("--use_advantage_delta", action="store_true", help="Enable advantage clamp via advantage_delta in GRPO loss.")
p.add_argument("--advantage_delta", type=float, default=0.65, help="Clamp magnitude for advantages when enabled.")
```

---

## 📊 Dataset Preprocessing

We provide a helper script to transform MIMIC-IV-Ext-iDS into SFT and RL training splits with subject-level isolation.

### Overview

- **Script**: `datasets/preprocess.py`
- **Default input**: `datasets/raw/MIMIC-IV-Ext_iDS_with_PR.csv`
- **Outputs** (under `datasets/processed/`):
  - **BCH**: `BCH_train_sft.csv`, `BCH_train_rl.csv`, `BCH_dev.csv`, `BCH_test.csv`
  - **DI**: `DI_train_sft.csv`, `DI_train_rl.csv`, `DI_dev.csv`, `DI_test.csv`

### Tasks

- **BCH**: Predict `brief_hospital_course` from structured inputs
- **DI**: Predict `discharge_instructions` from structured inputs plus hospital course and discharge info

### Important Notes

- Run the script from the `datasets/` directory so relative paths resolve correctly
- Splits are subject-level (patients do not overlap across train/dev/test)
- If your file name differs, either rename it to `MIMIC-IV-Ext_iDS_with_PR.csv` or modify `RAW_CSV` inside `preprocess.py`

### Usage (Ubuntu/Linux)

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

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `BCH` or `DI`; selects input/target columns for the task | - |
| `--split` | Train,dev,test patient ratios | `9,0.5,0.5` (90/5/5%) |
| `--rl_ratio` | Fraction of `train_sft` reused as `train_rl` | `0.2` |
| `--max_dev` / `--max_test` | Row caps for dev/test (0 = unlimited) | `0` |
| `--max_dev_patient` / `--max_test_patient` | Patient caps for dev/test (0 = unlimited) | `0` |
| `--seed` | Random seed for reproducibility | `42` |

---

## 🎯 SFT Training

After preprocessing the dataset, start SFT with the Unsloth-based trainer.

### Quick Start

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

### Notes

- Ensure patched TRL/Unsloth-Zoo are installed via `scripts/reinstall_trl_unsloth_zoo.sh` if you use the expectile/HAL features
- More options and details are in `models/README.md`

---

## 🖥️ Multi-GPU Training

### Using torchrun

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

### Guidance for Unsloth Multi-GPU

- You can use Accelerate or DeepSpeed with Unsloth to run DDP/FSDP today
- Ensure `ddp_find_unused_parameters = False` in SFTConfig/TrainingArguments (already set in this repo)
- **Launch options**:
  - `accelerate launch ./models/train_sft_multigpu.py --task DI ...`
  - `torchrun --nproc_per_node N ./models/train_sft_multigpu.py --task DI ...`
- **If VRAM is insufficient per GPU**, enable pipeline/model sharding by loading with `device_map="balanced"`:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
	"unsloth/Llama-3.3-70B-Instruct",
	load_in_4bit=True,
	device_map="balanced",
)
```

### Community Repos

Community repos that improve multi-GPU with Unsloth:
- [unsloth-5090-multiple](https://github.com/unslothai/unsloth-5090-multiple)
- [opensloth](https://github.com/unslothai/opensloth)

---

## 📝 Prompts and Templates

This repo ships reusable prompt assets for judging, reward shaping, structure extraction, and DI generation.

### Locations

- **`prompts/di_judge/`**: Clinician-style rubric to grade a Discharge Instruction (DI)
  - `system.txt`: Scoring rubric, JSON schema, and instructions
  - `user.txt`: Input template; replace `{{DI_TEXT}}` with the DI to grade
- **`prompts/reward_function/`**: LLM prompts used as reward sources in RL
  - `r_cover/`: Coverage/completeness
  - `r_medfact/`: Medical factual consistency/safety
  - `r_struct/`: Structure/format alignment
  - `r_style/`: Language clarity/readability/tone
  - Each subfolder contains `system.txt` and `user.txt`, to be wired into your RM harness
- **`prompts/R_struct_extract.txt`**: Extractor prompt to parse a DI into 9 canonical sections (fixed JSON keys)
- **`prompts/reasoning_datset/system.txt`**: System prompt to generate DI in physician letter-style (for reasoning SFT/data)

### Using with the OpenAI Helper

`scripts/openai_helper.py` provides two utilities:

- `load_client()`: Loads `OPENAI_API_BASE` and `OPENAI_API_KEY` from environment or `.env` and returns a client
- `chat_json_extract(...)`: Calls chat completions and returns parsed JSON, with smart handling for DeepSeek-R1 models

### DI Grading Example (JSON Parsed for You)

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

### Structure Extraction Example

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

### Notes

- The helper forces `response_format={"type":"json_object"}` for non-R1 models, and automatically strips `<think>...</think>` for DeepSeek-R1 before parsing JSON. Keep the prompts' JSON schema unchanged
- Set `GRPO_DEBUG=1` to print raw outputs to stderr when JSON parsing fails
- You can stream incrementally via `chat_json_extract(..., stream=True)`; final JSON is parsed after streaming completes
- Use `return_raw=True` to also get the original text: `parsed, raw = chat_json_extract(..., return_raw=True)`

---

## 🤖 GRPO Training (Reinforcement Learning)

Train the DI generator with GRPO using multiple LLM-based reward functions (structure, coverage, medical factuality/safety, and style).

### Prerequisites

- Ensure the patched TRL 0.19.1 and Unsloth-Zoo 2025.8.9 are installed (see [Third-Party Libraries](#-third-party-libraries-patched))
- Set your OpenAI-compatible endpoint in environment variables or a `.env` file: `OPENAI_API_BASE`, `OPENAI_API_KEY`
- Prepare a GRPO CSV (default path `data/processed/mo-grpo-med_dataset.csv`). Required columns:
  - `note_id`, `formated_source_note`, `gold_di`
  - `gold_map`, `t_source_anchors`, `t_gold_anchors` (and optionally `t_all_anchors`)

### Quick Start

```powershell
# PowerShell (Windows)
$env:OPENAI_API_BASE = "https://api.openai.com/v1"
$env:OPENAI_API_KEY = "YOUR_KEY"
# Optional: choose reward LLMs (DeepSeek-style IDs shown as examples)
$env:REWARD_DS_V3_MODEL = "deepseek-v3-250324"
$env:REWARD_DS_R1_MODEL = "deepseek-r1-250528"
# Optional: caching and logs
$env:GRPO_CACHE_DIR = "data\intermediate"   # where reward_cache.csv will live (can override by REWARD_CACHE_CSV)
$env:LLM_STREAM = "1"                         # stream LLM outputs for robustness
$env:GRPO_DEBUG = "0"                         # set to 1 to dump raw JSON on parse errors

python .\models\mo-grpo-med_train.py `
	--model_name unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit `
	--csv_path data/processed/mo-grpo-med_dataset.csv `
	--limit_samples 500 `
	--max_steps 50 `
	--use_expectile_baseline --expectile_tau 0.7 `
	--use_advantage_delta --advantage_delta 0.65
```

### Key Flags (Selected)

| Flag | Description |
|------|-------------|
| `--model_name` | Base model to train with Unsloth (4-bit recommended for VRAM efficiency) |
| `--csv_path` | Path to the GRPO training CSV (see required columns above) |
| `--limit_samples` | Downsample rows for quick experiments (0 = use all) |
| `--max_steps` | Number of GRPO steps |
| `--use_expectile_baseline` | Use expectile baseline (robust baseline) |
| `--expectile_tau` | Tau parameter for expectile baseline (default 0.7) |
| `--use_advantage_delta` | Enable Huberized advantages (stability) |
| `--advantage_delta` | Advantage delta value (default 0.65) |
| `--reward_ds_v3_model` | Reward LLM selection (also available via env) |
| `--reward_ds_r1_model` | Reward LLM selection (also available via env) |

### Outputs and Logging

- Checkpoints and logs under `checkpoints/grpo/<MODEL>_<TIMESTAMP>/`
- The script sets `GRPO_OUTPUT_DIR` to that folder for reward logs
- **Reward logs (JSONL)** include:
  - `logs/llm_calls.jsonl`: Raw/parsed LLM responses used by rewards
  - `logs/reward_steps.jsonl`: Per-generation intermediate reward data
  - `logs/*_zeros.jsonl`: Samples where a specific reward evaluated to 0
- **Caching**: `reward_cache.csv` (default path: `${GRPO_CACHE_DIR}/reward_cache.csv` or override via `REWARD_CACHE_CSV`)

### Troubleshooting

- **JSON parsing issues**: Set `GRPO_DEBUG=1` to print raw outputs; the helper automatically strips `<think>...</think>` for DeepSeek-R1 before parsing
- **Rate limits/instability**: The reward caller retries with exponential backoff; you can lower `LLM_STREAM` to `0` to disable streaming if needed
- **Missing CSV columns**: Ensure all required columns exist; otherwise the script will raise a clear error

For details on each reward component (structure/coverage/medfact/style), see `models/rewards/README.md`.

---

## 📝 Citation

This work has been accepted for presentation at the **IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2026)**, to be held in Barcelona, Spain, May 4-8, 2026.

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{shen2026mogrpomed,
  title={MO-GRPO-Med: A Multi-Objective Framework for Generating Safe and High-Quality Discharge Instructions},
  author={Shen, Qingyang and Zhang, Xiaozhi and Guo, Quan and Yi, Zhang},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  month={May},
  address={Barcelona, Spain},
  organization={IEEE},
  note={Paper ID: 11885}
}
```
