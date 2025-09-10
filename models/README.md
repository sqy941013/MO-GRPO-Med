# Models

Scripts and notes for training models used in MO-GRPO-Med.

## SFT Training (train_sft.py)

Supervised fine-tuning using Unsloth + TRL. The script expects preprocessed CSVs from `datasets/preprocess.py` under `datasets/processed/`.

Inputs

- DI task: `datasets/processed/DI_train_sft.csv` and `datasets/processed/DI_dev.csv`
- BCH task: `datasets/processed/BCH_train_sft.csv` and `datasets/processed/BCH_dev.csv`

Key defaults

- Model: `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` (4-bit loaded)
- Seq length: 8192
- LoRA: r=64, alpha=64, target modules q/k/v/o + MLP
- Output root: `checkpoints/sft`

Run (Ubuntu/Linux)

```bash
# DI task
python ./models/train_sft.py --task DI --data_dir datasets/processed --batch_size 4 --grad_accum 4 --epochs 2 --lr 2e-5

# BCH task
python ./models/train_sft.py --task BCH --data_dir datasets/processed --batch_size 4 --grad_accum 4 --epochs 2 --lr 2e-5
# Reasoning SFT (uses a separate dataset and chat template)
python ./models/train_sft.py --reasoning_sft \
	--reasoning_dataset_path data/related_datasets/MO-GRPO-Med-Reasoning-Dataset.csv \
	--batch_size 2 --grad_accum 8 --epochs 3 --lr 2e-4

# Resume from checkpoint (replace with your path)
python ./models/train_sft.py --task DI --resume_from_checkpoint checkpoints/sft/DI_*
```

Important

- Run `datasets/preprocess.py` first to generate the CSVs.
- If you patched TRL/Unsloth-Zoo, ensure you have run `scripts/reinstall_trl_unsloth_zoo.sh` so the local editable versions are active.
- The script auto-sets a chat template for DI; adapt `set_chat_template_for_di` if you change tasks/prompts.
 - For reasoning SFT, `--reasoning_sft` switches to a separate reasoning dataset pipeline and installs a `<think> ... </think><answer> ... </answer>` chat template.

Arguments (selected)

- `--task`: `DI` or `BCH`
- `--data_dir`: directory containing `*_train_sft.csv` and `*_dev.csv` (default `datasets/processed`)
- `--model_name`: base model to fine-tune
- `--max_seq_len`: sequence length (default 8192)
- `--max_sample`: limit training/dev rows for quick experiments (0 = unlimited)
- `--batch_size`, `--grad_accum`, `--lr`, `--epochs`, `--max_steps`, `--warmup_steps`, `--weight_decay`, `--optimizer`, `--lr_scheduler`
- `--save_strategy`, `--save_steps`, `--output_root`, `--save_format` (lora | merged_16bit | merged_4bit | gguf)
- `--use_gradient_checkpointing`, `--random_state`, `--resume_from_checkpoint`
Reasoning-only flags

- `--reasoning_sft`: enable reasoning mode.
- `--reasoning_dataset_path`: path to the reasoning CSV.
- `--reasoning_length_quantile`: keep samples below this token-length quantile (default 0.9; set 0 to disable).
- `--reasoning_limit_rows`: cap rows for quick tests (0 = unlimited).

Outputs

- Checkpoints under `checkpoints/sft/<TASK>_<MODEL>_<TIMESTAMP>/` in the selected save format.

## Multi-GPU SFT Training (train_sft_multigpu.py)

Distributed SFT using Unsloth + TRL with Hugging Face Accelerate/torchrun semantics.

- Unsloth works with Accelerate/DeepSpeed; you can use DDP/FSDP today.
- This repo already sets `ddp_find_unused_parameters = False` in SFTConfig as recommended.
- Launch via either:
	- Accelerate: `accelerate launch ./models/train_sft_multigpu.py --task DI ...`
	- Torchrun: `torchrun --nproc_per_node N ./models/train_sft_multigpu.py --task DI ...`

Pipeline/model sharding (insufficient VRAM per GPU)

To enable model splitting across GPUs (e.g., for very large models), pass `device_map="balanced"` at load time:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
		"unsloth/Llama-3.3-70B-Instruct",
		load_in_4bit=True,
		device_map="balanced",
)
```

Community efforts

- unsloth-5090-multiple: a fork focused on efficient multi-GPU for RTX 5090-like setups.
- opensloth: Unsloth with experimental multi-GPU training features.

Official, simplified multi-GPU support is in progress upstream—watch Unsloth announcements for updates.
