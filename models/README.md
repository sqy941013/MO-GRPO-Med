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

# Resume from checkpoint (replace with your path)
python ./models/train_sft.py --task DI --resume_from_checkpoint checkpoints/sft/DI_*
```

Important

- Run `datasets/preprocess.py` first to generate the CSVs.
- If you patched TRL/Unsloth-Zoo, ensure you have run `scripts/reinstall_trl_unsloth_zoo.sh` so the local editable versions are active.
- The script auto-sets a chat template for DI; adapt `set_chat_template_for_di` if you change tasks/prompts.

Arguments (selected)

- `--task`: `DI` or `BCH`
- `--data_dir`: directory containing `*_train_sft.csv` and `*_dev.csv` (default `datasets/processed`)
- `--model_name`: base model to fine-tune
- `--max_seq_len`: sequence length (default 8192)
- `--max_sample`: limit training/dev rows for quick experiments (0 = unlimited)
- `--batch_size`, `--grad_accum`, `--lr`, `--epochs`, `--max_steps`, `--warmup_steps`, `--weight_decay`, `--optimizer`, `--lr_scheduler`
- `--save_strategy`, `--save_steps`, `--output_root`, `--save_format` (lora | merged_16bit | merged_4bit | gguf)
- `--use_gradient_checkpointing`, `--random_state`, `--resume_from_checkpoint`

Outputs

- Checkpoints under `checkpoints/sft/<TASK>_<MODEL>_<TIMESTAMP>/` in the selected save format.
