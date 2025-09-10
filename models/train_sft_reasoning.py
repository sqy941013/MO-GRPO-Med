"""
Deprecated: Use models/train_sft.py with --reasoning_sft instead.

Example (Ubuntu/Linux):

  python ./models/train_sft.py \
    --reasoning_sft \
    --reasoning_dataset_path data/related_datasets/MO-GRPO-Med-Reasoning-Dataset.csv \
    --batch_size 2 \
    --grad_accum 8 \
    --epochs 3 \
    --lr 2e-4
"""

import sys

def main():
    print(
        "[DEPRECATED] Please run models/train_sft.py with --reasoning_sft.\n"
        "See models/README.md for usage."
    )

if __name__ == "__main__":
    main()
