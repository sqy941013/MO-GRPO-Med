# MO-GRPO-Med-code
Official implementation of MO-GRPO-Med: a multi-objective, critic-free GRPO pipeline for clinical discharge instructions with expectile baseline and Huberized advantage, including SFT/RL training, four reward heads (structure/coverage/medical-factuality/style), and evaluation on MIMIC-IV-Note.

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
