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
