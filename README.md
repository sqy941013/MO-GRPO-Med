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
