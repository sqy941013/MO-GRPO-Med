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

- Expectile baseline: controlled by --expectile_tau; computes a baseline as an expectile of group returns, which can be more robust to outliers.
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

- --expectile_tau (float, default=0.7): Tau parameter for expectile baseline calculation.
- --normalize_by_std (flag): Divide by group std (z-score) when computing advantages.
- --use_advantage_delta (flag): Enable advantage clamp via advantage_delta in GRPO loss.
- --advantage_delta (float, default=0.65): Clamp magnitude for advantages when enabled.
- --tau (float, default=0.7): Legacy tau parameter (use --expectile_tau instead).

Example snippet in argparse

```python
p.add_argument("--expectile_tau", type=float, default=0.7, help="Tau parameter for expectile baseline calculation.")
p.add_argument("--normalize_by_std", action="store_true", help="Divide by group std (z-score).")
p.add_argument("--use_advantage_delta", action="store_true", help="Enable advantage clamp via advantage_delta in GRPO loss.")
p.add_argument("--advantage_delta", type=float, default=0.65, help="Clamp magnitude for advantages when enabled.")
p.add_argument("--tau", type=float, default=0.7, help="Legacy tau parameter (use --expectile_tau instead)")
```

Notes

- --use_huber_advantage is commented out above but kept for clarity; the recommended path is --use_advantage_delta which aligns with the current GRPO implementation.
- Ensure your environment uses the editable installs so updates in third_party/ are immediately reflected.
