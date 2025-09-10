# from models.rewards.aggregate import compute_total_reward
from unsloth import FastLanguageModel
import torch
from datasets import Dataset as HFDataset
import pandas as pd
import numpy as np
import os, datetime, json
from pathlib import Path
from typing import List, Dict, Any
import argparse
import sys as _sys
import time
from contextlib import contextmanager
from collections import defaultdict
try:
    from transformers import TrainerCallback
except Exception:  # Should be available in training env; otherwise fall back to a dummy class
    class TrainerCallback:  # type: ignore
        pass

# Allow importing `models.*` when running as a script
_PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJ_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJ_ROOT))

# Parse CLI flags and set env vars before importing reward_functions
def _setup_reward_model_env():
    """Set environment variables before importing reward modules."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_ds_v3_model", type=str, default=os.getenv("REWARD_DS_V3_MODEL", "deepseek-v3-250324"))
    parser.add_argument("--reward_ds_r1_model", type=str, default=os.getenv("REWARD_DS_R1_MODEL", "deepseek-r1-250528"))
    # Parse only known args; ignore others so upstream scripts can pass extra flags
    args, _ = parser.parse_known_args()
    
    # Export as env vars for downstream reward modules
    if args.reward_ds_v3_model:
        os.environ["REWARD_DS_V3_MODEL"] = args.reward_ds_v3_model
    if args.reward_ds_r1_model:
        os.environ["REWARD_DS_R1_MODEL"] = args.reward_ds_r1_model
    
    print(f"[env_setup] REWARD_DS_V3_MODEL = {os.environ.get('REWARD_DS_V3_MODEL')}")
    print(f"[env_setup] REWARD_DS_R1_MODEL = {os.environ.get('REWARD_DS_R1_MODEL')}")

# Apply env vars
_setup_reward_model_env()

# Import reward functions after env setup
from models.reward_functions import (
    compute_r_struct_reward,
    compute_r_cover_reward,
    compute_r_medfact_reward,
    compute_r_style_reward,
    compute_total_reward
)
from trl import GRPOConfig, GRPOTrainer
# from models.expectile_huber_trainer import ExpectileHuberGRPOTrainer
from vllm import SamplingParams

# Define section mapping and order locally (do not import from train_sft)
SECTION_MAP = {
    "basic": [
        "note_id","subject_id","hadm_id","age_at_charttime","sex","race"
    ],
    "history": [
        "chief_complaint","history_present_illness","past_medical_history",
        "social_family_history","allergies","medications_admission"
    ],
    "admission": [
        "admission_type","admission_location","arrival_transport",
        "service","insurance"
    ],
    "diagnostic": [
        "physical_examination","major_procedures",
        "acute_issues","chronic_issues","pertinent_results"
    ],
    "course": ["brief_hospital_course"],
    "dx":     ["discharge_diagnosis"],
    "discharge_info": [
        "discharge_medications","discharge_condition","discharge_disposition"
    ],
}
SEC_ORDER = [
    "basic","history","admission",
    "diagnostic","course","dx","discharge_info"
]

# --- Simple timing utilities ---
_TIMES = defaultdict(float)
_COUNTS = defaultdict(int)

@contextmanager
def _timeit(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _TIMES[label] += dt
        _COUNTS[label] += 1
        print(f"[TIME] {label}: {dt:.3f}s", flush=True)

def _print_time_summary(top_n: int = 20) -> None:
    try:
        if not _TIMES:
            return
        items = sorted(_TIMES.items(), key=lambda kv: kv[1], reverse=True)
        print("[TIME] summary (total, count, mean):", flush=True)
        for i, (label, total) in enumerate(items[:top_n]):
            cnt = _COUNTS[label]
            mean = total / max(cnt, 1)
            print(f"[TIME]   {label}: total={total:.3f}s count={cnt} mean={mean:.3f}s", flush=True)
    except Exception:
        pass


class StepTimerCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()
        self._t0 = None
        self._last_step = -1

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[no-redef]
        try:
            self._t0 = time.perf_counter()
            self._last_step = getattr(state, "global_step", -1)
        except Exception:
            self._t0 = None

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-redef]
        try:
            if self._t0 is None:
                return
            dt = time.perf_counter() - self._t0
            step_id = getattr(state, "global_step", -1)
            label = "train_step"
            _TIMES[label] += dt
            _COUNTS[label] += 1
            print(f"[TIME] step {step_id} (prev {self._last_step}): {dt:.3f}s", flush=True)
        except Exception:
            pass

max_seq_length = 8192  # Increase if you need a longer reasoning path
lora_rank = 16  # Larger is often better but slower. Suggested: 8, 16, 32, 64, 128

def init_reasoning_model(model_name: str, init_lora_path: str | None = None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
    load_in_4bit = True,  # Set False for LoRA 16-bit; True loads 4-bit to save VRAM
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,  # Reduce this if you run out of VRAM
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2,  # *2 speeds up training a bit
        use_gradient_checkpointing = "unsloth",  # Reduce VRAM usage
        random_state = 3407,
    )

    # Optionally load a pre-trained LoRA adapter
    if init_lora_path:
        try:
            model.load_lora(init_lora_path)
            print(f"✅ Initialized LoRA from: {init_lora_path}")
        except Exception as e:
            print(f"[Warning] Failed to load initial LoRA from {init_lora_path}: {e}")

    # Build chat template
    reasoning_start = "<think>"  # Acts as <think>
    reasoning_end   = "</think>"  # Acts as </think>
    solution_start  = "<answer>"
    solution_end    = "</answer>"

    system_prompt = \
    f"""You are a clinical language model that writes clear, safe, patient-friendly Discharge Instructions (DI) from source clinical documentation, in a hospital letter-style.
Think about the writing process and provide your thinking process.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your discharge instructions between {solution_start}{solution_end}"""

    chat_template = (
        r"{% if messages[0]['role'] == 'system' %}"
        r"{{ messages[0]['content'] + eos_token }}"
        r"{% set loop_messages = messages[1:] %}"
        r"{% else %}"
        r"{{ '{system_prompt}' + eos_token }}"
        r"{% set loop_messages = messages %}"
        r"{% endif %}"
        r"{% for message in loop_messages %}"
        r"{% if message['role'] == 'user' %}"
        r"{{ message['content'] }}"
        r"{% elif message['role'] == 'assistant' %}"
        r"{{ message['content'] + eos_token }}"
        r"{% endif %}"
        r"{% endfor %}"
        r"{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
        r"{% endif %}"
    )

    # Replace with our specific template:
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template

    # Simple template rendering check
    tokenizer.apply_chat_template([
        {"role" : "user", "content" : "What is 1+1?"},
        {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role" : "user", "content" : "What is 2+2?"},
    ], tokenize = False, add_generation_prompt = True)

    return model, tokenizer


GRPO_CSV_PATH = "data/processed/mo-grpo-med_dataset.csv"

# 9-section keys (must match reward functions)
NINE_KEYS: List[str] = [
    "Why admitted",
    "What happened in hospital",
    "What to do after discharge",
    "Medication changes",
    "Follow-up",
    "Red flags (when to seek care)",
    "Lifestyle / Activity",
    "Wound / Device care",
    "Contacts / Emergency",
]


def load_grpo_dataframe(csv_path: str = GRPO_CSV_PATH) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"GRPO CSV not found: {csv_path}. Please generate it first.")
    df = pd.read_csv(csv_path)
    required_cols = [
        "note_id", "formated_source_note", "gold_di",
        "gold_map", "t_source_anchors", "t_gold_anchors"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"GRPO CSV missing required columns: {missing}")
    return df

# Note: compute_r_struct_reward is defined in models/reward_functions.py

# If executed as a script, build dataset here
def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MO-GRPO-Med GRPO Trainer")
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit")
    p.add_argument("--init_lora_path", type=str, default=None, help="Path to initial LoRA adapter to load")
    p.add_argument("--csv_path", type=str, default=GRPO_CSV_PATH)
    p.add_argument("--limit_samples", type=int, default=5000, help="Limit number of rows to load from CSV (0=load all)")
    p.add_argument("--max_seq_length", type=int, default=max_seq_length, help="Model context window (tokens)")
    p.add_argument("--max_steps", type=int, default=250, help="Maximum number of training steps")
    # Expectile/Huber toggles and hyperparameters
    p.add_argument("--use_expectile_baseline", action="store_true", help="Use τ-expectile baseline instead of mean.")
    p.add_argument("--expectile_tau", type=float, default=0.7, help="Tau parameter for expectile baseline calculation.")
    p.add_argument("--normalize_by_std", action="store_true", help="Divide by group std (z-score).")
    # p.add_argument("--use_huber_advantage", action="store_true", help="Apply Huber clipping on advantages.")
    # Advantage delta (Huberized advantages) controls for Unsloth GRPO loss
    p.add_argument("--use_advantage_delta", action="store_true", help="Enable advantage clamp via advantage_delta in GRPO loss.")
    p.add_argument("--advantage_delta", type=float, default=0.65, help="Clamp magnitude for advantages when enabled.")
    p.add_argument("--tau", type=float, default=0.7, help="Legacy tau parameter (use --expectile_tau instead)")
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--eps_norm", type=float, default=1e-6)
    # Reward model names (LLM used inside reward computations)
    p.add_argument("--reward_ds_v3_model", type=str, default=os.getenv("REWARD_DS_V3_MODEL", "deepseek-v3-250324"), help="Model name for DeepSeek-V3 style extraction (struct/cover/medfact)")
    p.add_argument("--reward_ds_r1_model", type=str, default=os.getenv("REWARD_DS_R1_MODEL", "deepseek-r1-250528"), help="Model name for DeepSeek-R1 style scoring")
    # Wandb run name
    p.add_argument("--run_name", type=str, default=None, help="Custom name for wandb run. If None, auto-generated based on parameters.")
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = _get_args()

        # Apply configurable max_seq_length before model init
        try:
            max_seq_length = int(args.max_seq_length)
        except Exception:
            max_seq_length = 8192

        # 1) Load processed GRPO CSV
        grpo_df = load_grpo_dataframe(args.csv_path)
        print(f"✅ Loaded GRPO CSV: {args.csv_path} | rows={len(grpo_df)}")
        if isinstance(args.limit_samples, int) and args.limit_samples > 0 and len(grpo_df) > args.limit_samples:
            # Use random downsampling for diversity; set random_state for reproducibility if needed
            grpo_df = grpo_df.sample(n=args.limit_samples, replace=False, random_state=3407).reset_index(drop=True)
            print(f"🔎 Using a subset of {len(grpo_df)} rows (limit_samples={args.limit_samples})")

        # 2) Build HF Dataset and map to prompt/answer
        def _map_row(ex):
            return {
                "prompt": [
                    {"role": "system", "content": ""},  # placeholder; will fill system_prompt later
                    {"role": "user",   "content": ex.get("formated_source_note", "")},
                ],
                "answer": ex.get("gold_di", ""),
                "note_id": ex.get("note_id", ""),
            }

        with _timeit("dataset.build_hf_dataset"):
            ds = HFDataset.from_pandas(grpo_df, preserve_index=False)
        with _timeit("dataset.map_prompt_answer"):
            ds = ds.map(_map_row)

        # Inject system_prompt into prompt[0]
        def _inject_system(ex):
            p = ex["prompt"]
            if isinstance(p, list) and len(p) >= 1 and isinstance(p[0], dict):
                p[0]["content"] = (
                    f"""You are given a formatted clinical note for writing discharge instructions.
Think about the writing process and provide your thinking process.
Place it between <think> </think>.
Then, provide your discharge instructions between <answer> </answer>"""
                )
            return {"prompt": p}

        with _timeit("dataset.inject_system_prompt"):
            ds = ds.map(_inject_system)

        # Optionally parse reward-related fields (so downstream code can use objects instead of raw strings)
        def _parse_rewards(ex):
            def parse_dict(v):
                if isinstance(v, dict):
                    return v
                if isinstance(v, str):
                    s = v.strip()
                    if not s:
                        return {}
                    try:
                        obj = json.loads(s)
                        return obj if isinstance(obj, dict) else {}
                    except Exception:
                        return {}
                return {}
            return {
                "gold_map_obj": parse_dict(ex.get("gold_map", "")),
                "t_source_anchors_obj": parse_dict(ex.get("t_source_anchors", "")),
                "t_gold_anchors_obj": parse_dict(ex.get("t_gold_anchors", "")),
                "t_all_anchors_obj": parse_dict(ex.get("t_all_anchors", "")),
            }

        with _timeit("dataset.parse_reward_fields"):
            ds = ds.map(_parse_rewards)

        # Brief preview
        try:
            print({k: ds[0][k] for k in ["note_id", "prompt", "answer", "gold_map_obj", "t_source_anchors_obj", "t_gold_anchors_obj", "t_all_anchors_obj"] if k in ds.column_names})
        except Exception:
            pass

        # 3) Initialize model and tokenizer
        with _timeit("model.init_reasoning_model"):
            model, tokenizer = init_reasoning_model(args.model_name, args.init_lora_path)
        print("✅ Dataset and model are ready for GRPO training.")

        # 4) Tokenize, estimate lengths, and filter out long samples
        def _tokenize_map(batch):
            toks = [
                tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=True)
                for p in batch["prompt"]
            ]
            return {"tokens": toks}

        with _timeit("tokenize.dataset"):
            tokenized = ds.map(_tokenize_map, batched=True)
            tokenized = tokenized.map(lambda ex: {"L": len(ex["tokens"])})
        try:
            maximum_length = int(np.quantile(tokenized["L"], 0.9))
        except Exception:
            maximum_length = min(int(max_seq_length * 0.5), 4096)
        print("Max Length = ", maximum_length)

        import numpy as _np
        with _timeit("dataset.filter_by_length"):
            _idx = _np.where(_np.array(tokenized["L"]) <= maximum_length)[0]
            ds = ds.select(_idx.tolist())
            del tokenized

        # 5) Sampling and training config
        max_prompt_length = maximum_length + 1  # +1 as a small guard band
        max_completion_length = max_seq_length - max_prompt_length

        vllm_sampling_params = SamplingParams(
            min_p=0.1,
            top_p=1.0,
            top_k=-1,
            seed=3407,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=True,
        )

        # Build output directory: checkpoints/grpo/<model>_<timestamp>
        base_name = Path(getattr(getattr(model, "config", None), "_name_or_path", "model")).name
        run_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = str(Path("checkpoints") / "grpo" / f"{base_name}_{run_tag}")

        # Environment variables already set at module import time

        # Generate run name if not provided
        if args.run_name is None:
            model_short = args.model_name.split('/')[-1].replace('-unsloth-bnb-4bit', '')
            run_name = f"grpo-{model_short}"
            if args.use_expectile_baseline:
                run_name += f"-expectile{args.expectile_tau}"
            else:
                run_name += "-baseline"
            if args.use_advantage_delta:
                run_name += f"-huber{args.advantage_delta}"
            else:
                run_name += "-nohuber"
            run_name += f"-{args.max_steps}steps"
        else:
            run_name = args.run_name

        training_args = GRPOConfig(
            vllm_sampling_params=vllm_sampling_params,
            temperature=1.0,
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=2,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=args.max_steps,
            save_steps=25,
            save_total_limit=10,
            report_to="wandb",
            run_name=run_name,
            output_dir=output_dir,
        )

        # Set expectile baseline and tau
        training_args.use_expectile_baseline = args.use_expectile_baseline
        training_args.expectile_tau = args.expectile_tau

        # Set Huber advantage clipping parameters
        training_args.use_advantage_delta = args.use_advantage_delta
        training_args.advantage_delta = args.advantage_delta

        # Debug: print expectile and huber settings
        print(f"[config_debug] Expectile baseline: use_expectile_baseline={args.use_expectile_baseline}")
        print(f"[config_debug] Expectile tau: expectile_tau={args.expectile_tau}")
        print(f"[config_debug] Huber advantage clipping: use_advantage_delta={args.use_advantage_delta}")
        print(f"[config_debug] Advantage delta: advantage_delta={args.advantage_delta}")
        print(f"[config_debug] Wandb run name: run_name='{run_name}'")
        print(f"[config_debug] Args run_name: args.run_name='{args.run_name}'")

        # 6) Initialize training monitor (optional)
        try:
            from models.training_monitor import create_monitor, get_global_monitor
        except Exception:
            def create_monitor(output_dir: str, log_interval: int = 5):
                return None
            def get_global_monitor():
                return None
        monitor = create_monitor(output_dir, log_interval=5)
        print(f"[monitor] Training monitor initialized, logs will be saved to: {output_dir}/logs/")

        # 7) Build GRPOTrainer and start training
        # Expose output_dir for reward functions to write logs
        os.environ["GRPO_OUTPUT_DIR"] = output_dir
        # Enable streaming for reward evaluation (more robust with long outputs/weak networks)
        os.environ.setdefault("LLM_STREAM", "1")

        with _timeit("trainer.build"):
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[
                    # compute_r_struct_reward,
                    # compute_r_cover_reward,
                    # compute_r_medfact_reward,
                    # compute_r_style_reward,
                    compute_total_reward
                ],
                args=training_args,
                train_dataset=ds,
                tau=args.tau,
                delta=args.delta,
                eps_norm=args.eps_norm,
            )

        print("🎯 Starting GRPO training...")
        try:
            if hasattr(trainer, "add_callback"):
                trainer.add_callback(StepTimerCallback())
        except Exception:
            pass

        with _timeit("trainer.train"):
            trainer.train()

        # Finalize monitor and write summary stats
        try:
            monitor_global = get_global_monitor()
            if monitor_global:
                monitor_global.finalize()
                print(f"[monitor] Training metrics saved to: {monitor_global.logs_dir}")
        except Exception as monitor_error:
            print(f"[monitor] Error finalizing monitor: {monitor_error}")

        _print_time_summary()
    except Exception as e:
        print(f"[Warning] Failed to prepare GRPO training setup: {e}")

