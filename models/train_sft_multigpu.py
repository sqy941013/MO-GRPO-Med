#!/usr/bin/env python3
# models/train_sft_multigpu.py
import argparse, datetime, os, pandas as pd, torch
from pathlib import Path

# Import Unsloth FIRST — no PatchSFTTrainer needed anymore
from unsloth import FastLanguageModel, unsloth_train      # auto-patches TRL internally
# ─────────────────────────────────────────────────────────────────────────────

from transformers import TrainingArguments
from transformers import Trainer as HFTrainer
from transformers import set_seed
from datasets import Dataset, DatasetDict, load_dataset
from trl import SFTTrainer, SFTConfig
from loguru import logger

SYSTEM_PROMPT = (
    "You are a clinical language model that writes clear, safe, "
    "patient-friendly discharge instructions."
)

# 7 clinical sections
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

# ----------------- Monkey patch HF Trainer to avoid in-place loss scaling ----
def _patch_trainer_compute_loss_no_inplace():
    def _compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        if self.label_smoother is not None and "labels" in inputs:
            loss = self.label_smoother(outputs, inputs["labels"])
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss = loss.clone()
        try:
            num_processes = getattr(self.accelerator, "num_processes", 1) or 1
        except Exception:
            num_processes = 1
        if num_processes > 1:
            loss = loss * num_processes
        return (loss, outputs) if return_outputs else loss

    HFTrainer.compute_loss = _compute_loss

_patch_trainer_compute_loss_no_inplace()

# ----------------- Safe trainer to avoid inplace on fused loss --------------
class SafeSFTTrainer(SFTTrainer):
    """Overrides compute_loss to avoid in-place ops on Unsloth fused loss views.

    - Clones the loss tensor before any scaling
    - Uses out-of-place scaling for multi-process runs
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        if self.label_smoother is not None and "labels" in inputs:
            loss = self.label_smoother(outputs, inputs["labels"])
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        # Avoid in-place modification on a potential view returned by Unsloth
        loss = loss.clone()

        # Match HF behavior without in-place multiply for DDP
        try:
            num_processes = getattr(self.accelerator, "num_processes", 1) or 1
        except Exception:
            num_processes = 1
        if num_processes > 1:
            loss = loss * num_processes

        return (loss, outputs) if return_outputs else loss

# ----------------- CLI -------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("SFT trainer (Unsloth)")
    p.add_argument("--task", choices=["DI", "BCH"], default="DI")
    p.add_argument("--data_dir", default="datasets/processed")
    p.add_argument("--model_name",
                   default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
    p.add_argument("--max_seq_len", type=int, default=8192)
    p.add_argument("--lora_r",      type=int, default=64)
    p.add_argument("--lora_alpha",  type=int, default=64)
    p.add_argument("--target_modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--grad_accum",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--epochs",      type=int, default=2)
    p.add_argument("--max_steps",   type=int, default=None, help="Max training steps (overrides epochs)")
    p.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--optimizer",   default="adamw_8bit", help="Optimizer type")
    p.add_argument("--lr_scheduler", default="linear", help="Learning rate scheduler")
    p.add_argument("--logging_steps",     type=int, default=50)
    p.add_argument("--eval_steps",  type=int, default=None, help="Evaluation steps")
    p.add_argument("--save_strategy", default="steps", choices=["no", "steps", "epoch"], help="Save strategy")
    p.add_argument("--save_steps",  type=int, default=500, help="Save steps if save_strategy is steps")
    p.add_argument("--output_root", default="checkpoints/sft")
    p.add_argument("--save_format", default="lora", choices=["lora", "merged_16bit", "merged_4bit", "gguf"], 
                   help="Model save format")
    p.add_argument("--use_gradient_checkpointing", default="unsloth", help="Gradient checkpointing method")
    p.add_argument("--random_state", type=int, default=3407, help="Random seed")
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path")
    p.add_argument("--testrun", action="store_true", help="Use a tiny subset (100 samples) for quick test runs")
    return p.parse_args()

# ----------------- prompt builder -------------------------------------------
def row_to_chat(row: pd.Series):
    blocks = []
    for idx, sec in enumerate(SEC_ORDER, start=1):
        lines = [f"{col}: {row[col]}" for col in SECTION_MAP[sec]
                 if col in row and pd.notna(row[col]) and row[col] != ""]
        text = "\n".join(lines) if lines else "N/A"
        blocks.append(f"[SEC_{idx}] {text}")

    user_msg = (
        "\n\n".join(blocks) +
        "\n\nPlease generate a Discharge Instructions for the patient based on the information from the above 7 sections."
    )
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": row["discharge_instructions"]},
    ]

# ---------------- dataset builder (memory-efficient) -------------------
def build_dataset(csv_path: Path, tokenizer: object, is_train: bool, *, limit_samples: int | None = None):
    """
    Build dataset and preprocess in a memory-efficient way.
    """
    logger.info(f"📊 Loading and processing {csv_path}...")
    
    # Define a processing function for .map()
    def format_for_sft(batch):
        # batch is a dict of lists (e.g., {'note_id': [...], 'subject_id': [...]})
        # Reconstruct rows to fit row_to_chat
        reconstructed_rows = []
        num_rows = len(next(iter(batch.values())))
        for i in range(num_rows):
            row_dict = {key: values[i] for key, values in batch.items()}
            reconstructed_rows.append(pd.Series(row_dict))

        # Apply original chat formatting logic
        formatted_texts = []
        for row in reconstructed_rows:
            chat = row_to_chat(row)
            formatted_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        
        return {"text": formatted_texts}

    # Load from CSV with datasets.load_dataset to save memory
    raw_dataset = load_dataset("csv", data_files=str(csv_path), split="train", keep_in_memory=False)

    # Optionally limit dataset size for quick test runs
    if limit_samples is not None and limit_samples > 0:
        raw_dataset = raw_dataset.select(range(min(limit_samples, len(raw_dataset))))
    
    # Use .map() with batching to avoid loading all data into memory
    processed_dataset = raw_dataset.map(
        format_for_sft,
        batched=True,
        batch_size=1000, # process 1000 rows per batch
        remove_columns=raw_dataset.column_names, # keep only the "text" column
        num_proc=max(1, os.cpu_count() // 2), # use multiple CPU cores
    )
    
    logger.info(f"✅ Dataset preprocessing completed for {csv_path}")
    return processed_dataset

# ----------------- memory monitoring -------------------------------------
def show_memory_stats(stage=""):
    """Show GPU memory stats"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        memory_used = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        memory_percent = round(memory_used / max_memory * 100, 3)
        
        print(f"🖥️  GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"🖥️  {stage} Memory reserved = {memory_used} GB ({memory_percent}%).")
        return memory_used
    return 0

# ----------------- model saving utilities -------------------------------
def save_model(model, tokenizer, args, ckpt_dir):
    """Save model according to requested format"""
    print(f"💾 Saving model in {args.save_format} format to: {ckpt_dir}")
    
    if args.save_format == "lora":
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print("✅ LoRA adapters saved")
        
    elif args.save_format == "merged_16bit":
        model.save_pretrained_merged(ckpt_dir, tokenizer, save_method="merged_16bit")
        print("✅ Merged 16bit model saved")
        
    elif args.save_format == "merged_4bit":
        model.save_pretrained_merged(ckpt_dir, tokenizer, save_method="merged_4bit")
        print("✅ Merged 4bit model saved")
        
    elif args.save_format == "gguf":
        model.save_pretrained_gguf(ckpt_dir, tokenizer, quantization_method="q8_0")
        print("✅ GGUF model saved")

# ----------------- main ------------------------------------------------------
def main():
    args = get_args()
    # Ensure deterministic init across ranks
    try:
        set_seed(args.random_state)
    except Exception:
        pass
    
    print("🚀 Starting SFT training with improved configuration")
    print(f"📝 Task: {args.task}")
    print(f"🤖 Model: {args.model_name}")
    print(f"💾 Save format: {args.save_format}")

    train_csv = Path(args.data_dir) / f"{args.task}_train_sft.csv"
    dev_csv   = Path(args.data_dir) / f"{args.task}_dev.csv"

    # Check if files exist
    if not train_csv.exists():
        raise FileNotFoundError(f"Training file not found: {train_csv}")
    if not dev_csv.exists():
        raise FileNotFoundError(f"Development file not found: {dev_csv}")

    start_memory = show_memory_stats("Initial")

    # In multi-GPU runs, let Accelerate/DDP handle device placement; avoid custom maps that cause DDP inconsistencies
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass

    # 1) load model & LoRA
    print("🔧 Loading model and setting up LoRA...")
    # For 4bit/8bit quantized weights, place on target device at load time
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    device_map = {"": current_device}

    model, tok = FastLanguageModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        # fast_inference=True, # Enable vLLM fast inference
        trust_remote_code=True,
        device_map=device_map
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=False,   # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    # Log parameter counts per rank for debugging DDP consistency
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[rank{os.environ.get('LOCAL_RANK','0')}] Params total={total_params}, trainable={trainable_params}")
    except Exception:
        pass

    # 2) datasets
    subset_limit = 100 if args.testrun else None
    train_ds = build_dataset(train_csv, tok, is_train=True,  limit_samples=subset_limit)
    dev_ds   = build_dataset(dev_csv,   tok, is_train=False, limit_samples=subset_limit)

    model_memory = show_memory_stats("Model loaded")

    # 3) trainer with SFTConfig
    print("🏋️  Setting up trainer...")
    
    # Build output directory
    slug  = args.model_name.split("/")[-1].replace(".", "_").replace("-", "_")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    ckpt_dir = Path(args.output_root) / f"{args.task}_{slug}_{stamp}"
    
    # Ensure max_steps has an integer default (-1) to avoid Trainer conflicts
    max_steps = args.max_steps if args.max_steps is not None else -1

    training_config = SFTConfig(
        # Basics
        output_dir=str(ckpt_dir),
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        
        # Training args
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        
        # Optimizer & scheduler
        optim=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        
        # Logging & saving
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        
        # Performance
        # bf16=True,
        dataloader_pin_memory=False,
        group_by_length=True,
        
        # Misc
        seed=args.random_state,
        report_to="wandb",
        ddp_find_unused_parameters = False
    )
    
    trainer = SafeSFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=training_config,
    )

    # Ensure Unsloth's wrapper uses a safe non-inplace compute_loss
    try:
        import types

        def _safe_old_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            if self.label_smoother is not None and "labels" in inputs:
                loss = self.label_smoother(outputs, inputs["labels"])
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss = loss.clone()
            try:
                num_processes = getattr(self.accelerator, "num_processes", 1) or 1
            except Exception:
                num_processes = 1
            if num_processes > 1:
                loss = loss * num_processes
            return (loss, outputs) if return_outputs else loss

        trainer._old_compute_loss = types.MethodType(_safe_old_compute_loss, trainer)
    except Exception:
        pass

    # 4) Train
    print("🎯 Starting training...")
    
    try:
        if args.resume_from_checkpoint:
            print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer_stats = unsloth_train(trainer, resume_from_checkpoint=args.resume_from_checkpoint)
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

    # 5) Show training stats
    training_memory = show_memory_stats("Training completed")
    
    if hasattr(trainer_stats, 'metrics'):
        runtime = trainer_stats.metrics.get('train_runtime', 0)
        print(f"⏱️  Training time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        
        if start_memory > 0:
            training_memory_used = round(training_memory - start_memory, 3)
            max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)
            training_percent = round(training_memory_used / max_memory * 100, 3)
            print(f"🖥️  Peak training memory = {training_memory_used} GB ({training_percent}%)")

    # 6) Save model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        save_model(model, tok, args, ckpt_dir)
        
    # Save training args
        import json
        config_path = ckpt_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"📋 Training config saved to: {config_path}")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
    # Try at least saving LoRA
        try:
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print("✅ Fallback: LoRA adapters saved")
        except Exception as e2:
            print(f"❌ Fallback save also failed: {e2}")
        raise

    print(f"🎉 Training completed successfully!")
    print(f"📁 Model saved to: {ckpt_dir}")
    
    return ckpt_dir

if __name__ == "__main__":
    main()
