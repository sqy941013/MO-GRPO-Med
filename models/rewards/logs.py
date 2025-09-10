import json
import time
from typing import Any, Dict, List, Tuple

from .utils import get_logs_dir, read_text_file, write_all_generations, log_jsonl


def log_llm_call(
    call_name: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    raw_output: str,
    parsed_obj: Any,
    ok: bool,
    file_name: str = "llm_calls.jsonl",
) -> None:
    logs_dir = get_logs_dir()
    if logs_dir is None:
        return
    rec = {
        "t": time.time(),
        "call": call_name,
        "model": model_name,
        "ok": bool(ok),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_output": raw_output,
        "parsed": parsed_obj if isinstance(parsed_obj, (dict, list, str, int, float, bool)) else str(parsed_obj),
    }
    try:
        with open(logs_dir / file_name, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_reward_steps(sample_idx: int, gen_idx: int, user_text: str, raw_completion: str, step_info: Dict[str, Any], file_name: str = "reward_steps.jsonl") -> None:
    rec = {
        "t": time.time(),
        "sample_idx": int(sample_idx),
        "gen_idx": int(gen_idx),
        "user": user_text,
        "completion": raw_completion,
        "steps": step_info,
    }
    log_jsonl(file_name, rec)


def log_zero_rewards(flat_scores: List[float], prompts: List[List[Dict[str, str]]], completions: List[List[Dict[str, str]]], index_map: List[Tuple[int, int]], reward_name: str) -> None:
    logs_dir = get_logs_dir()
    if logs_dir is None:
        return
    path = logs_dir / f"{reward_name}_zeros.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as f:
            for flat_idx, sc in enumerate(flat_scores):
                try:
                    if float(sc) != 0.0:
                        continue
                except Exception:
                    continue
                if flat_idx >= len(index_map):
                    continue
                sample_idx, gen_idx = index_map[flat_idx]
                user_text = ""
                try:
                    if 0 <= sample_idx < len(prompts):
                        for m in prompts[sample_idx]:
                            if m.get("role") == "user":
                                user_text = m.get("content", "")
                                break
                except Exception:
                    pass
                completion_text = ""
                try:
                    if 0 <= sample_idx < len(completions):
                        comp_list = completions[sample_idx] or []
                        if 0 <= gen_idx < len(comp_list):
                            item = comp_list[gen_idx]
                            if isinstance(item, dict):
                                completion_text = str(item.get("content", ""))
                except Exception:
                    pass
                rec = {
                    "sample_idx": int(sample_idx),
                    "gen_idx": int(gen_idx),
                    "reward_name": reward_name,
                    "score": float(sc) if isinstance(sc, (int, float)) else 0.0,
                    "user": user_text,
                    "completion": completion_text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log_zero_rewards] failed to write logs for {reward_name}: {e}")


