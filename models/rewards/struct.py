import json
import os
from typing import Any, Dict, List
from pathlib import Path

from .utils import NINE_KEYS, ensure_nine_keys_text, read_text_file, vprint, extract_post_think_answer
import time as _time
from .openai_io import chat_json_with_retry
from .cache import cache_lookup, cache_store
from .logs import log_llm_call


def _build_user_prompt(user_template: str, di_text: str) -> str:
    marker = "{PASTE THE DI_gold TEXT HERE}"
    return user_template.replace(marker, di_text)


def get_ds_v3_model() -> str:
    return os.getenv("REWARD_DS_V3_MODEL", "deepseek-v3-250324")


def extract_generated_map_via_llm(generated_text: str, model: str | None = None, temperature: float = 0.0) -> Dict[str, str]:
    try:
        if model is None:
            model = get_ds_v3_model()
        _t0 = _time.time()
        vprint("r_struct: start extracting 9-key map from generated text")
        proj_root = Path(__file__).resolve().parents[2]
        candidates = [
            proj_root / "docs" / "prompts" / "reward_function" / "r_struct",
            proj_root / "prompts" / "reward_function" / "r_struct",
        ]
        pdir = None
        for c in candidates:
            if (c / "system.txt").exists() and (c / "user.txt").exists():
                pdir = c
                break
        if pdir is None:
            raise FileNotFoundError("r_struct prompts not found in prompts/... or docs/prompts/...")

        system_prompt = read_text_file(pdir / "system.txt")
        user_template = read_text_file(pdir / "user.txt")
        user_prompt = _build_user_prompt(user_template, generated_text)

        vprint("r_struct: prompts loaded, checking cache for generated_map")
        key = _sha1_text(f"STRUCT_MAP::{model}::{generated_text}")
        _tc = _time.time()
        cached = cache_lookup("r_struct_extract_map", key, model)
        vprint(f"r_struct: cache lookup took {(_time.time()-_tc):.3f}s")
        if cached is not None and isinstance(cached, dict):
            vprint("r_struct: hit cache for generated_map")
            vprint(f"r_struct: total took {(_time.time()-_t0):.3f}s")
            return ensure_nine_keys_text(cached)

        vprint("r_struct: cache miss, calling LLM to extract map")
        _tllm = _time.time()
        obj, _raw, ok = chat_json_with_retry(
            model=str(model),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            want_raw=False,
            retries=4,
        )
        vprint(f"r_struct: llm call took {(_time.time()-_tllm):.3f}s")
        try:
            log_llm_call(
                call_name="r_struct_extract_map",
                model_name=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=_raw or "",
                parsed_obj=obj if isinstance(obj, dict) else {},
                ok=ok,
            )
        except Exception:
            pass
        if ok and isinstance(obj, dict):
            try:
                vprint("r_struct: LLM extracted map ok, storing cache")
                _ts = _time.time()
                cache_store("r_struct_extract_map", key, model, obj)
                vprint(f"r_struct: cache store took {(_time.time()-_ts):.3f}s")
            except Exception:
                pass
        else:
            print("[r_struct] JSON extract failed after retries.")
        _tout = ensure_nine_keys_text(obj if isinstance(obj, dict) else {})
        vprint(f"r_struct: total took {(_time.time()-_t0):.3f}s")
        return _tout
    except Exception as e:
        print(f"[r_struct] JSON extract failed: {e}")
        return {k: "" for k in NINE_KEYS}


def _sha1_text(text: str) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def compute_c_sec(pred_map: Dict[str, str], gold_map: Dict[str, str] | None = None, L_min: int = 8) -> float:
    PLACEHOLDERS = {"", "n/a", "na", "none", "___", "-"}
    weights = {
        "Medication changes": 0.30,
        "Follow-up": 0.25,
        "What to do after discharge": 0.20,
        "Red flags (when to seek care)": 0.15,
        "Why admitted": 0.03,
        "What happened in hospital": 0.03,
        "Lifestyle / Activity": 0.02,
        "Wound / Device care": 0.01,
        "Contacts / Emergency": 0.01,
    }

    total_weighted_score = 0.0
    total_weight = 0.0

    for section in NINE_KEYS:
        pred_text = (pred_map.get(section, "") or "").strip().lower()
        if gold_map is not None:
            gold_text = (gold_map.get(section, "") or "").strip()
            if not gold_text or gold_text.lower() in PLACEHOLDERS:
                continue
        present_score = 0.0 if (pred_text in PLACEHOLDERS or len(pred_text) < L_min) else 1.0
        w = weights.get(section, 0.0)
        total_weighted_score += w * present_score
        total_weight += w

    if total_weight == 0:
        return 0.0
    return total_weighted_score / total_weight


def compute_A_sec_gold(generated_map: Dict[str, str], gold_standard_map: Dict[str, str], L_min: int = 5) -> float:
    PLACEHOLDERS = {"", "n/a", "none", "not applicable", "null"}
    Y_plus, G_plus = set(), set()

    for section, content in generated_map.items():
        if content and str(content).strip():
            s = str(content).strip().lower()
            if s not in PLACEHOLDERS and len(s) >= L_min:
                Y_plus.add(section)

    for section, content in gold_standard_map.items():
        if content and str(content).strip():
            s = str(content).strip().lower()
            if s not in PLACEHOLDERS and len(s) >= L_min:
                G_plus.add(section)

    union_size = len(Y_plus.union(G_plus))
    if union_size == 0:
        return 1.0
    intersection_size = len(Y_plus.intersection(G_plus))
    return intersection_size / union_size


def compute_final_r_struct(generated_map: Dict[str, str], gold_standard_map: Dict[str, str], alpha: float = 0.6, beta: float = 0.4, L_min: int = 10) -> float:
    C_sec = compute_c_sec(generated_map, gold_standard_map, L_min)
    A_sec_gold = compute_A_sec_gold(generated_map, gold_standard_map, L_min)
    return alpha * C_sec + beta * A_sec_gold


def compute_r_struct_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    note_id: List[str] = None,
    gold_map_obj: List[Dict[str, Any]] = None,
    **kwargs,
) -> List[float]:
    # Return one score per prompt (averaged across generations)
    groups: List[List[str]] = []
    for comp_list in (completions or []):
        grp: List[str] = []
        for item in (comp_list or []):
            grp.append(str(item.get("content", "")) if isinstance(item, dict) else "")
        groups.append(grp)

    out_scores: List[float] = []
    for i, grp in enumerate(groups):
        gen_scores: List[float] = []
        for content in grp:
            # 提取 </think> 之后的最终输出内容
            di_text = extract_post_think_answer(content)
            if not di_text:
                gen_scores.append(0.0)
                continue
            try:
                gen_map = extract_generated_map_via_llm(di_text, model=None, temperature=0.0)
                gold_map = gold_map_obj[i] if gold_map_obj and i < len(gold_map_obj) else {k: "" for k in NINE_KEYS}
                score = compute_final_r_struct(gen_map, gold_map, alpha=0.6, beta=0.4, L_min=10)
                gen_scores.append(float(score))
            except Exception as e:
                print(f"[r_struct] scoring failed sample_idx={i}: {e}")
                gen_scores.append(0.0)
        final_score = float(sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
        out_scores.append(final_score)
    return out_scores


