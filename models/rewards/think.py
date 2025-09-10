import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from .utils import NINE_KEYS, extract_think_text, extract_post_think_answer
from .struct import extract_generated_map_via_llm, compute_final_r_struct
from .cover import extract_ie_from_generated_map, calculate_risk_weighted_f1, calculate_combined_precision, calculate_r_cover
from .medfact import evaluate_medfact_for_ie_results, calculate_r_medfact
from .style import evaluate_style_for_text, calculate_advanced_style_score
from .logs import log_reward_steps, log_zero_rewards
from .utils import write_all_generations


def compute_think_format_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    **kwargs,
) -> List[float]:
    groups: List[List[float]] = []
    import re as _re
    THINK_RE = _re.compile(r"<think>([\s\S]*?)</think>", _re.IGNORECASE)
    ANSWER_RE = _re.compile(r"<answer>([\s\S]*?)</answer>", _re.IGNORECASE)
    for comp_list in (completions or []):
        gen_scores: List[float] = []
        for item in (comp_list or []):
            try:
                content = item.get("content", "") if isinstance(item, dict) else ""
                think_text = THINK_RE.search(content or "")
                ans_text = ANSWER_RE.search(content or "")
                ok_think = bool(think_text and think_text.group(1).strip())
                ok_answer = bool(ans_text and ans_text.group(1).strip())
                gen_scores.append(1.0 if (ok_think and ok_answer) else 0.0)
            except Exception:
                gen_scores.append(0.0)
        groups.append(gen_scores or [0.0])

    out_scores: List[float] = []
    for gen_scores in groups:
        out_scores.append(float(sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0)

    try:
        write_all_generations(prompts, completions, file_name="all_generations.jsonl")
    except Exception:
        pass
    return out_scores


def compute_think_gated_total_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    note_id: List[str] = None,
    formated_source_note: List[str] = None,
    gold_map_obj: List[Dict[str, Any]] = None,
    t_source_anchors_obj: List[Dict[str, Any]] = None,
    t_gold_anchors_obj: List[Dict[str, Any]] = None,
    t_all_anchors_obj: List[Dict[str, Any]] = None,
    weights: Dict[str, float] | None = None,
    apply_gating: bool = True,
    critical_keys: List[str] | None = None,
    per_missing_penalty: float = 0.85,
    hard_kill_score: float = 0.02,
    **kwargs,
) -> List[float]:
    if weights is None:
        weights = {'struct': 0.15, 'cover': 0.35, 'medfact': 0.40, 'style': 0.10}
    if critical_keys is None:
        critical_keys = [
            "Medication changes",
            "Follow-up",
            "What to do after discharge",
            "Red flags (when to seek care)",
        ]

    target_k: int | None = None
    try:
        maybe_k = kwargs.get("num_generations", None)
        if isinstance(maybe_k, int) and maybe_k > 0:
            target_k = maybe_k
        if target_k is None:
            env_k = os.getenv("GRPO_NUM_GENERATIONS", "")
            if env_k.isdigit():
                target_k = int(env_k)
    except Exception:
        pass
    if target_k is None:
        try:
            lengths = []
            for comp_list in completions or []:
                lengths.append(len(comp_list) if isinstance(comp_list, list) else 0)
            if lengths:
                from collections import Counter as _Ctr
                target_k = _Ctr(lengths).most_common(1)[0][0] or 1
            else:
                target_k = 1
        except Exception:
            target_k = 2

    # 展平
    di_preds: List[str | None] = []
    index_map: List[Tuple[int, int]] = []
    # 使用 </think> 之后的最终输出作为答案文本
    for i, comp_list in enumerate(completions or []):
        lst = comp_list if isinstance(comp_list, list) else []
        if len(lst) >= target_k:
            norm = lst[:target_k]
        elif len(lst) == 0:
            norm = [{} for _ in range(target_k)]
        else:
            last = lst[-1]
            norm = lst + [last for _ in range(target_k - len(lst))]
        for j, item in enumerate(norm):
            content = item.get("content", "") if isinstance(item, dict) else ""
            di_text = extract_post_think_answer(content)
            di_preds.append(di_text if di_text else None)
            index_map.append((i, j))

    out_scores: List[float] = []
    for flat_idx, ans_text in enumerate(di_preds):
        sample_idx, gen_idx = index_map[flat_idx] if flat_idx < len(index_map) else (0, 0)
        raw_content = ""
        try:
            comp_list = completions[sample_idx] if 0 <= sample_idx < len(completions) else []
            if isinstance(comp_list, list) and 0 <= gen_idx < len(comp_list) and isinstance(comp_list[gen_idx], dict):
                raw_content = str(comp_list[gen_idx].get("content", ""))
        except Exception:
            raw_content = ""

        # Think 门控：先检查 <think> 段
        raw_think = ""
        try:
            raw_think = extract_think_text(raw_content) or ""
        except Exception:
            raw_think = ""
        think_len = len((raw_think or "").strip())
        # 阈值（可通过环境变量覆盖 THINK_SHORT_CHARS）
        short_thr = 64
        try:
            env_thr = os.getenv("THINK_SHORT_CHARS", "").strip()
            if env_thr.isdigit():
                short_thr = int(env_thr)
        except Exception:
            pass
        if think_len == 0:
            out_scores.append(0.1)
            continue
        if think_len <= short_thr:
            out_scores.append(0.2)
            continue

        # 没有答案文本也不给高分
        has_answer = bool(ans_text and str(ans_text).strip())
        if not has_answer:
            out_scores.append(0.1)
            continue

        try:
            step_info: Dict[str, Any] = {}
            gen_map = extract_generated_map_via_llm(str(ans_text), model=None, temperature=0.0)
            step_info["generated_map"] = gen_map
            ie_result = extract_ie_from_generated_map(gen_map, model=None, temperature=0.2)
            step_info["ie_result"] = ie_result

            gold_map = gold_map_obj[sample_idx] if gold_map_obj and sample_idx < len(gold_map_obj) else {k: "" for k in NINE_KEYS}
            r_struct = compute_final_r_struct(gen_map, gold_map, alpha=0.6, beta=0.4, L_min=10)
            step_info["r_struct"] = r_struct

            gold_true = t_gold_anchors_obj[sample_idx] if t_gold_anchors_obj and sample_idx < len(t_gold_anchors_obj) else {k: [] for k in NINE_KEYS}
            source_true = t_source_anchors_obj[sample_idx] if t_source_anchors_obj and sample_idx < len(t_source_anchors_obj) else {k: [] for k in NINE_KEYS}
            all_true = t_all_anchors_obj[sample_idx] if t_all_anchors_obj and sample_idx < len(t_all_anchors_obj) else {k: [] for k in NINE_KEYS}
            f1_gold, _ = calculate_risk_weighted_f1(gold_true, ie_result)
            f1_source, _ = calculate_risk_weighted_f1(source_true, ie_result)
            p_all = calculate_combined_precision(all_true, ie_result)
            r_cover = calculate_r_cover(f1_gold, f1_source, p_all, w_gold=0.7, w_source=0.3, lambda_penalty=0.2)
            step_info["r_cover_components"] = {"f1_gold": f1_gold, "f1_source": f1_source, "p_all": p_all}
            step_info["r_cover"] = r_cover

            row_dict = {
                "note_id": note_id[sample_idx] if note_id and sample_idx < len(note_id) else f"row_{sample_idx}",
                "formated_source_note": formated_source_note[sample_idx] if formated_source_note and sample_idx < len(formated_source_note) else "",
            }
            src_row = pd.Series(row_dict)
            medfact_eval = evaluate_medfact_for_ie_results(ie_results=ie_result, source_row=src_row)
            medfact_stats = calculate_r_medfact(medfact_eval)
            r_medfact = float(medfact_stats.get("r_medfact_score", 0.0))
            step_info["r_medfact_stats"] = medfact_stats
            has_major_unsafe = bool(medfact_stats.get("has_major_unsafe_flag", False))

            style_out = evaluate_style_for_text(str(ans_text), model=None, temperature=0.0)
            r_style = calculate_advanced_style_score(style_out.get("scores", {}))
            step_info["r_style_scores"] = style_out.get("scores", {})
            step_info["r_style"] = r_style

            total = (
                weights.get('struct', 0.0) * float(r_struct)
                + weights.get('cover', 0.0) * float(r_cover)
                + weights.get('medfact', 0.0) * float(r_medfact)
                + weights.get('style', 0.0) * float(r_style)
            )

            if apply_gating:
                miss_cnt = 0
                for sec in critical_keys:
                    gold_txt = str(gold_map.get(sec, "")).strip()
                    ie_list = ie_result.get(sec, [])
                    if gold_txt and (not ie_list):
                        miss_cnt += 1
                if miss_cnt > 0:
                    total *= (per_missing_penalty ** miss_cnt)
                if has_major_unsafe:
                    total = hard_kill_score

            out_scores.append(float(total))
            try:
                user_txt = ""
                if 0 <= sample_idx < len(prompts):
                    for m in prompts[sample_idx]:
                        if m.get("role") == "user":
                            user_txt = m.get("content", "")
                            break
                log_reward_steps(sample_idx, gen_idx, user_txt, raw_content, step_info)
            except Exception:
                pass
        except Exception as e:
            print(f"[think_gated_total] scoring failed sample_idx={sample_idx}, gen_idx={gen_idx}: {e}")
            out_scores.append(0.0)

    try:
        log_zero_rewards(out_scores, prompts, completions, index_map, reward_name="compute_think_gated_total_reward")
    except Exception:
        pass
    try:
        write_all_generations(prompts, completions, file_name="all_generations.jsonl")
    except Exception:
        pass
    return out_scores


