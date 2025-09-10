from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import time as _time

import pandas as pd

from .utils import NINE_KEYS
from .struct import extract_generated_map_via_llm, compute_final_r_struct
from .cover import extract_ie_from_generated_map, calculate_risk_weighted_f1, calculate_combined_precision, calculate_r_cover
from .medfact import evaluate_medfact_for_ie_results, calculate_r_medfact
from .style import evaluate_style_for_text, calculate_advanced_style_score
from .utils import extract_post_think_answer, vprint


def compute_total_reward(
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
    apply_gating: bool = False,
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

    groups: List[List[str]] = []
    for comp_list in (completions or []):
        grp: List[str] = []
        for item in (comp_list or []):
            grp.append(str(item.get("content", "")) if isinstance(item, dict) else "")
        groups.append(grp)

    # Debug: print the size of each generation group
    try:
        sizes = [len(g) for g in groups]
        vprint(f"[aggregate] groups={len(groups)} grp_sizes={sizes}")
    except Exception:
        pass

    def _score_one_generation(i: int, gen_idx: int, content: str) -> float:
        di_text = extract_post_think_answer(content)
        try:
            if not di_text:
                return 0.0

            _t_sample0 = _time.time()
            # 并行计算：在进行主链路(gen_map/ie/struct/cover/medfact)时并发触发 style 评估
            def _task_style() -> float:
                _ts = _time.time()
                style_out_local = evaluate_style_for_text(di_text, model=None, temperature=0.0)
                r_style_local = float(calculate_advanced_style_score(style_out_local.get("scores", {})))
                vprint(f"[aggregate] gen={gen_idx} style took {(_time.time()-_ts):.3f}s")
                return r_style_local

            with ThreadPoolExecutor(max_workers=2) as _ex:
                fut_style = _ex.submit(_task_style)

                # 主链路：gen_map -> ie_result -> struct/cover -> medfact
                _t0 = _time.time()
                gen_map = extract_generated_map_via_llm(di_text, model=None, temperature=0.0)
                _t1 = _time.time()
                vprint(f"[aggregate] gen={gen_idx} gen_map took {(_t1-_t0):.3f}s")
                ie_result = extract_ie_from_generated_map(gen_map, model=None, temperature=0.0)
                _t2 = _time.time()
                vprint(f"[aggregate] gen={gen_idx} ie_result took {(_t2-_t1):.3f}s")

                gold_map = gold_map_obj[i] if gold_map_obj and i < len(gold_map_obj) else {k: "" for k in NINE_KEYS}
                _ts1 = _time.time()
                r_struct = compute_final_r_struct(gen_map, gold_map, alpha=0.6, beta=0.4, L_min=10)
                _ts2 = _time.time()
                vprint(f"[aggregate] gen={gen_idx} struct took {(_ts2-_ts1):.3f}s")

                gold_true = t_gold_anchors_obj[i] if t_gold_anchors_obj and i < len(t_gold_anchors_obj) else {k: [] for k in NINE_KEYS}
                source_true = t_source_anchors_obj[i] if t_source_anchors_obj and i < len(t_source_anchors_obj) else {k: [] for k in NINE_KEYS}
                all_true = t_all_anchors_obj[i] if t_all_anchors_obj and i < len(t_all_anchors_obj) else {k: [] for k in NINE_KEYS}
                _tc1 = _time.time()
                f1_gold, _ = calculate_risk_weighted_f1(gold_true, ie_result)
                f1_source, _ = calculate_risk_weighted_f1(source_true, ie_result)
                p_all = calculate_combined_precision(all_true, ie_result)
                r_cover = calculate_r_cover(f1_gold, f1_source, p_all, w_gold=0.7, w_source=0.3, lambda_penalty=0.2)
                _tc2 = _time.time()
                vprint(f"[aggregate] gen={gen_idx} cover took {(_tc2-_tc1):.3f}s")

                row_dict = {
                    "note_id": note_id[i] if note_id and i < len(note_id) else f"row_{i}",
                    "formated_source_note": formated_source_note[i] if formated_source_note and i < len(formated_source_note) else "",
                }
                src_row = pd.Series(row_dict)
                _tm1 = _time.time()
                medfact_eval = evaluate_medfact_for_ie_results(ie_results=ie_result, source_row=src_row)
                medfact_stats = calculate_r_medfact(medfact_eval)
                _tm2 = _time.time()
                vprint(f"[aggregate] gen={gen_idx} medfact took {(_tm2-_tm1):.3f}s")
                r_medfact = float(medfact_stats.get("r_medfact_score", 0.0))

                # 等待 style 结果
                _tw1 = _time.time()
                r_style = fut_style.result()
                _tw2 = _time.time()
                vprint(f"[aggregate] gen={gen_idx} wait_style took {(_tw2-_tw1):.3f}s")

            total = (
                weights.get('struct', 0.15) * float(r_struct)
                + weights.get('cover', 0.35) * float(r_cover)
                + weights.get('medfact', 0.4) * float(r_medfact)
                + weights.get('style', 0.1) * float(r_style)
            )

            vprint(
                f"[aggregate] sample={i} gen={gen_idx} r_struct={r_struct:.4f} r_cover={r_cover:.4f} r_medfact={r_medfact:.4f} r_style={r_style:.4f} total={total:.4f} elapsed={(\
                    _time.time()-_t_sample0):.3f}s"
            )

            return float(total)
        except Exception as e:
            print(f"[aggregate] scoring failed sample_idx={i} gen={gen_idx}: {e}")
            return 0.0

    def _score_one_sample(i: int, grp: List[str]) -> float:
    # Debug: print the current group's length
        try:
            vprint(f"[aggregate] sample={i} grp_len={len(grp)}")
        except Exception:
            pass
        gen_scores: List[float] = []
    # Score multiple generations for the same prompt in parallel, then average
        with ThreadPoolExecutor(max_workers=min(len(grp), 4)) as _group_pool:
            futures = [_group_pool.submit(_score_one_generation, i, gen_idx, content) for gen_idx, content in enumerate(grp)]
            for fut in futures:
                gen_scores.append(fut.result())
        final_score = float(sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
        return final_score

    # Process all samples in parallel
    out_scores: List[float] = []
    with ThreadPoolExecutor(max_workers=min(len(groups), 4)) as _sample_pool:
        futures = [_sample_pool.submit(_score_one_sample, i, grp) for i, grp in enumerate(groups)]
        for fut in futures:
            out_scores.append(fut.result())
    return out_scores


