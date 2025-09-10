import json
from typing import Any, Dict, List
import pandas as pd
from pathlib import Path

from .utils import NINE_KEYS, read_text_file
from .openai_io import chat_json_with_retry
import os
from .struct import extract_generated_map_via_llm, get_ds_v3_model
from .utils import extract_post_think_answer


TARGET_CATEGORIES: List[str] = [
    "Why admitted", "What happened in hospital", "What to do after discharge",
    "Medication changes", "Follow-up", "Red flags (when to seek care)",
    "Lifestyle / Activity", "Wound / Device care", "Contacts / Emergency",
]


def _to_list_safe(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if isinstance(x, (str, int, float)) and str(x).strip()]
            except Exception:
                pass
        try:
            obj = eval(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if isinstance(x, (str, int, float)) and str(x).strip()]
        except Exception:
            return [s]
        return [s]
    return [str(value).strip()] if str(value).strip() else []


def _build_stmts_from_ie_results(ie_results: Dict[str, Any], max_per_category: int | None = None) -> Dict[str, List[str]]:
    stmts: Dict[str, List[str]] = {}
    for cat in TARGET_CATEGORIES:
        vals = _to_list_safe(ie_results.get(cat, []))
        if max_per_category is not None and max_per_category > 0:
            vals = vals[:max_per_category]
        stmts[cat] = vals
    return stmts


def _build_source_notes_from_row(row: pd.Series) -> str:
    txt = str(row.get("formated_source_note", ""))
    return txt.strip()


def evaluate_medfact_for_ie_results(
    ie_results: Dict[str, Any],
    source_row: Dict[str, Any] | pd.Series,
    model: str | None = None,
    temperature: float = 0.0,
    max_per_category: int | None = None,
) -> Dict[str, Any]:
    if isinstance(source_row, dict):
        source_row = pd.Series(source_row)
    elif isinstance(source_row, pd.DataFrame):
        if len(source_row) == 0:
            raise ValueError("source_row DataFrame is empty")
        source_row = source_row.iloc[0]

    proj_root = Path(__file__).resolve().parents[2]
    candidates = [
        proj_root / "docs" / "prompts" / "reward_function" / "r_medfact",
        proj_root / "prompts" / "reward_function" / "r_medfact",
    ]
    pdir = None
    for c in candidates:
        if (c / "system.txt").exists() and (c / "user.txt").exists():
            pdir = c
            break
    if pdir is None:
        raise FileNotFoundError("r_medfact prompts not found in prompts/... or docs/prompts/...")

    system_prompt = read_text_file(pdir / "system.txt")
    user_template = read_text_file(pdir / "user.txt")

    source_notes = _build_source_notes_from_row(source_row)
    stmts = _build_stmts_from_ie_results(ie_results, max_per_category=max_per_category)

    user_prompt = user_template
    user_prompt = user_prompt.replace("{source_notes}", source_notes)
    user_prompt = user_prompt.replace("{stmt}", json.dumps(stmts, ensure_ascii=False))

    if model is None:
        model = get_ds_v3_model()
    result_obj, _raw, ok = chat_json_with_retry(
        model=str(model),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        want_raw=False,
        retries=4,
    )
    if not ok or not isinstance(result_obj, dict):
        empty_results = {cat: [{"factuality": "", "safety": "", "quote": ""} for _ in stmts.get(cat, [])] for cat in TARGET_CATEGORIES}
        result_obj = {"results": empty_results, "error": "retry_failed"}

    if not isinstance(result_obj, dict) or "results" not in result_obj:
        result_obj = {"results": {cat: [{"factuality": "", "safety": "", "quote": ""} for _ in stmts.get(cat, [])] for cat in TARGET_CATEGORIES}}

    return {
        "note_id": str(source_row.get("note_id", "")),
        "stmts": stmts,
        "results": result_obj.get("results", {}),
    }


def calculate_r_medfact(medfact_results: Dict[str, Any]) -> Dict[str, Any]:
    FACTUALITY_MAP = {
        "supported": 1.0,
        "partial": 0.5,
        "unsupported": 0.0,
        "conflict": 0.0,
    }
    SAFETY_DEDUCTION = {
        "safe": 0.0,
        "unsafe_moderate": -0.5,
        "unsafe_major": -1.0,
    }

    total_statements = 0
    total_score = 0.0
    has_major_unsafe_flag = False
    category_stats: Dict[str, Any] = {}

    results = medfact_results.get("results", {}) if isinstance(medfact_results, dict) else {}
    for category, judgments in results.items():
        category_scores: List[float] = []
        if not isinstance(judgments, list):
            continue
        for judgment in judgments:
            factuality = str(judgment.get("factuality", "")).strip().lower()
            safety = str(judgment.get("safety", "")).strip().lower()
            quote = str(judgment.get("quote", "")).strip()

            if factuality in ["supported", "partial"] and not quote:
                factuality = "unsupported"

            factuality_score = FACTUALITY_MAP.get(factuality, 0.0)
            safety_deduction = SAFETY_DEDUCTION.get(safety, 0.0)

            if safety == "unsafe_major":
                has_major_unsafe_flag = True

            final_score = max(0.0, factuality_score + safety_deduction)
            category_scores.append(final_score)
            total_score += final_score
            total_statements += 1

        if category_scores:
            category_stats[category] = {
                "count": len(category_scores),
                "avg_score": sum(category_scores) / len(category_scores),
                "total_score": sum(category_scores),
            }
        else:
            category_stats[category] = {"count": 0, "avg_score": 0.0, "total_score": 0.0}

    r_medfact_score = total_score / total_statements if total_statements > 0 else 0.0
    return {
        "r_medfact_score": r_medfact_score,
        "total_statements": total_statements,
        "total_score": total_score,
        "has_major_unsafe_flag": has_major_unsafe_flag,
        "category_stats": category_stats,
    }


def compute_r_medfact_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    note_id: List[str] = None,
    formated_source_note: List[str] = None,
    **kwargs,
) -> List[float]:
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
            di_text = extract_post_think_answer(content)
            try:
                if not di_text:
                    gen_scores.append(0.0)
                    continue
                from .cover import extract_ie_from_generated_map
                gen_map = extract_generated_map_via_llm(di_text, model=None, temperature=0.0)
                ie_result = extract_ie_from_generated_map(gen_map, model=None, temperature=0.2)

                row_dict = {
                    "note_id": note_id[i] if note_id and i < len(note_id) else f"row_{i}",
                    "formated_source_note": formated_source_note[i] if formated_source_note and i < len(formated_source_note) else "",
                }
                src_row = pd.Series(row_dict)
                medfact_eval = evaluate_medfact_for_ie_results(ie_results=ie_result, source_row=src_row)
                medfact_stats = calculate_r_medfact(medfact_eval)
                gen_scores.append(float(medfact_stats.get("r_medfact_score", 0.0)))
            except Exception as e:
                print(f"[r_medfact] scoring failed sample_idx={i}: {e}")
                gen_scores.append(0.0)
        final_score = float(sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
        out_scores.append(final_score)
    return out_scores


