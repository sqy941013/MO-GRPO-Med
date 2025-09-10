import json
import re
from typing import Any, Dict, List
from pathlib import Path

from .utils import read_text_file, extract_post_think_answer
from .openai_io import chat_json_with_retry
import os


_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def _build_user_prompt_for_style(user_template: str, generated_text: str) -> str:
    candidates = ["{generated_text}", "{text}", "{input_text}"]
    for marker in candidates:
        if marker in user_template:
            return user_template.replace(marker, generated_text)
    return f"{user_template}\n\n[TEXT]\n{generated_text}"


def get_ds_r1_model() -> str:
    return os.getenv("REWARD_DS_R1_MODEL", "deepseek-r1-250528")


def evaluate_style_for_text(
    generated_text: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    proj_root = Path(__file__).resolve().parents[2]
    candidates = [
        proj_root / "docs" / "prompts" / "reward_function" / "r_style",
        proj_root / "prompts" / "reward_function" / "r_style",
    ]
    pdir = None
    for c in candidates:
        if (c / "system.txt").exists() and (c / "user.txt").exists():
            pdir = c
            break
    if pdir is None:
        raise FileNotFoundError("r_style prompts not found in prompts/... or docs/prompts/...")

    system_prompt = read_text_file(pdir / "system.txt")
    user_template = read_text_file(pdir / "user.txt")
    user_prompt = _build_user_prompt_for_style(user_template, generated_text)

    scores: Dict[str, int] = {"readability": 1, "tone": 1, "succinctness": 1}
    think_process = ""
    try:
        if model is None:
            model = get_ds_r1_model()
        js, raw_out, ok = chat_json_with_retry(
            model=str(model),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            want_raw=True,
            retries=4,
        )
        if isinstance(raw_out, str):
            m = _THINK_RE.search(raw_out)
            if m:
                think_process = m.group(1).strip()
        obj = js if isinstance(js, dict) else {}
        if not obj or not any(k in obj for k in ("scores", "readability", "tone", "succinctness")):
            try:
                content_wo_think = _THINK_RE.sub("", raw_out or "") if isinstance(raw_out, str) else ""
                _m = re.search(r"\{[\s\S]*\}", content_wo_think)
                if _m:
                    obj = json.loads(_m.group(0))
            except Exception:
                pass
        obj_scores = obj.get("scores", obj)
        def _clamp3(x: Any) -> int:
            try:
                v = int(x)
            except Exception:
                v = 1
            return 1 if v < 1 else 3 if v > 3 else v
        scores = {
            "readability": _clamp3(obj_scores.get("readability", 1)),
            "tone": _clamp3(obj_scores.get("tone", 1)),
            "succinctness": _clamp3(obj_scores.get("succinctness", 1)),
        }
    except Exception as e:
        print(f"[r_style] evaluate failed: {e}")

    return {"scores": scores, "think_process": think_process}


def calculate_advanced_style_score(style_ratings: Dict[str, int], weights: Dict[str, float] | None = None) -> float:
    if weights is None:
        weights = {"readability": 0.5, "tone": 0.25, "succinctness": 0.25}
    r = (style_ratings.get("readability", 1) - 1) / 2
    t = (style_ratings.get("tone", 1) - 1) / 2
    s = (style_ratings.get("succinctness", 1) - 1) / 2
    return float(r * weights.get("readability", 0.0) + t * weights.get("tone", 0.0) + s * weights.get("succinctness", 0.0))


def compute_r_style_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    **kwargs,
) -> List[float]:
    groups: List[List[str]] = []
    for comp_list in (completions or []):
        grp: List[str] = []
        for item in (comp_list or []):
            grp.append(str(item.get("content", "")) if isinstance(item, dict) else "")
        groups.append(grp)

    out_scores: List[float] = []
    for grp in groups:
        gen_scores: List[float] = []
        for content in grp:
            di_text = extract_post_think_answer(content)
            if not di_text:
                gen_scores.append(0.0)
                continue
            try:
                style_out = evaluate_style_for_text(di_text, model=None, temperature=0.0)
                style_scores = style_out.get("scores", {})
                final = calculate_advanced_style_score(style_scores)
                gen_scores.append(float(final))
            except Exception as e:
                print(f"[r_style] scoring failed: {e}")
                gen_scores.append(0.0)
        final_score = float(sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
        out_scores.append(final_score)
    return out_scores


