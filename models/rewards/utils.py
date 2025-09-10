import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)


def extract_answer_text(text: str) -> str | None:
    m = _ANSWER_RE.search(text or "")
    return m.group(1) if m else None


def extract_think_text(text: str) -> str | None:
    m = _THINK_RE.search(text or "")
    return m.group(1) if m else None


def extract_post_think_answer(text: str) -> str:
    """Extract final answer as content after </think>.

    Fallbacks:
    - If no </think> is found, try <answer>...</answer>.
    - If neither is found, return the full text.
    """
    s = text or ""
    if not isinstance(s, str):
        s = str(s)
    m = _THINK_RE.search(s)
    if m:
        out = s[m.end():]
        # strip potential trailing control tokens/spaces
        out = out.strip()
        # If wrapped again in <answer>, unwrap once
        m2 = _ANSWER_RE.search(out)
        if m2:
            return m2.group(1).strip()
        return out
    m3 = _ANSWER_RE.search(s)
    if m3:
        return m3.group(1).strip()
    return s.strip()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_nine_keys_text(obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in NINE_KEYS:
        v = obj.get(k, "") if isinstance(obj, dict) else ""
        if isinstance(v, str):
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[k] = ""
    return out


def ensure_nine_keys_arrays(obj: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k in NINE_KEYS:
        v = obj.get(k, []) if isinstance(obj, dict) else []
        if isinstance(v, list):
            arr = [str(s).strip() for s in v if isinstance(s, (str, int, float)) and str(s).strip()]
        elif isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    arr = [str(x).strip() for x in parsed if isinstance(x, (str, int, float)) and str(x).strip()]
                except Exception:
                    arr = [s] if s else []
            else:
                arr = [s] if s else []
        else:
            arr = []
    # De-duplicate while preserving order
        seen, deduped = set(), []
        for s in arr:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        out[k] = deduped
    return out


def verbose_enabled() -> bool:
    try:
        v = os.getenv("REWARD_VERBOSE", "").strip().lower()
        return v not in ("", "0", "false", "off")
    except Exception:
        return False


def vprint(msg: str) -> None:
    if verbose_enabled():
        try:
            print(f"[reward] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass


def get_logs_dir() -> Path | None:
    out_dir = os.getenv("GRPO_OUTPUT_DIR", "").strip()
    if not out_dir:
        return None
    logs_dir = Path(out_dir) / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return logs_dir


def log_jsonl(rel_name: str, record: Dict[str, Any]) -> None:
    logs_dir = get_logs_dir()
    if logs_dir is None:
        return
    try:
        path = logs_dir / rel_name
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def write_all_generations(prompts: List[List[Dict[str, str]]], completions: List[List[Dict[str, str]]], file_name: str = "all_generations.jsonl") -> None:
    ts = time.time()
    for idx, comp_list in enumerate(completions or []):
        # 提取 user 文本
        user_text = ""
        try:
            if 0 <= idx < len(prompts):
                for m in prompts[idx]:
                    if m.get("role") == "user":
                        user_text = m.get("content", "")
                        break
        except Exception:
            pass
        gens: List[str] = []
        try:
            for item in (comp_list or []):
                if isinstance(item, dict):
                    gens.append(str(item.get("content", "")))
        except Exception:
            pass
        log_jsonl(file_name, {"t": ts, "sample_idx": int(idx), "user": user_text, "generations": gens})


