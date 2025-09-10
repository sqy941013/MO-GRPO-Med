import os
import random
import time
from typing import Any, Dict, Tuple

from pathlib import Path
import sys

# Keep compatibility with previous layout: import scripts.openai_helper from project root
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.openai_helper import load_client, chat_json_extract  # type: ignore


def chat_json_with_retry(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    want_raw: bool = False,
    retries: int = 10,
    base_wait: float = 1.5,
) -> Tuple[Dict[str, Any], str | None, bool]:
    client = load_client()
    last_err = None
    try:
        _env_stream = os.getenv("LLM_STREAM", "1").strip()
        use_stream = _env_stream not in ("0", "false", "False")
    except Exception:
        use_stream = True

    for attempt in range(retries):
        try:
            if want_raw:
                js, raw = chat_json_extract(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    return_raw=True,
                    stream=use_stream,
                )
                return js, raw, True
            else:
                js = chat_json_extract(
                    client=client,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    stream=use_stream,
                )
                return js, None, True
        except Exception as e:
            last_err = e
            print(f"[LLM Retry] {model} attempt {attempt+1}/{retries} failed: {e}")
            wait = base_wait ** attempt + random.random()
            time.sleep(wait)
    print(f"[LLM Retry] {model} failed after {retries} attempts. Giving up.")
    return {}, None, False


