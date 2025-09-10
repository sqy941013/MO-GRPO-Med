import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _reward_cache_path() -> Path:
    p = os.getenv("REWARD_CACHE_CSV", "")
    if p:
        path = Path(p)
    else:
        base = Path(os.getenv("GRPO_CACHE_DIR", Path.cwd() / "data" / "intermediate"))
        path = base / "reward_cache.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def cache_lookup(cache_type: str, key_sha1: str, model_name: str) -> Dict[str, Any] | None:
    path = _reward_cache_path()
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        sub = df[(df.get("type", "") == cache_type) & (df.get("key_sha1", "") == key_sha1) & (df.get("model", "") == model_name)]
        if not sub.empty:
            row = sub.iloc[0]
            data = row.get("data_json", "{}")
            return json.loads(data)
    except Exception:
        return None
    return None


def cache_store(cache_type: str, key_sha1: str, model_name: str, data: Dict[str, Any]) -> None:
    path = _reward_cache_path()
    try:
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=["type", "key_sha1", "model", "data_json", "updated_at"])
        rec = {
            "type": cache_type,
            "key_sha1": key_sha1,
            "model": model_name,
            "data_json": json.dumps(data, ensure_ascii=False),
            "updated_at": int(time.time()),
        }
        mask = (df.get("type", "") == cache_type) & (df.get("key_sha1", "") == key_sha1) & (df.get("model", "") == model_name)
        if mask.any():
            df.loc[mask, list(rec.keys())] = list(rec.values())
        else:
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        df.to_csv(path, index=False)
    except Exception:
        pass


