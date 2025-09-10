#!/usr/bin/env python3
import json, os, re, sys
from typing import Dict, Any

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Please install 'openai' v1+ (pip install openai)") from e


def load_client() -> OpenAI:
    load_dotenv()
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError("OPENAI_API_BASE and OPENAI_API_KEY must be set in environment or .env")
    return OpenAI(base_url=api_base, api_key=api_key)


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # fallback: find first top-level JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from model output")


def chat_json_extract(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.3, return_raw: bool = False, stream: bool = False):
    # Detect DeepSeek-R1 models; avoid forcing JSON format so the <think> block is preserved
    use_json_format = not ("deepseek-r1" in model.lower() or "deepseek-r1-250528" in model.lower())
    
    # Build request parameters
    chat_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    
    # Only non‑R1 models should enforce JSON output
    if use_json_format:
        chat_params["response_format"] = {"type": "json_object"}
    
    content = ""
    if stream:
        # Stream tokens and accumulate; parse JSON at the end
        try:
            for chunk in client.chat.completions.create(stream=True, **chat_params):
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                    if delta and getattr(delta, "content", None):
                        content += delta.content
                except Exception:
                    pass
        except Exception as e:
            # Some providers don't support response_format with streaming; retry without response_format
            if "response_format" in chat_params:
                try:
                    _tmp = dict(chat_params)
                    _tmp.pop("response_format", None)
                    for chunk in client.chat.completions.create(stream=True, **_tmp):
                        try:
                            delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
                            if delta and getattr(delta, "content", None):
                                content += delta.content
                        except Exception:
                            pass
                except Exception:
                    raise e
            else:
                raise e
    else:
        resp = client.chat.completions.create(**chat_params)
        content = resp.choices[0].message.content or ""
    # For R1-like models, strip <think> blocks before attempting to parse JSON
    THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
    try:
        parse_target = content
        if not use_json_format:
            # Remove <think> first to avoid interference when parsing JSON
            parse_target = THINK_RE.sub("", content or "")
        parsed = _extract_json(parse_target)
    except Exception as e:
        # Debug: dump raw content when JSON parsing fails
        try:
            dbg = os.getenv("GRPO_DEBUG", "").strip().lower()
            if dbg and dbg not in ("0", "false", "off"):
                print("[DEBUG] chat_json_extract failed to parse JSON. Raw content follows:", file=sys.stderr)
                try:
                    print(content, file=sys.stderr)
                except Exception:
                    pass
                # Also print the content with <think> removed
                try:
                    print("[DEBUG] content_without_think:", file=sys.stderr)
                    print(parse_target, file=sys.stderr)
                except Exception:
                    pass
        except Exception:
            pass
        raise
    if return_raw:
        return parsed, content
    return parsed


