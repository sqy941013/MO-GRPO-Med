import json
from typing import Any, Dict, List
from pathlib import Path

from .utils import (
    NINE_KEYS,
    ensure_nine_keys_arrays,
    read_text_file,
    vprint,
)
import time as _time
import os as _os
from .openai_io import chat_json_with_retry
from .cache import cache_lookup, cache_store
from .logs import log_llm_call
from .struct import extract_generated_map_via_llm, get_ds_v3_model
from .utils import extract_post_think_answer


def _build_user_prompt_for_cover(user_template: str, input_obj: Dict[str, Any]) -> str:
    marker = "{PASTE THE INPUT OBJECT HERE}"
    return user_template.replace(marker, json.dumps(input_obj, ensure_ascii=False))


def extract_ie_from_generated_map(
    generated_map: Dict[str, Any],
    model: str | None = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    try:
        _t0 = _time.time()
        vprint("r_cover: start extracting IE from generated_map")
        proj_root = Path(__file__).resolve().parents[2]
        candidates = [
            proj_root / "docs" / "prompts" / "reward_function" / "r_cover",
            proj_root / "prompts" / "reward_function" / "r_cover",
        ]
        pdir = None
        for c in candidates:
            if (c / "system.txt").exists() and (c / "user.txt").exists():
                pdir = c
                break
        if pdir is None:
            raise FileNotFoundError("r_cover prompts not found in prompts/... or docs/prompts/...")

        system_prompt = read_text_file(pdir / "system.txt")
        user_template = read_text_file(pdir / "user.txt")

        input_obj: Dict[str, Any] = {}
        for k in NINE_KEYS:
            v = generated_map.get(k, "")
            if isinstance(v, list):
                lst = [str(s).strip() for s in v if isinstance(s, (str, int, float)) and str(s).strip()]
                input_obj[k] = lst if lst else None
            elif isinstance(v, str):
                vv = v.strip()
                input_obj[k] = vv if vv else None
            else:
                input_obj[k] = None

        user_prompt = _build_user_prompt_for_cover(user_template, input_obj)

        vprint("r_cover: prompts loaded, checking cache for IE result")
        ie_key = _sha1_text(f"IE::{model}::{json.dumps(input_obj, ensure_ascii=False, sort_keys=True)}")
        _tc = _time.time()
        cached_ie = cache_lookup("r_cover_extract_ie", ie_key, model)
        vprint(f"r_cover: cache lookup took {(_time.time()-_tc):.3f}s")
        if cached_ie is not None and isinstance(cached_ie, dict):
            vprint("r_cover: hit cache for IE result")
            norm = ensure_nine_keys_arrays(cached_ie)
            norm["raw_input"] = input_obj
            norm["raw_output"] = cached_ie.get("_raw", "")
            norm["flag"] = 1
            vprint(f"r_cover: total took {(_time.time()-_t0):.3f}s")
            return norm

        vprint("r_cover: cache miss, calling LLM to extract IE")
        _tllm = _time.time()
        if model is None:
            model = get_ds_v3_model()
        js, raw_out, success = chat_json_with_retry(
            model=str(model),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            want_raw=True,
            retries=4,
        )
        vprint(f"r_cover: llm call took {(_time.time()-_tllm):.3f}s")
        try:
            log_llm_call(
                call_name="r_cover_extract_ie",
                model_name=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_out or "",
                parsed_obj=js if isinstance(js, dict) else {},
                ok=success,
            )
        except Exception:
            pass

        if not isinstance(js, dict):
            js = {k: [] for k in NINE_KEYS}
        if raw_out is None:
            raw_out = ""

        norm = ensure_nine_keys_arrays(js)
        norm["raw_input"] = input_obj
        norm["raw_output"] = raw_out
        norm["flag"] = 1 if success and bool(raw_out) else 0
        try:
            vprint("r_cover: IE extracted ok, storing cache")
            _ts = _time.time()
            to_store = dict(js if isinstance(js, dict) else {})
            if isinstance(raw_out, str):
                to_store["_raw"] = raw_out
            cache_store("r_cover_extract_ie", ie_key, model, to_store)
            vprint(f"r_cover: cache store took {(_time.time()-_ts):.3f}s")
        except Exception:
            pass
        vprint(f"r_cover: total took {(_time.time()-_t0):.3f}s")
        return norm
    except Exception as e:
        print(f"[r_cover] IE extraction failed: {e}")
        return {k: [] for k in NINE_KEYS}


def _sha1_text(text: str) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


# Semantic matching (lazy-load the embedding model)
_SEM_MODEL = None
_SEM_LOG_EVERY = int((_os.getenv("REWARD_SEM_LOG_EVERY", "50") or "50"))
_SEM_LOG_COUNT = 0
_SEM_ACCUM = 0.0
_SEM_ACCUM_N = 0


def _get_semantic_model():
    global _SEM_MODEL
    if _SEM_MODEL is None:
        try:
            _t0 = _time.time()
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            
            # Force CPU to avoid 'meta' tensor issues on some environments
            model_name = 'emilyalsentzer/Bio_ClinicalBERT'
            print(f"[r_cover] Loading {model_name} on CPU to avoid meta tensor issues")
            
            # Force CPU via environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            try:
                # Method 1: load directly on CPU
                _SEM_MODEL = SentenceTransformer(model_name, device='cpu')
                print(f"[r_cover] Successfully loaded {model_name} on CPU")
                
            except Exception as cpu_error:
                print(f"[r_cover] CPU loading failed: {cpu_error}")
                
                try:
                    # Method 2: manual construction using transformers
                    from transformers import AutoModel, AutoTokenizer
                    from sentence_transformers import SentenceTransformer
                    from sentence_transformers.models import Transformer, Pooling
                    
                    print(f"[r_cover] Trying manual construction with transformers")
                    
                    # Manually build SentenceTransformer
                    word_embedding_model = Transformer(model_name, device='cpu')
                    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
                    
                    _SEM_MODEL = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                    _SEM_MODEL = _SEM_MODEL.to('cpu')
                    
                    print(f"[r_cover] Manual construction successful")
                    
                except Exception as manual_error:
                    print(f"[r_cover] Manual construction failed: {manual_error}")
                    
                    try:
                        # Method 3: specify dtype to help avoid device issues
                        print(f"[r_cover] Trying with torch_dtype=torch.float32")
                        
                        _SEM_MODEL = SentenceTransformer(
                            model_name, 
                            device='cpu',
                            model_kwargs={'torch_dtype': torch.float32}
                        )
                        
                        print(f"[r_cover] torch_dtype method successful")
                        
                    except Exception as dtype_error:
                        print(f"[r_cover] torch_dtype method failed: {dtype_error}")
                        _SEM_MODEL = None
                        
            # 确保模型在CPU上并且可用
            if _SEM_MODEL is not None:
                _SEM_MODEL = _SEM_MODEL.to('cpu')
                _SEM_MODEL.eval()  # 设置为评估模式
                
                # 测试模型是否正常工作
                try:
                    test_text = ["This is a test sentence."]
                    test_embedding = _SEM_MODEL.encode(test_text)
                    print(f"[r_cover] Model test successful, embedding shape: {test_embedding.shape}")
                except Exception as test_error:
                    print(f"[r_cover] Model test failed: {test_error}")
                    _SEM_MODEL = None
                        
            if _SEM_MODEL is not None:
                vprint(f"r_cover: semantic model loaded in {(_time.time()-_t0):.3f}s")
            else:
                print(f"[r_cover] Failed to load {model_name} on CPU")
                
        except Exception as e:
            print(f"[r_cover] load semantic model failed: {e}")
            _SEM_MODEL = None
    return _SEM_MODEL


def _semantic_match(a: str, b: str, threshold: float = 0.8) -> bool:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    try:
        model = _get_semantic_model()
        if model is None:
            return a.lower() == b.lower()
        from sentence_transformers import util
        _t0 = _time.time()
        emb1 = model.encode(a, convert_to_tensor=True)
        emb2 = model.encode(b, convert_to_tensor=True)
        _t1 = _time.time()
        cos = util.pytorch_cos_sim(emb1, emb2)
        took = _time.time() - _t0
        global _SEM_LOG_COUNT, _SEM_ACCUM, _SEM_ACCUM_N
        _SEM_LOG_COUNT += 1
        _SEM_ACCUM += took
        _SEM_ACCUM_N += 1
        if (_SEM_LOG_COUNT % _SEM_LOG_EVERY == 0) or (took >= 0.1):
            try:
                mean_t = _SEM_ACCUM / max(_SEM_ACCUM_N, 1)
                vprint(f"r_cover: semantic match encode+sim mean={mean_t:.3f}s last={took:.3f}s count={_SEM_LOG_COUNT}")
            except Exception:
                pass
            _SEM_ACCUM = 0.0
            _SEM_ACCUM_N = 0
        return float(cos) > threshold
    except Exception as e:
        print(f"[r_cover] semantic match failed: {e}")
        return a.lower() == b.lower()


CATEGORY_TO_SECTION_MAP = {
    'why_admitted': "Why admitted",
    'what_happened': "What happened in hospital",
    'tasks': "What to do after discharge",
    'med_changes': "Medication changes",
    'followups': "Follow-up",
    'red_flags': "Red flags (when to seek care)",
    'lifestyle_activity': "Lifestyle / Activity",
    'wound_device_care': "Wound / Device care",
    'contacts_emergency': "Contacts / Emergency"
}

RISK_WEIGHTS = {
    'red_flags': 4.0,
    'med_changes': 3.5,
    'followups': 3.0,
    'tasks': 2.0,
    'contacts_emergency': 1.5,
    'lifestyle_activity': 1.0,
    'wound_device_care': 0.5,
    'why_admitted': 0.2,
    'what_happened': 0.2,
}


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, tuple):
        return [str(x) for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                return _to_list(parsed)
            except Exception:
                return [s]
        return [s]
    return [str(value)]


def calculate_risk_weighted_f1(y_true: Dict[str, Any], y_pred: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    target_categories = list(CATEGORY_TO_SECTION_MAP.values())
    category_scores: Dict[str, Any] = {}
    weighted_scores: List[float] = []
    total_weight: float = 0.0

    for cat_key, section_name in CATEGORY_TO_SECTION_MAP.items():
        if section_name not in target_categories:
            continue
        weight = RISK_WEIGHTS[cat_key]

        true_values = _to_list(y_true.get(section_name, [])) if isinstance(y_true, dict) else []
        pred_values = _to_list(y_pred.get(section_name, [])) if isinstance(y_pred, dict) else []

        tp = 0.0
        if true_values and pred_values:
            matched_true = set()
            for pred_item in pred_values:
                best, best_idx = 0.0, -1
                for i, true_item in enumerate(true_values):
                    if i in matched_true:
                        continue
                    score = 1.0 if _semantic_match(true_item, pred_item) else 0.0
                    if score > best:
                        best, best_idx = score, i
                if best_idx != -1:
                    tp += best
                    matched_true.add(best_idx)

        precision = tp / len(pred_values) if pred_values else 0.0
        recall = tp / len(true_values) if true_values else 0.0
        if not true_values and not pred_values:
            f1 = 1.0
            precision = 1.0
            recall = 1.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        category_scores[cat_key] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'weight': weight,
            'weighted_f1': f1 * weight,
            'true_count': len(true_values),
            'pred_count': len(pred_values),
            'matched_count': tp,
        }

        weighted_scores.append(f1 * weight)
        total_weight += weight

    weighted_f1 = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
    return weighted_f1, category_scores


def calculate_combined_precision(t_all_anchors: Dict[str, Any], ie_result: Dict[str, Any]) -> float:
    target_categories = list(CATEGORY_TO_SECTION_MAP.values())

    def parse_list(v: Any) -> List[str]:
        return _to_list(v)

    total_generated = 0
    total_matched = 0

    for section_name in target_categories:
        pred_values = parse_list(ie_result.get(section_name)) if section_name in ie_result else []
        true_values = parse_list(t_all_anchors.get(section_name)) if section_name in t_all_anchors else []
        if not pred_values:
            continue
        total_generated += len(pred_values)
        for pred_item in pred_values:
            if any(_semantic_match(pred_item, tv) for tv in true_values):
                total_matched += 1

    if total_generated == 0:
        any_true = any(parse_list(v) for k, v in t_all_anchors.items() if k in target_categories)
        return 1.0 if not any_true else 0.0
    return total_matched / total_generated


def calculate_r_cover(f1_gold: float, f1_source: float, p_all_combined: float, w_gold: float = 0.7, w_source: float = 0.3, lambda_penalty: float = 0.2) -> float:
    weighted_coverage = w_gold * f1_gold + w_source * f1_source
    return weighted_coverage * (1 - lambda_penalty) + p_all_combined * lambda_penalty


def compute_r_cover_reward(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answer: List[str],
    note_id: List[str] = None,
    t_source_anchors_obj: List[Dict[str, Any]] = None,
    t_gold_anchors_obj: List[Dict[str, Any]] = None,
    t_all_anchors_obj: List[Dict[str, Any]] = None,
    gold_map_obj: List[Dict[str, Any]] = None,
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
                gen_map = extract_generated_map_via_llm(di_text, model=None, temperature=0.0)
                ie_result = extract_ie_from_generated_map(gen_map, model=None, temperature=0.2)

                gold_true = t_gold_anchors_obj[i] if t_gold_anchors_obj and i < len(t_gold_anchors_obj) else {k: [] for k in NINE_KEYS}
                source_true = t_source_anchors_obj[i] if t_source_anchors_obj and i < len(t_source_anchors_obj) else {k: [] for k in NINE_KEYS}
                all_true = t_all_anchors_obj[i] if t_all_anchors_obj and i < len(t_all_anchors_obj) else {k: [] for k in NINE_KEYS}

                f1_gold, _ = calculate_risk_weighted_f1(gold_true, ie_result)
                f1_source, _ = calculate_risk_weighted_f1(source_true, ie_result)
                p_all = calculate_combined_precision(all_true, ie_result)
                r_cover = calculate_r_cover(f1_gold, f1_source, p_all, w_gold=0.7, w_source=0.3, lambda_penalty=0.2)
                gen_scores.append(float(r_cover))
            except Exception as e:
                print(f"[r_cover] scoring failed sample_idx={i}: {e}")
                gen_scores.append(0.0)
        final_score = float(sum(gen_scores) / len(gen_scores)) if gen_scores else 0.0
        out_scores.append(final_score)
    return out_scores


