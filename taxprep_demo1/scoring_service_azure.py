# scoring_service_azure.py
import os
import json
import time
import logging
import re
import hashlib
import ast
from typing import List, Dict, Optional
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from cachetools import TTLCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ensure logs folder exists
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LLM_LOG_FILE = LOG_DIR / "llm_responses.log"


# ---------------------------
# Heuristics fallback & mock
# ---------------------------
def heuristics_score(row: dict) -> Dict:
    score = 0.5
    sla_days = 7
    if row.get("turnaround_time_days", 0) > sla_days + 3:
        score -= 0.2
    if row.get("error_rate_pct", 0) > 5:
        score -= 0.25
    if row.get("communication_count", 0) < 2:
        score -= 0.05
    label = "Satisfied" if score >= 0.5 else "Dissatisfied"
    return {"label": label, "confidence": round(max(0.0, min(1.0, score)), 2)}


def mock_llm_judgement(row: dict) -> List[Dict]:
    drivers = []
    if row.get("turnaround_time_days", 0) > 10:
        drivers.append(
            {
                "factor": "turnaround_time_days",
                "impact": "High",
                "explain": f"{row.get('turnaround_time_days')} days vs SLA 7",
            }
        )
    if row.get("error_rate_pct", 0) > 2:
        drivers.append(
            {
                "factor": "error_rate_pct",
                "impact": "Medium",
                "explain": f"{row.get('error_rate_pct')}% error rate",
            }
        )
    if not drivers:
        drivers.append(
            {
                "factor": "communication_count",
                "impact": "Low",
                "explain": "regular communication",
            }
        )
    return drivers


# ---------------------------
# Azure OpenAI: LangChain or openai direct fallback
# ---------------------------
USE_AZURE_LANGCHAIN = False
USE_OPENAI_SDK = False
try:
    from langchain.chat_models import AzureChatOpenAI

    USE_AZURE_LANGCHAIN = True
    logger.info("LangChain AzureChatOpenAI available")
except Exception as e:
    logger.info("LangChain AzureChatOpenAI not available: %s", e)
    try:
        import openai  # fallback

        USE_OPENAI_SDK = True
        logger.info("openai SDK available (will use direct Azure path)")
    except Exception as e2:
        logger.info("openai SDK not available: %s", e2)
        USE_OPENAI_SDK = False

# Very explicit prompt asking for strict JSON
SCORING_PROMPT = """You are a strict JSON-only responder. Given client attributes, output EXACTLY one JSON object with keys:
- label: "Satisfied" or "Dissatisfied"
- confidence: number between 0.0 and 1.0
- top_drivers: list of objects each with { factor, impact, explain } where impact ∈ {"High","Medium","Low"}

REQUIREMENTS:
1) Return ONLY the JSON object and nothing else (no preface, no explanation, no trailing text, no code fences).
2) Use double quotes for all strings.
3) confidence must be numeric (e.g., 0.82). If you calculate a percent, convert to 0-1 scale.
4) top_drivers must be a JSON array (even if empty).
5) If you cannot determine a value, fill in a safe default:
   - label: "Dissatisfied"
   - confidence: 0.50
   - top_drivers: [{"factor":"parsing_failure","impact":"High","explain":"insufficient evidence"}]

Client: {attributes}
PeerExamples: {examples}
"""

# repair suffix used when first parse fails
REPAIR_SUFFIX = (
    "REPAIR: Your previous output was not valid JSON. Return ONLY valid JSON that matches the schema: "
    '{"label":"Satisfied|Dissatisfied","confidence":0.0-1.0,"top_drivers":[{"factor":"...","impact":"High|Medium|Low","explain":"..."}] }.'
)


# ---------- caching: manual TTL cache keyed by client_id or row-hash ----------
CACHE_TTL_SECONDS = 60 * 5
_cache = TTLCache(maxsize=2000, ttl=CACHE_TTL_SECONDS)


def _row_to_hash_key(row_dict: dict) -> str:
    try:
        normalized = json.dumps(row_dict, sort_keys=True, default=str)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(str(row_dict).encode("utf-8")).hexdigest()


def _cache_get(key: str):
    try:
        return _cache.get(key)
    except Exception:
        return None


def _cache_set(key: str, value, ttl: int = CACHE_TTL_SECONDS):
    try:
        _cache[key] = value
    except Exception:
        logger.exception("Failed to write to cache")


# ---------- helpers: logging LLM responses ----------
def _log_llm(name: str, text: str, parsed_candidate):
    try:
        with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("----\n")
            f.write(f"TIME: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(f"MODEL: {name}\n")
            f.write("RAW:\n")
            f.write(text + "\n")
            f.write("PARSED:\n")
            try:
                f.write(json.dumps(parsed_candidate, default=str, indent=2) + "\n")
            except Exception:
                f.write(str(parsed_candidate) + "\n")
            f.write("----\n\n")
    except Exception:
        logger.exception("Failed to write LLM log")


# ---------- robust JSON extraction + normalization ----------
def extract_json_from_text(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None

    # 1) direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) find first {...} (non-greedy) and try a few repairs
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        block = m.group(0)
        # try raw
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # try replace single quotes with double quotes
        try:
            parsed = json.loads(block.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # try ast.literal_eval (handles python dict formatting)
        try:
            parsed = ast.literal_eval(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 3) full-text attempts: strip code fences and try again
    cleaned = text.strip()
    # remove triple backticks if present
    cleaned = re.sub(
        r"^```.*?```$", lambda m: m.group(0).strip("`"), cleaned, flags=re.S
    )
    # try replacing single quotes
    try:
        parsed = json.loads(cleaned.replace("'", '"'))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def _normalize_parsed(parsed: dict) -> dict:
    """
    Normalize keys (strip whitespace, surrounding quotes) and coerce types
    """
    if not isinstance(parsed, dict):
        return parsed

    normalized = {}
    for k, v in parsed.items():
        nk = k.strip()
        # remove surrounding quotes if present in key string
        if (nk.startswith('"') and nk.endswith('"')) or (
            nk.startswith("'") and nk.endswith("'")
        ):
            nk = nk[1:-1].strip()
        nk = nk.strip()
        normalized[nk] = v

    # normalize label to str
    if "label" in normalized and normalized["label"] is not None:
        try:
            normalized["label"] = str(normalized["label"]).strip().strip('"').strip("'")
        except Exception:
            pass

    # normalize confidence to float between 0-1
    if "confidence" in normalized:
        conf = normalized.get("confidence")
        try:
            if isinstance(conf, str) and conf.endswith("%"):
                normalized["confidence"] = float(conf.strip().strip("%")) / 100.0
            else:
                normalized["confidence"] = float(conf)
        except Exception:
            normalized["confidence"] = None

    # normalize top_drivers into list of dicts
    if "top_drivers" in normalized:
        td = normalized.get("top_drivers")
        if isinstance(td, str):
            try:
                td_parsed = json.loads(td)
                td = td_parsed
            except Exception:
                try:
                    td_parsed = ast.literal_eval(td)
                    td = td_parsed
                except Exception:
                    td = [td]
        if isinstance(td, dict):
            td = [td]
        if not isinstance(td, list):
            td = [td]
        normalized["top_drivers"] = td

    return normalized


# ---------- Azure call wrappers with retries ----------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=6),
    retry=retry_if_exception_type(Exception),
)
def _call_azure_langchain(prompt: str) -> str:
    model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4")
    azure = AzureChatOpenAI(deployment_name=model_name, temperature=0.0)
    resp = azure.generate([{"role": "user", "content": prompt}])
    try:
        return resp.generations[0][0].text
    except Exception:
        try:
            return resp.generations[0].text
        except Exception:
            return str(resp)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=6),
    retry=retry_if_exception_type(Exception),
)
def _call_openai_direct(prompt: str) -> str:
    # expects AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_MODEL
    import openai

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4")
    if not api_key or not api_base:
        raise RuntimeError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
    openai.api_key = api_key
    openai.api_base = api_base.rstrip("/")
    openai.api_type = "azure"
    if os.environ.get("AZURE_OPENAI_API_VERSION"):
        openai.api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    resp = openai.ChatCompletion.create(
        deployment_id=deployment,
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    try:
        # newer openai returns message.content
        return resp.choices[0].message.content
    except Exception:
        return str(resp)


# call_and_parse: model call + parse + optional repair retries
def call_and_parse(prompt_text: str, model_fn, max_retries: int = 2) -> Optional[dict]:
    prompt = prompt_text
    for attempt in range(max_retries + 1):
        raw = model_fn(prompt)
        parsed = extract_json_from_text(raw)
        normalized = _normalize_parsed(parsed) if parsed else None
        _log_llm("azure", raw, normalized if normalized is not None else parsed)
        if normalized and isinstance(normalized, dict) and normalized.get("label"):
            return normalized
        # append repair instruction and retry
        prompt = prompt + "\n\n" + REPAIR_SUFFIX
    return None


# adjudicate final text
def adjudicate_text_and_return(
    text: str, attributes: dict, model_name: str = "azure"
) -> Dict:
    parsed = extract_json_from_text(text)
    parsed_norm = _normalize_parsed(parsed) if parsed else None
    _log_llm(model_name, text, parsed_norm if parsed_norm is not None else parsed)
    if parsed_norm and isinstance(parsed_norm, dict) and parsed_norm.get("label"):
        return parsed_norm
    # fallback to heuristics
    heur = heuristics_score(attributes)
    return {
        "label": heur["label"],
        "confidence": heur["confidence"],
        "top_drivers": mock_llm_judgement(attributes),
    }


# ---------- main call: calls Azure, parses, falls back ----------
def call_azure_for_judgement(attributes: dict, examples: List[dict] = None) -> Dict:
    examples = examples or []
    prompt = SCORING_PROMPT.format(
        attributes=json.dumps(attributes), examples=json.dumps(examples)
    )
    # prefer LangChain wrapper
    if USE_AZURE_LANGCHAIN:
        try:
            parsed = call_and_parse(prompt, _call_azure_langchain, max_retries=2)
            if parsed:
                return parsed
            # if parsing failed but raw returned, fall back to parsing raw run
            raw = _call_azure_langchain(prompt + "\n\n" + REPAIR_SUFFIX)
            return adjudicate_text_and_return(
                raw, attributes, model_name="azure_langchain"
            )
        except Exception as e:
            logger.exception("Azure LangChain call failed: %s", e)
    # fallback to direct openai SDK
    if USE_OPENAI_SDK:
        try:
            parsed = call_and_parse(prompt, _call_openai_direct, max_retries=2)
            if parsed:
                return parsed
            raw = _call_openai_direct(prompt + "\n\n" + REPAIR_SUFFIX)
            return adjudicate_text_and_return(
                raw, attributes, model_name="azure_openai_sdk"
            )
        except Exception as e:
            logger.exception("Azure openai direct call failed: %s", e)
    # final fallback heuristics
    heur = heuristics_score(attributes)
    drivers = mock_llm_judgement(attributes)
    return {
        "label": heur["label"],
        "confidence": heur["confidence"],
        "top_drivers": drivers,
    }


# ---------- scoring internals with manual cache ----------
def score_row_internal(r: dict) -> Dict:
    # call the LLM pipeline
    try:
        judgement = call_azure_for_judgement(r, examples=[])
    except Exception as e:
        logger.exception("LLM pipeline failed: %s", e)
        judgement = {
            "label": heuristics_score(r)["label"],
            "confidence": heuristics_score(r)["confidence"],
            "top_drivers": mock_llm_judgement(r),
        }

    # normalize and provide safe defaults
    label = judgement.get("label") if isinstance(judgement, dict) else None
    confidence = judgement.get("confidence") if isinstance(judgement, dict) else None
    top_drivers = judgement.get("top_drivers") if isinstance(judgement, dict) else None

    if label is None:
        heur = heuristics_score(r)
        label = heur["label"]
    if confidence is None:
        confidence = heuristics_score(r)["confidence"]
    if top_drivers is None:
        top_drivers = mock_llm_judgement(r)

    # ensure top_drivers serializable
    try:
        td_json = json.dumps(top_drivers)
    except Exception:
        td_json = json.dumps([str(top_drivers)])

    return {
        "client_id": r.get("client_id"),
        "label": label,
        "confidence": (
            round(float(confidence), 2)
            if isinstance(confidence, (int, float, str))
            else 0.5
        ),
        "top_drivers": td_json,
    }


def score_row(row) -> Dict:
    try:
        r = row.to_dict()
    except Exception:
        r = dict(row)

    # cache key: prefer client_id, else hash of row
    cid = r.get("client_id")
    if cid:
        cache_key = f"client:{cid}"
    else:
        cache_key = f"row:{_row_to_hash_key(r)}"

    cached_val = _cache_get(cache_key)
    if cached_val is not None:
        return cached_val

    result = score_row_internal(r)
    _cache_set(cache_key, result)
    return result


def score_batch(df) -> List[Dict]:
    results = []
    for _, r in df.iterrows():
        results.append(score_row(r))
        time.sleep(0.05)
    return results
