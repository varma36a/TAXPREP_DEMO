import os
import json
import time
import logging
import re
import hashlib
import ast
from typing import List, Dict, Optional
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache

# ----------------------------
# Logging Setup
# ----------------------------
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LLM_LOG_FILE = LOG_DIR / "llm_responses.log"

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s")
    )
    logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


# ---------------------------
# Heuristics Fallback & Mock
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
            {"factor": "turnaround_time_days", "impact": "High", "explain": f"{row.get('turnaround_time_days')} days vs SLA 7"}
        )
    if row.get("error_rate_pct", 0) > 2:
        drivers.append(
            {"factor": "error_rate_pct", "impact": "Medium", "explain": f"{row.get('error_rate_pct')}% error rate"}
        )
    if not drivers:
        drivers.append({"factor": "communication_count", "impact": "Low", "explain": "regular communication"})
    return drivers


# ---------------------------
# Azure OpenAI Setup
# ---------------------------
USE_AZURE_LANGCHAIN = False
USE_OPENAI_SDK = False
AzureChatOpenAI = None

try:
    from langchain_openai import AzureChatOpenAI
    USE_AZURE_LANGCHAIN = True
    logger.info("Imported AzureChatOpenAI from langchain_openai")
except Exception as e:
    logger.warning("LangChain AzureChatOpenAI not available: %s", e)

if not USE_AZURE_LANGCHAIN:
    try:
        import openai
        USE_OPENAI_SDK = True
        logger.info("Using OpenAI SDK fallback")
    except Exception as e:
        logger.warning("OpenAI SDK not available: %s", e)


# ---------------------------
# Prompt Instructions
# ---------------------------
SCORING_PROMPT_INSTRUCTIONS = (
    "You are an automated classification assistant for client satisfaction.\n"
    "Follow these rules exactly:\n\n"
    "1. Output only ONE JSON object, no text before or after it.\n"
    "2. Use this exact schema:\n"
    "{\n"
    '  "label": "Satisfied" or "Dissatisfied",\n'
    '  "confidence": number between 0.0 and 1.0,\n'
    '  "top_drivers": [{"factor": "string", "impact": "High|Medium|Low", "explain": "string"}]\n'
    "}\n\n"
    "3. Use double quotes for all keys and values.\n"
    "4. Do not include markdown, explanation, or comments.\n"
    "5. If unsure, output this exact JSON:\n"
    '{"label": "Dissatisfied", "confidence": 0.5, "top_drivers": [{"factor": "unclear", "impact": "Low", "explain": "insufficient data"}]}\n'
)


REPAIR_SUFFIX = (
    "REPAIR: Your previous output was not valid JSON. Return ONLY valid JSON matching schema: "
    '{"label":"Satisfied|Dissatisfied","confidence":0.0-1.0,'
    '"top_drivers":[{"factor":"...","impact":"High|Medium|Low","explain":"..."}]}'
)


# ---------------------------
# Cache Setup
# ---------------------------
CACHE_TTL_SECONDS = 60 * 5
_cache = TTLCache(maxsize=2000, ttl=CACHE_TTL_SECONDS)


def _row_to_hash_key(row_dict: dict) -> str:
    try:
        normalized = json.dumps(row_dict, sort_keys=True, default=str)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(str(row_dict).encode("utf-8")).hexdigest()


# ---------------------------
# JSON Parsing Utils
# ---------------------------
def extract_json_from_text(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        block = match.group(0)
        for candidate in (block, block.replace("'", '"')):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        try:
            parsed = ast.literal_eval(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _normalize_parsed(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return parsed

    norm = dict(parsed)
    if "confidence" in norm:
        try:
            norm["confidence"] = float(norm["confidence"])
        except Exception:
            norm["confidence"] = 0.5

    if isinstance(norm.get("top_drivers"), str):
        try:
            norm["top_drivers"] = json.loads(norm["top_drivers"])
        except Exception:
            norm["top_drivers"] = []

    if not isinstance(norm.get("top_drivers"), list):
        norm["top_drivers"] = [norm["top_drivers"]]

    return norm


# ---------------------------
# Azure Calls
# ---------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
def _call_azure_langchain(prompt: str) -> str:
    if not USE_AZURE_LANGCHAIN:
        raise RuntimeError("LangChain not available")

    model_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    llm = AzureChatOpenAI(
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        openai_api_version=api_version,
        openai_api_key=api_key,
        max_tokens=1500,  # Increased for safety
        # Removed temperature to fix "unsupported_value"
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content if response else ""


# ---------------------------
# LLM Call + Parse
# ---------------------------
def call_and_parse(prompt_text: str, model_fn, max_retries: int = 2) -> Optional[dict]:
    prompt = prompt_text
    for attempt in range(max_retries + 1):
        try:
            raw = model_fn(prompt)
            if not raw or not raw.strip():
                logger.warning(f"Attempt {attempt}: Empty model response")
                prompt += "\n\n" + REPAIR_SUFFIX
                continue

            logger.debug(f"Raw model output (first 200 chars): {repr(raw[:200])}")
            parsed = extract_json_from_text(raw)
            normalized = _normalize_parsed(parsed) if parsed else None

            if normalized and normalized.get("label"):
                return normalized

            logger.warning(f"Attempt {attempt}: Failed to parse JSON, retrying")
            prompt += "\n\n" + REPAIR_SUFFIX

        except Exception as e:
            logger.error(f"Call attempt {attempt} failed: {e}")
            prompt += "\n\n" + REPAIR_SUFFIX

    return None


# ---------------------------
# Core Judgement Logic
# ---------------------------
def call_azure_for_judgement(attributes: dict, examples: List[dict] = None) -> Dict:
    examples = examples or []
    prompt = (
        SCORING_PROMPT_INSTRUCTIONS
        + "\nClient: "
        + json.dumps(attributes)
        + "\nPeerExamples: "
        + json.dumps(examples)
    )

    try:
        parsed = call_and_parse(prompt, _call_azure_langchain, max_retries=2)
        if parsed:
            return parsed
    except Exception as e:
        logger.exception("Azure LangChain call failed: %s", e)

    return fallback_response(attributes)


def fallback_response(attributes: dict) -> Dict:
    heur = heuristics_score(attributes)
    return {
        "label": heur["label"],
        "confidence": heur["confidence"],
        "top_drivers": mock_llm_judgement(attributes),
    }


# ---------------------------
# Batch + Caching
# ---------------------------
def score_row_internal(r: dict) -> Dict:
    try:
        judgement = call_azure_for_judgement(r, examples=[])
    except Exception as e:
        logger.exception("LLM pipeline failed: %s", e)
        judgement = fallback_response(r)

    label = judgement.get("label")
    confidence = judgement.get("confidence")
    top_drivers = judgement.get("top_drivers")

    return {
        "client_id": r.get("client_id"),
        "label": label,
        "confidence": round(float(confidence), 2),
        "top_drivers": json.dumps(top_drivers, ensure_ascii=False),
    }


def score_row(row) -> Dict:
    try:
        r = row.to_dict()
    except Exception:
        r = dict(row)

    cid = r.get("client_id")
    key = f"client:{cid}" if cid else f"row:{_row_to_hash_key(r)}"

    if key in _cache:
        return _cache[key]

    result = score_row_internal(r)
    _cache[key] = result
    return result


def score_batch(df) -> List[Dict]:
    results = []
    for _, row in df.iterrows():
        results.append(score_row(row))
        time.sleep(0.05)
    return results
