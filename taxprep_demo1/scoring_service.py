import time
import os
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import LangChain LLM wrappers; if unavailable, we'll fallback to local heuristics/mock LLM behavior.
USE_REAL_LLM = False
try:
    from langchain import LLMChain, PromptTemplate  # optional usage
    from langchain.chat_models import AzureChatOpenAI, AnthropicChat
    USE_REAL_LLM = True
except Exception as e:
    logger.info(
        "LangChain or model packages not available or not configured; falling back to mock. (%s)",
        e,
    )
    USE_REAL_LLM = False


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
    drivers: List[Dict] = []
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


# Prompt template for LLM scoring (structured JSON output)
SCORING_PROMPT = """You are an analyst that receives structured client attributes and must decide if the client is 'Satisfied' or 'Dissatisfied'.
Return a JSON object exactly with keys: label (Satisfied|Dissatisfied), confidence (0-1), top_drivers (list of {factor,impact,explain}).
Client: {attributes}
PeerExamples: {examples}
"""


def call_llms_for_judgement(attributes: dict, examples: List[dict] = None) -> Dict:
    examples = examples or []
    if USE_REAL_LLM:
        # Build prompt and call Azure Chat + Anthropic and perform simple adjudication.
        prompt = SCORING_PROMPT.format(
            attributes=json.dumps(attributes), examples=json.dumps(examples)
        )
        responses = []
        try:
            # Azure OpenAI Chat (AzureChatOpenAI) -- expects env AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and model name in AZURE_OPENAI_MODEL
            azure_model = AzureChatOpenAI(
                deployment_name=os.environ.get("AZURE_OPENAI_MODEL", "gpt-4"),
                temperature=0,
            )
            azure_resp = azure_model.generate([{"role": "user", "content": prompt}])
            # langchain ChatModel generate shape can vary; guard access
            azure_text = ""
            try:
                azure_text = azure_resp.generations[0][0].text
            except Exception:
                try:
                    # older LangChain versions
                    azure_text = azure_resp.generations[0].text
                except Exception:
                    azure_text = str(azure_resp)
            responses.append(("azure", azure_text))
        except Exception as e:
            logger.exception("Azure model call failed: %s", e)

        # fallback to heuristics if parsing fails
        return {
            "label": "Dissatisfied",
            "confidence": 0.5,
            "top_drivers": [
                {
                    "factor": "parsing_failure",
                    "impact": "High",
                    "explain": "LLM response could not be parsed",
                }
            ],
        }
    else:
        # Mock behavior for local demo/testing
        heur = heuristics_score(attributes)
        drivers = mock_llm_judgement(attributes)
        return {
            "label": heur["label"],
            "confidence": heur["confidence"],
            "top_drivers": drivers,
        }


def score_row(row) -> Dict:
    try:
        r = row.to_dict()
    except Exception:
        r = dict(row)
    # call LLMs with a small set of peer examples (in production: use vector DB nearest neighbors)
    examples = []
    judgement = call_llms_for_judgement(r, examples)
    # Ensure output fields exist
    label = judgement.get("label") or heuristics_score(r)["label"]
    confidence = judgement.get("confidence") or heuristics_score(r)["confidence"]
    top_drivers = judgement.get("top_drivers") or mock_llm_judgement(r)
    return {
        "client_id": r.get("client_id"),
        "label": label,
        "confidence": (
            round(float(confidence), 2)
            if isinstance(confidence, (int, float, str))
            else 0.5
        ),
        "top_drivers": json.dumps(top_drivers),
    }


def score_batch(df) -> List[Dict]:
    results = []
    for _, r in df.iterrows():
        results.append(score_row(r))
        time.sleep(0.01)
    return results
