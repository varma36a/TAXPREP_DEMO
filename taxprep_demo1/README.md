# TaxPrep Demo — Streamlit + GenAI Agentic Scaffold (No Azure Functions)

This demo contains:
- Streamlit app (`app.py`) — upload CSV, run scoring, download results
- Scoring service (`scoring_service.py`) with LangChain/Azure+Anthropic scaffolding and graceful fallback to mock
- Synthetic data generator (`data_utils.py`)
- Dockerfile and GitHub Actions workflow for CI/CD (build & push + deploy to Azure Container Instances template)
- Jupyter notebook (notebooks/demo_taxprep.ipynb)
- PowerPoint slide deck (TaxPrep_Demo_Deck.pptx)

## Run locally (mock LLM)
1. Create virtualenv: `python -m venv .venv && source .venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## To enable real LLMs (requires credentials)
- Set environment variables:
  - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL (deployment name)
  - ANTHROPIC_API_KEY, ANTHROPIC_MODEL (optional)
- The `scoring_service.py` will attempt to call AzureChatOpenAI and AnthropicChat via LangChain; if calls fail it will fall back to the mock heuristics.
