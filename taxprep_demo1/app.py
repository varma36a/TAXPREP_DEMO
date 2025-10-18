# app.py (updated)
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import json
from data_utils import load_or_generate_sample
from scoring_service_azure import score_batch
# put this at top of app.py or run in notebook before imports

import logging
from pathlib import Path

# ensure logs folder
Path("logs").mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s"
logging.basicConfig(
    level=logging.DEBUG,  # show DEBUG and above
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),  # file
    ],
)
# optionally reduce noise from third-party libs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO)


# Load .env (if present)
load_dotenv()

st.set_page_config(page_title="TaxPrep Client Experience — Demo", layout="wide")
st.title("TaxPrep Client Experience — GenAI Agent Demo")

# --- Sidebar: data upload / generate ---
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload customersatisfaction.csv", type=["csv"])
    if uploaded is None:
        st.info("No file uploaded. Use 'Generate sample dataset' or upload your CSV.")
        if st.button("Generate sample dataset"):
            df = load_or_generate_sample(200)
            st.session_state["df"] = df
    else:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["df"] = df
            st.success(f"Loaded {len(df)} rows from uploaded file.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

# --- Main view ---
if "df" not in st.session_state:
    st.info("No dataset loaded yet. Upload or generate a sample in the sidebar.")
    st.stop()

df = st.session_state["df"]

# Show top of data
st.subheader("Raw dataset (preview)")
st.dataframe(df.head(50))

# Run scoring
if st.button("Run GenAI Scoring"):
    with st.spinner("Scoring batch — calling Azure OpenAI... (this may take a while)"):
        try:
            results = score_batch(df)
            results_df = pd.DataFrame(results)

            # parse top_drivers JSON string into readable column
            def parse_drivers(x):
                try:
                    items = x
                    if isinstance(x, str):
                        items = json.loads(x)
                    if isinstance(items, list):
                        return " | ".join(
                            [
                                f"{d.get('factor')}({d.get('impact')}): {d.get('explain')}"
                                for d in items
                            ]
                        )
                    return str(items)
                except Exception:
                    return str(x)

            results_df["top_drivers_readable"] = results_df["top_drivers"].apply(
                parse_drivers
            )
            st.session_state["results_df"] = results_df
            st.success("Scoring complete — results are ready.")
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            st.stop()

# If results available, show them and visuals
if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]

    st.subheader("Scoring results (preview)")
    st.dataframe(results_df.head(100))

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total scored", len(results_df))
    avg_conf = results_df["confidence"].astype(float).mean()
    col2.metric("Average confidence", f"{avg_conf:.2f}")
    dissatisfied_count = (results_df["label"] == "Dissatisfied").sum()
    col3.metric("Predicted Dissatisfied", int(dissatisfied_count))

    # Confidence distribution chart
    st.subheader("Confidence distribution")
    st.bar_chart(results_df["confidence"].astype(float).rename("confidence"))

    # Show readable drivers and low-confidence review bucket
    st.subheader("Top drivers (sample)")
    st.dataframe(
        results_df[["client_id", "label", "confidence", "top_drivers_readable"]].head(
            100
        )
    )

    # Flag low confidence for review
    review_threshold = st.slider("Mark for review if confidence <=", 0.0, 1.0, 0.60)
    to_review = results_df[results_df["confidence"].astype(float) <= review_threshold]
    st.markdown(
        f"**{len(to_review)}** records flagged for review (confidence ≤ {review_threshold})"
    )
    if not to_review.empty:
        st.dataframe(
            to_review[["client_id", "label", "confidence", "top_drivers_readable"]]
        )

    # Download buttons
    csv = results_df.to_csv(index=False)
    st.download_button(
        "Download full results CSV",
        csv,
        file_name="scoring_results.csv",
        mime="text/csv",
    )

    if not to_review.empty:
        csv_review = to_review.to_csv(index=False)
        st.download_button(
            "Download review CSV",
            csv_review,
            file_name="scoring_review.csv",
            mime="text/csv",
        )

else:
    st.info("No scoring results yet. Click 'Run GenAI Scoring' to start.")
