import streamlit as st
from data_utils import load_or_generate_sample
from scoring_service import score_batch
import pandas as pd

st.set_page_config(page_title='TaxPrep Client Experience — Demo')
st.title('TaxPrep Client Experience — GenAI Agent Demo')

uploaded = st.file_uploader('Upload customersatisfaction.csv', type=['csv'])
if uploaded is None:
    st.info('No file uploaded. Using synthetic sample dataset — click Generate to create one.')
    if st.button('Generate sample dataset'):
        df = load_or_generate_sample(200)
        st.session_state['df'] = df
else:
    df = pd.read_csv(uploaded)
    st.session_state['df'] = df

if 'df' in st.session_state:
    st.dataframe(st.session_state['df'].head(50))
    if st.button('Run GenAI Scoring'):
        with st.spinner('Scoring batch — calling LLMs...'):
            results = score_batch(st.session_state['df'])
        st.success('Scoring complete')
        st.dataframe(pd.DataFrame(results))
        st.download_button('Download results CSV', pd.DataFrame(results).to_csv(index=False), file_name='scoring_results.csv')
