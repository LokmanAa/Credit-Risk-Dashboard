import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import gc

st.set_page_config(page_title="Features Importance", layout="wide")

@st.cache_data
def load_data():
    mean_values = joblib.load("data/mean_values.pkl")
    index = joblib.load("data/index.pkl")
    feature_names = joblib.load("data/feature_names.pkl")
    return mean_values, feature_names, index

# Chargement des valeurs de SHAP
def load_local_values(index):
    values = joblib.load("data/values.pkl")
    abs_index = index[index == client_id].index
    local_values = pd.DataFrame(values[abs_index].T, index=feature_names, columns=["values"])
    del values
    gc.collect()
    local_values["absolute_values"] = local_values["values"].abs()
    local_values = local_values.sort_values("absolute_values", ascending=False).reset_index().head(10)
    return local_values

mean_values, feature_names, index = load_data()

with st.sidebar:
    st.title('Configuration')
    client_id = st.number_input('Client ID', min_value=0, value=100000, step=1)

col_left, col_center, col_right = st.columns([1.8, 0.2, 1.8])

with col_left:
    st.header('Features Importance Globale')
    st.altair_chart(alt.Chart(mean_values).mark_bar(color="#d55f01").encode(
        x=alt.X('mean_values', title='Mean values'),
        y=alt.Y('index', title='Features', sort='-x')
    ).properties(width=450, height=400))
    
with col_right:
    st.header('Features Importance Client')
    local_chart = st.markdown("<div style='display: flex; justify-content: center; align-items: center; height: 300px; font-size: 16px;'>Aucun client sélectionné pour le moment</div>", unsafe_allow_html=True)
    
if client_id:
    if client_id not in index.values:
        local_chart.error('Le client ID n\'existe pas')
    else: # Utilisation d'Altair pour un graphique interactif
        local_chart.altair_chart(alt.Chart(load_local_values(index)).mark_bar().encode(
        x=alt.X('values', title='Values'),
        y=alt.Y('index', title='Features', sort=alt.SortField('abs_values', order='descending')),
        color=alt.condition(
                alt.datum.values > 0,
                alt.value("#d55f01"),
                alt.value("#0172b3")
            )
    ).properties(width=450, height=400))