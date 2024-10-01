import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    index = df["SK_ID_CURR"].values.tolist()
    columns = df.columns.tolist()
    return df, index, columns

@st.cache_data
def cat_or_num(df):  # Retourne les colonnes catégorielles et numériques d'un dataframe
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    cat += df.columns[(df.nunique() == 2) & (df.dtypes != "object")].tolist()

    num = df.columns[(df.nunique() > 2) & (df.dtypes != "object")].tolist()
    if "TARGET" in cat:
        cat.remove("TARGET")
    if "SK_ID_CURR" in num:
        num.remove("SK_ID_CURR")
    return cat, num

# Fonction pour afficher la jauge de corrélation
def plot_correlation(correlation):
    if correlation < 0:
        color = "#0172b3"
    else:
        color = "#d55f01"
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=correlation,
            gauge={
                "axis": {"visible": True, "range": [-1, 1]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [{"range": [0, correlation], "color": color}],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 1,
                    "value": 0,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]}
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=50),
        height=280
    ) 
    return fig

def encode(series):
    return pd.Series(LabelEncoder().fit_transform(series))

st.set_page_config(page_title="Analyse des features", layout="wide")

df, index, columns = load_data("./data/X_index.csv")
cat, num = cat_or_num(df)

st.title("Analyse des features")

with st.sidebar:
    st.title("Configuration")
    feature1 = st.selectbox("Feature 1 :", columns)
    feature2 = st.selectbox("Feature 2 :", columns)
    if feature1 == "SK_ID_CURR" or feature2 == "SK_ID_CURR":
        st.warning("Veuillez sélectionner une feature autre que SK_ID_CURR.")
        st.stop()
    button = st.button("Analyser")


    
if button: # Clic sur le bouton
    if (feature1 in cat and feature2 in cat) or (feature1 in num and feature2 in num):
        if feature1 in cat:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_correlation(encode(df[feature1])
                                     .corr(encode(df[feature2])))
                    )
            with col2:
                confusion_matrix = pd.crosstab(encode(df[feature1]), encode(df[feature2]))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix, annot=True, fmt="d", ax=ax)
                ax.set_xlabel(feature2)
                ax.set_ylabel(feature1)
                st.pyplot(fig)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_correlation(encode(df[feature1])
                                     .corr(encode(df[feature2])))
                    )
            with col2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=feature1, y=feature2, data=df, ax=ax, palette="colorblind")
                st.pyplot(fig)
    
    else:
        if feature1 in cat:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_correlation(encode(df[feature1])
                                     .corr(df[feature2]))
                    )
            with col2:
                fig, ax = plt.subplots()
                sns.violinplot(x=feature2, y=feature1, data=df, ax=ax, palette="colorblind", orient="h")
                st.pyplot(fig)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_correlation(df[feature1]
                                     .corr(encode(df[feature2])))
                    )
            with col2:
                fig, ax = plt.subplots()
                sns.violinplot(x=feature1, y=feature2, data=df, ax=ax, palette="colorblind", orient="h")
                st.pyplot(fig)
            
    