import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_boxplot(feature, ax):
    fig, ax = plt.subplots()
    sns.boxplot(df[feature], ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Nombre")
    return fig

st.set_page_config(page_title="Analyse clients", layout="wide")

df, index, columns = load_data("./data/X_index.csv")
cat, num = cat_or_num(df)

st.title("Analyse des clients")

with st.sidebar:
    st.title("Configuration")
    client_id = st.number_input("Client ID", min_value=0, value=100000, step=1)
    if client_id:
        if client_id not in index:
            st.warning("Client ID invalide.")
            st.stop()
        else:
            client_data = df.loc[df["SK_ID_CURR"] == client_id]
    type_switch = st.radio("Type de variable", ["Numérique", "Catégorielle"])

if type_switch == "Numérique":
    selected_features = st.multiselect("Features à afficher :", num)
else:
    selected_features = st.multiselect("Features à afficher :", cat)

if len(selected_features) == 0:
    st.warning("Veuillez sélectionner au moins une feature.")
    st.stop()
else:
    num_plots = len(selected_features)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows == 1:
        axes = axes[np.newaxis, :]
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    # Tracé des boxplots
    for i, feature in enumerate(selected_features):
        row = i // num_cols
        col = i % num_cols
        if type_switch == "Numérique":
            sns.boxplot(y=df[feature], ax=axes[row, col], palette="colorblind")
            axes[row, col].scatter(y=client_data[feature], x=[0], color='#d55f01', marker='o', linewidth=4, zorder=10, label='Client')
            axes[row, col].legend()
        else: # Tracé d'un countplot
            sns.countplot(y=df[feature], ax=axes[row, col], palette="colorblind")
            client_data = df.loc[df["SK_ID_CURR"] == client_id]
            client_feature_value = client_data[feature].values[0]

            # Ajouter un marqueur sur la barre correspondante au client
            counts = df[feature].value_counts()
            client_count = counts[client_feature_value]
            y_position = counts.index.tolist().index(client_feature_value)
            axes[row, col].scatter(client_count, y_position, color='#f04dc1', zorder=10, s=200, label='Client')
            axes[row, col].legend()
        axes[row, col].set_title(feature)

    # Suppression des sous-graphiques vides
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flat[j])

    st.pyplot(fig)