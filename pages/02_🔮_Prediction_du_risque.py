import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import gc
import requests
import pickle

# Affichage de la jauge de risque
def plot_probability(probability, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=0,
            gauge={
                "axis": {"visible": True, "range": [None, 1]},
                "steps": [{"range": [0, probability], "color": "#d55f01"}],
                "threshold": {
                    "line": {"color": "#0172b3", "width": 5},
                    "thickness": 1,
                    "value": threshold,
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

# Chargement des données depuis un repository GitHub
@st.cache_data
def load_data_from_github():
    url = 'https://raw.githubusercontent.com/LokmanAa/Credit-Risk-Dashboard/main/data/X_index.csv'
    df = pd.read_csv(url)
    df.set_index("SK_ID_CURR", inplace=True)
    index = df.index
    columns = df.columns.tolist()
    return df, index, columns

# Chargement du modèle depuis GitHub
@st.cache_resource
def load_model_from_github():
    model_url = 'https://github.com/LokmanAa/Credit-Risk-Dashboard/blob/main/models/model.pkl?raw=true'
    response = requests.get(model_url)
    with open('model.pkl', 'wb') as f:
        f.write(response.content)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)  # Utilisation de pickle pour charger le modèle au format pkl
    return model

# Prédiction avec le modèle chargé
def predict(model, data):
    return model.predict_proba([data])[0]

sentences = ['n\'est **pas à risque**', 'est **à risque**']

st.set_page_config(page_title="Prédiction")

with st.sidebar:
    st.title('Configuration')
    client_id = st.sidebar.number_input('Client ID', min_value=0, value=100000, step=1)
    predict_btn = st.button('Prédire')

st.header('Prédiction du risque')
proba_plot = st.empty()
proba_text = st.empty()
pred_text = st.empty()

df, index, columns = load_data_from_github()
model = load_model_from_github()

if predict_btn:  # Clic sur le bouton de prédiction
    if client_id not in index:
        st.toast('Le client ID n\'existe pas', icon='⚠️')
    else:
        data = df.loc[df.index == client_id, :].values.tolist()[0]
        prediction_prob = predict(model, data)
        probability = prediction_prob[1]  # Probabilité d'appartenir à la classe 1
        threshold = 0.27  # Exemple de seuil de décision, à ajuster si besoin

        proba_plot.plotly_chart(plot_probability(probability, threshold))
        proba_text.write(f'La probabilité d\'appartenir à la classe 1 est de **{round(probability * 100, 2)}%**, pour un seuil de décision de **{round(threshold * 100, 2)}%**.\n')
        pred_class = int(probability >= threshold)
        pred_text.write(f'La classe prédite est donc la classe **{pred_class}**, ce qui signifie que, selon la métrique personnalisée, le client {sentences[pred_class]}')

        st.dataframe(df.loc[df.index == client_id, :])
        
        del data, prediction_prob
        gc.collect()

