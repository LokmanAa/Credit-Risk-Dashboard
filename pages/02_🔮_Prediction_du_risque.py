import pandas as pd
import streamlit as st
import requests
import json
import joblib
import plotly.graph_objects as go
import gc

# Fonction de requete
def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

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

# Chargement des données
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.set_index("SK_ID_CURR", inplace=True)
    index = df.index
    columns = df.columns.tolist()
    return df, index, columns

sentences = ['n\'est **pas à risque**', 'est **à risque**']

FLASK_URI = 'https://prediction-bank-a50ad8910f87.herokuapp.com/predict_flask'
AZURE_URI = 'https://bank-app-prediction.azurewebsites.net/predict_flask'
FLASK_LOCAL_URI = 'http://127.0.0.1:8000/predict_flask'

st.set_page_config(page_title="Prédiction")

with st.sidebar:
    st.title('Configuration')
    api_choice = st.selectbox(
        'Quelle API souhaitez vous utiliser', ['Azure', 'Flask', 'Local'])

    client_id = st.sidebar.number_input('Client ID', min_value=0, value=100000, step=1)
    
    predict_btn = st.button('Prédire')

st.header('Prédiction du risque')
proba_plot = st.empty()
proba_text = st.empty()
pred_text = st.empty()

df, index, columns = load_data('./data/X_index.csv')

if predict_btn: # Clic sur le bouton de prédiction
    if client_id not in index:
        st.toast('Le client ID n\'existe pas', icon='⚠️')
    else:
        data = df.loc[df.index == client_id, :].values.tolist()[0]
        json_data = {"dataframe_split": {"columns": columns, "data": [data]}}
        abs_index = index.get_loc(client_id)

        if api_choice == 'Flask':
            prediction = request_prediction(FLASK_URI, json_data)
            proba_plot.plotly_chart(plot_probability(prediction["probability"], prediction["threshold"]))
            proba_text.write(f'La probabilité d\'appartenir à la classe 1 est de **{round(prediction["probability"] * 100, 2)}%**, pour un seuil de décision de **{round(prediction["threshold"] * 100, 2)}%**.\n')
            pred_text.write(f'La classe prédite est donc la classe **{prediction["prediction"]}**, ce qui signifie que, selon la métrique personnalisée, le client {sentences[prediction["prediction"]]}')
        elif api_choice == 'Local':
            prediction = request_prediction(FLASK_LOCAL_URI, json_data)
            proba_plot.plotly_chart(plot_probability(prediction["probability"], prediction["threshold"]))
            proba_text.write(f'La probabilité d\'appartenir à la classe 1 est de **{round(prediction["probability"] * 100, 2)}%**, pour un seuil de décision de **{round(prediction["threshold"] * 100, 2)}%**.\n')
            pred_text.write(f'La classe prédite est donc la classe **{prediction["prediction"]}**, ce qui signifie que, selon la métrique personnalisée, le client {sentences[prediction["prediction"]]}')
        else:
            prediction = request_prediction(AZURE_URI, json_data)
            proba_plot.plotly_chart(plot_probability(prediction["probability"], prediction["threshold"]))
            proba_text.write(f'La probabilité d\'appartenir à la classe 1 est de **{round(prediction["probability"] * 100, 2)}%**, pour un seuil de décision de **{round(prediction["threshold"] * 100, 2)}%**.\n')
            pred_text.write(f'La classe prédite est donc la classe **{prediction["prediction"]}**, ce qui signifie que, selon la métrique personnalisée, le client {sentences[prediction["prediction"]]}')
            
        st.dataframe(df.loc[df.index == client_id, :])
        del data, json_data, abs_index
        gc.collect()
    