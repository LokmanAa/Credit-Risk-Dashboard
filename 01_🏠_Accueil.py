import streamlit as st

st.title("Dashboard d'Analyse des Données Clients")

st.markdown("Ce dashboard permet d'explorer et de comprendre les données des clients de la société, afin d'évaluer leur risque d'attribution de crédit et les facteurs qui l'influencent. Les réglages visuels se trouvent dans le menu en haut à droite de l'écran")

st.header("Sections et fonctionnalités")

st.markdown("""
            - **Prédiction du risque** : visualiser le score de crédit attribué à chaque client, accompagné d'une recommandation en conséquence.
            - **Importance des variables** : comprendre les variables qui influencent le plus l'évaluation du risque au global, et spécifiquement pour un client donné.
            - **Analyse du client** : comparer les caractéristiques d'un client à celles des autres via des analyses univariées.
            - **Analyse des variables** : explorer les relations entre les variables par le biais de graphiques et de statistiques bivariés.
            
            Vous trouverez dans la barre de gauche les différentes pages correpspondantes, ainsi que les paramètres de configuration.
            
            
            """)
st.text("")
st.text("")
st.caption("Ce dashboard est hébergé en ligne grâce à Microsoft Azure et déployé vvia Github Actions.")