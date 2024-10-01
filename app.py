from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load('model.pkl')

@app.route('/predict_flask', methods=['POST'])
def predict():
    try:
        # Extraire les données de la requête
        data = request.json
        dataframe_split = data['dataframe_split']

        # Recréer le DataFrame
        df = pd.DataFrame(data=dataframe_split['data'], columns=dataframe_split['columns'])

        # Effectuer la prédiction
        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.5)  # Exemple : seuil à 0.5

        # Retourner la prédiction sous forme de JSON
        return jsonify({
            "probability": probability,
            "prediction": prediction,
            "threshold": 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
