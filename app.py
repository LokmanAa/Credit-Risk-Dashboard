import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Charger le modèle
model_path = 'models/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict_flask', methods=['POST'])
def predict():
    if not request.json or 'features' not in request.json:
        return jsonify({"error": "Bad Request", "message": "Features are required."}), 400
    
    features = request.json['features']  # Suppose que les données d'entrée sont sous la clé 'features'
    
    # Faire la prédiction
    prediction = model.predict([features])  # Le modèle attend une liste de caractéristiques

    # Retourner la prédiction
    return jsonify({"prediction": prediction.tolist()})  # Convertir en liste pour JSON

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
