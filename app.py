from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict_flask', methods=['POST'])
def predict():
    # Récupère les données de la requête
    data = request.json
    # Effectue ici ta logique de prédiction
    # Par exemple, un modèle de prédiction fictif
    prediction = {"result": "Ceci est un exemple de prédiction."}
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
