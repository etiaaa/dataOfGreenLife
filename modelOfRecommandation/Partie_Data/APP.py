from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import pandas as pd

# Initialiser Flask
app = Flask(__name__)
swagger = Swagger(app)

# Charger le modèle et l’encodeur
model = joblib.load("modele_recommandation.pkl")
encoder = joblib.load("label_encoder.pkl")

# Fonction recommandation globale
def recommandation_globale(score):
    if score < 50:
        return "🌱 Très bien, continuez comme ça !"
    elif score < 100:
        return "⚖️ Moyenne, améliorable."
    else:
        return "🚨 Empreinte élevée, il est temps d'agir !"

# Route principale de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédiction de recommandation CO2
    ---
    parameters:
      - name: input
        in: body
        required: true
        schema:
          type: object
          properties:
            equipements_kg_mois:
              type: number
              example: 6.25
            logement_total_kg_mois:
              type: number
              example: 56.25
            transport_kg_mois:
              type: number
              example: 62.4
    responses:
      200:
        description: Recommandations retournées
    """
    data = request.get_json()
    df = pd.DataFrame([data])
    total = df.sum(axis=1)[0]

    pred = model.predict(df)[0]
    reco_specific = encoder.inverse_transform([pred])[0]
    reco_global = recommandation_globale(total)

    return jsonify({
        "prediction_kgCO2_mois": total,
        "recommandation_specifique": reco_specific,
        "recommandation_globale": reco_global
    })

# Lancer l'app
if __name__ == '__main__':
    app.run(debug=True)
