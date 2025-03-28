from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("model_emission_carbone.pkl")

# Décomposition manuelle des émissions
def decomposer_emissions(user_row):
    co2_par_km = {"voiture": 120, "velo": 5, "marche": 0}
    coef_chauffage = {"electrique": 20, "gaz": 25, "pompe à chaleur": 10}

    km_annee = user_row["transport_km_semaine"] * 52
    transport_kg_an = km_annee * co2_par_km.get(user_row["transport_mode"], 120) / 1000

    equip_kg_an = (
        user_row["ampoules"] * 10 +
        user_row["lave_linge"] * 25 +
        user_row["lave_vaisselle"] * 20 +
        user_row["four"] * 30 +
        user_row["climatisation"] * 150
    )

    chauffage_kg_an = user_row["surface_m2"] * coef_chauffage.get(user_row["chauffage"], 20)
    logement_total_an = (chauffage_kg_an + equip_kg_an) / max(user_row["nb_personnes"], 1)

    return {
        "transport_kg_mois": round(transport_kg_an / 12, 2),
        "equipements_kg_mois": round(equip_kg_an / 12 / max(user_row["nb_personnes"], 1), 2),
        "logement_total_kg_mois": round(logement_total_an / 12, 2)
    }

# Lancer l'app Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convertir en DataFrame pour le modèle
    df_input = pd.DataFrame([data])
    prediction = model.predict(df_input)[0]
    details = decomposer_emissions(df_input.iloc[0])

    return jsonify({
        "prediction_kgCO2_mois": round(prediction, 2),
        "details": details
    })

if __name__ == '__main__':
    app.run(debug=True)
