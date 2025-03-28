# Modèle_Recommandation.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Fonction pour générer une recommandation globale selon le total CO₂
def recommandation_globale(score):
    if score < 50:
        return "🌱 Très bien, continuez comme ça !"
    elif score < 100:
        return "⚖️ Vous êtes dans la moyenne, mais vous pouvez encore réduire vos émissions."
    else:
        return "🚨 Votre empreinte est élevée, il est temps de passer à l’action !"

# 1. Charger le dataset d'entraînement
df = pd.read_csv("reco_training_dataset.csv")

# 2. Définir les colonnes d'entrée (features) et la colonne de sortie (target)
X = df[["equipements_kg_mois", "logement_total_kg_mois", "transport_kg_mois"]]
y = df["recommandation"]

# 3. Encoder les recommandations (texte → entier)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 4. Diviser le dataset pour entraîner et tester le modèle
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Créer et entraîner le modèle
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 6. Évaluer la précision du modèle (optionnel)
score = model.score(X_test, y_test)
print(f"✅ Précision du modèle : {score:.2f}")

# 7. Tester une prédiction avec de nouvelles données
new_data = pd.DataFrame([
    {
        "equipements_kg_mois": 6.25,
        "logement_total_kg_mois": 56.25,
        "transport_kg_mois": 62.4
    },
    {
        "equipements_kg_mois": 3.0,
        "logement_total_kg_mois": 25.0,
        "transport_kg_mois": 15.0
    }
])

# 8. Calcul du score CO2 total
new_data["prediction_kgCO2_mois"] = new_data.sum(axis=1)

# 9. Prédiction spécifique via ML
predictions = model.predict(new_data[["equipements_kg_mois", "logement_total_kg_mois", "transport_kg_mois"]])
reco_results = encoder.inverse_transform(predictions)

# 10. Générer les recommandations globales
global_recos = new_data["prediction_kgCO2_mois"].apply(recommandation_globale)

# 11. Afficher les recommandations combinées
for i, (specific, global_msg) in enumerate(zip(reco_results, global_recos)):
    print(f"\n🔎 Utilisateur {i+1} → Recommandation spécifique : {specific}")
    print(f"💬 Recommandation globale : {global_msg}")

# 12. Sauvegarder le modèle et l'encodeur
joblib.dump(model, "modele_recommandation.pkl")
joblib.dump(encoder, "label_encoder.pkl")
