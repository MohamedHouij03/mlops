
import mlflow
import mlflow.sklearn
import pandas as pd

# ===== 1. Connexion à la base MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 2. Charger le modèle champion =====
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Modèle chargé :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# ===== 3. Créer un nouveau client =====
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# ===== 4. Adapter les colonnes au modèle =====
client = client[list(model.feature_names_in_)]

# ===== 5. Prédire le départ du client =====
prediction = model.predict(client)[0]

print("Le client va-t-il partir ? ->", "OUI" if prediction == 1 else "NON")
