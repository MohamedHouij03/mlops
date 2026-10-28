
import mlflow
from mlflow import MlflowClient

# ===== 1. Connexion à MLflow =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# ===== 2. Retrouver le run essai-2 =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

if runs.empty:
    raise ValueError("Aucun run nommé essai-2 n'a été trouvé.")

run_id = runs.iloc[0]["run_id"]

# ===== 3. Enregistrer une nouvelle version du modèle =====
version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model"
)

print("Nouvelle version créée :", version.version)

# ===== 4. Afficher l'alias champion actuel =====
current = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print("Champion actuel : version", current.version)

# ===== 5. Déplacer temporairement l'alias =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    version.version
)

print(
    "Après déplacement : champion -> version",
    client.get_model_version_by_alias(
        "churn-model", "champion"
    ).version
)

# ===== 6. Restaurer l'alias initial =====
client.set_registered_model_alias(
    "churn-model",
    "champion",
    current.version
)

print(
    "Après retour : champion -> version",
    client.get_model_version_by_alias(
        "churn-model", "champion"
    ).version
)
