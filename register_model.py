
# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Se connecter à la même base que le TP3 =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Retrouver le meilleur run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty:
    raise ValueError("Aucun run trouvé dans l'expérience churn-prediction.")

best = runs.iloc[0]
run_id = best["run_id"]

print(
    "Meilleur run :",
    best.get("tags.mlflow.runName", "Sans nom"),
    "| id :",
    run_id[:8],
    "..."
)

# ===== 4. Enregistrer le modèle dans le Registry =====
model_uri = f"runs:/{run_id}/model"
model_name = "churn-model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(
    "Modèle enregistré :",
    model_name,
    "| version",
    result.version
)

# ===== 5. Définir l'alias « champion » =====
client = MlflowClient()

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)
