import mlflow  # Lire les résultats enregistrés dans MLflow

# Rechercher tous les runs de l'expérience et les trier
# par score F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Colonnes à afficher
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

# Vérifier si des résultats existent
if runs.empty:
    print("Aucun run trouvé dans l'expérience churn-prediction.")
else:
    # Afficher le tableau des résultats
    print(runs[colonnes].to_string(index=False))

    # Récupérer le meilleur run
    meilleur = runs.iloc[0]

    print()

    # Afficher le nom et l'identifiant du meilleur run
    print(
        "Meilleur run :",
        meilleur["tags.mlflow.runName"],
        "| id :",
        meilleur["run_id"][:8],
        "..."
    )
