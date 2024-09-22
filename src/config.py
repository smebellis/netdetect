from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

# Define the parameter grid
param_grid = {
    "n_estimators": [50, 75, 100],
    "max_samples": [0.25, 0.5, 0.75],
    "max_depth": [2, 4, 6],
    "criterion": ["gini", "entropy"],
}

# Define the scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "recall": make_scorer(recall_score, average="weighted"),
    "precision": make_scorer(precision_score, average="weighted"),
    "f1": make_scorer(f1_score, average="weighted"),
}
