import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Config:
    # Constants
    DEFAULT_TEST_SIZE: float = 0.3
    DEFAULT_RANDOM_STATE: int = 42
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.01

    # Parameter grids
    # PARAM_GRID_RF: Dict[str, Any] = field(
    #     default_factory=lambda: {
    #         "n_estimators": [50, 75, 100],
    #         "max_samples": [0.25, 0.5, 0.75],
    #         "max_depth": [2],
    #         "criterion": ["gini", "entropy"],
    #     }
    # )

    # PARAM_GRID_DT: Dict[str, Any] = field(
    #     default_factory=lambda: {
    #         "max_depth": [None, 10, 20, 30],
    #         "min_samples_split": [2, 5, 10],
    #         "min_samples_leaf": [1, 2, 4],
    #         "criterion": ["gini", "entropy"],
    #     }
    # )

    PARAM_GRID_GNB: Dict[str, Any] = field(
        default_factory=lambda: {"var_smoothing": np.logspace(-9, -7, num=3)}
    )

    # PARAM_GRID_XGB: Dict[str, Any] = field(
    #     default_factory=lambda: {
    #         "n_estimators": [50, 100, 150],
    #         "learning_rate": [0.01, 0.1, 0.3],
    #         "max_depth": [3, 6, 10],
    #         "gamma": [0, 0.1, 0.3],
    #         "subsample": [0.6, 0.8, 1.0],
    #     }
    # )

    MODEL_PARAM_GRIDS: Dict[str, Dict[str, Any]] = field(init=False)

    SCORING_METRICS: Dict[str, str] = field(
        default_factory=lambda: {
            "accuracy": "accuracy",
            "recall": "recall_weighted",
            "precision": "precision_weighted",
            "f1": "f1_weighted",
        }
    )

    MODELS: Dict[str, Any] = field(init=False)

    DIRECTORIES: Dict[str, str] = field(
        default_factory=lambda: {
            "models_dir": "models",
            "params_dir": "model_parameters",
            "feature_importances_dir": "feature_importances",
            "metrics_dir": "metrics",
            "plots_dir": "plots",
        }
    )

    def __post_init__(self):
        self.MODEL_PARAM_GRIDS = {
            # "Random Forest": self.PARAM_GRID_RF,
            # "Decision Tree": self.PARAM_GRID_DT,
            "Naïve Bayes": self.PARAM_GRID_GNB,
            # "XGBoost": self.PARAM_GRID_XGB,
        }
        self.MODELS = {
            # "Random Forest": RandomForestClassifier(
            #     random_state=self.DEFAULT_RANDOM_STATE, verbose=2
            # ),
            # "Decision Tree": DecisionTreeClassifier(
            #     random_state=self.DEFAULT_RANDOM_STATE
            # ),
            "Naïve Bayes": GaussianNB(),
            # "XGBoost": XGBClassifier(
            #     random_state=self.DEFAULT_RANDOM_STATE,
            #     eval_metric="mlogloss",
            # ),
        }


config = Config()
