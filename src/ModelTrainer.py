import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config
from helper import (
    GridSearchCV,
    StratifiedKFold,
    evaluate_model,
    export_feature_importances,
    generate_plot,
    save_best_params,
    split_data,
)
from LoggerSingleton import get_logger

# Configure logger
logger = get_logger(__name__)

random_state = config.DEFAULT_RANDOM_STATE


class ModelTrainer:
    def __init__(self, config, directories, scoring_metrics, random_state):
        self.config = config
        self.directories = directories
        self.scoring_metrics = scoring_metrics
        self.random_state = random_state
        self.metrics = []
        self.training_times = []

    def train_and_evaluate(
        self, model_name, model, X_train, y_train, X_test, y_test, label_encoder
    ):
        logger.info(f"Training model: {model_name}")

        param_grid = self.config.MODEL_PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            logger.warning(
                f"No parameter grid found for {model_name}. Skipping GridSearchCV."
            )
            return

        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=self.scoring_metrics,
                refit="f1",
                cv=StratifiedKFold(n_splits=3),
                n_jobs=10,
                verbose=1,
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train, early_stopping_rounds=10)
            duration = time.time() - start_time
            self.training_times.append(
                {"model_name": model_name, "training_time": duration}
            )
            logger.info(f"{model_name} training completed in {duration:.2f} seconds.")

            best_clf = grid_search.best_estimator_
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

            # Save best parameters
            parameter_filename = os.path.join(
                self.directories["params_dir"],
                f"best_params_{model_name.replace(' ', '_')}.json",
            )
            best_params = grid_search.best_params_
            save_best_params(model_name, best_params, parameter_filename)
            logger.info(f"Best parameters saved to {parameter_filename}")

            # Evaluate the model
            metrics = evaluate_model(
                best_clf,
                X_train,
                y_train,
                X_test,
                y_test,
                label_encoder=label_encoder,
                model_name=model_name,
                plot_pr_curve=False,
            )
            logger.info(f"{model_name} Metrics: {metrics}")

            self.metrics.append(metrics)

            # Save the model
            model_filename = os.path.join(
                self.directories["models_dir"],
                f"best_model_{model_name.replace(' ', '_')}.joblib",
            )
            joblib.dump(best_clf, model_filename)
            logger.info(f"Model saved to {model_filename}")

            # Export feature importances if available
            if hasattr(best_clf, "feature_importances_"):
                export_feature_importances(
                    best_clf,
                    feature_names=X_train.columns,
                    file_path=os.path.join(
                        self.directories["feature_importances_dir"],
                        f"feature_importances_{model_name.replace(' ', '_')}.csv",
                    ),
                )
                logger.info(f"Feature importances saved for {model_name}.")
            else:
                logger.info(f"{model_name} does not support feature importances.")

        except Exception as e:
            logger.error(f"Error training or evaluating model {model_name}: {e}")

    def save_metrics(self):
        if self.metrics:
            try:
                metrics_df = pd.DataFrame(self.metrics)
                metrics_filename = os.path.join(
                    self.directories["metrics_dir"], "all_model_metrics.csv"
                )
                metrics_df.to_csv(metrics_filename, index=False)
                logger.info(f"All model metrics saved to {metrics_filename}")
            except Exception as e:
                logger.error(f"Error saving metrics: {e}")
        else:
            logger.warning("No metrics to save.")

    def save_training_times(self):
        try:
            if self.training_times:
                training_times_df = pd.DataFrame(self.training_times)
                training_times_filename = os.path.join(
                    self.directories["metrics_dir"], "training_times.csv"
                )
                training_times_df.to_csv(training_times_filename, index=False)
                logger.info(f"Training times saved to {training_times_filename}")
        except Exception as e:
            logger.error(f"Error saving training times: {e}")
