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
    balance_classes,
    clean_column_names,
    evaluate_model,
    export_feature_importances,
    generate_plot,
    load_data,
    preprocess_data,
    save_best_params,
    split_data,
)
from logger_singleton import get_logger
from train import train_models

# Configure logger
logger = get_logger(__name__)

random_state = config.DEFAULT_RANDOM_STATE


def parse_args():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--label_column", type=str, required=True, help="Name of the label column"
    )
    parser.add_argument(
        "--balance", action="store_true", help="Whether to balance classes using SMOTE"
    )
    parser.add_argument(
        "--scale", action="store_true", help="Whether to scale numerical features"
    )
    parser.add_argument(
        "--feature_selection",
        action="store_true",
        help="Whether to perform feature selection",
    )
    parser.add_argument(
        "--k_features", type=int, default=10, help="Number of top features to select"
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="metrics/model_evaluation_metrics.csv",
        help="Path to save evaluation metrics",
    )
    parser.add_argument(
        "--params_dir",
        type=str,
        default="model_parameters",
        help="Directory to save model parameters",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--feature_importances_dir",
        type=str,
        default="feature_importances",
        help="Directory to save feature importances",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots", help="Directory to save plots"
    )
    return parser.parse_args()


# Load the data from file
def main():

    try:
        args = parse_args()
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        return

    # Ensure the directories exist for saving
    logger.info("Creating directories for saving...")
    directories = config.DIRECTORIES
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories created or already exist.")

    # Load data
    try:
        logger.info("Loading data...")
        df = load_data(args.data_path)
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data_path}")
        return
    except pd.errors.EmptyDataError:
        logger.error(f"Data file is empty: {args.data_path}")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Preprocess the data
    try:
        logger.info("Preprocessing data...")
        df, le, scaler = preprocess_data(
            df,
            label_column=args.label_column,
            scale=args.scale,
            feature_selection=args.feature_selection,
            k_features=args.k_features,
        )
        logger.info("Data preprocessing completed.")
    except KeyError as e:
        logger.error(f"Missing column during preprocessing: {e}")
        return
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # Split the data
    try:
        logger.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = split_data(
            df,
            label_column=args.label_column,
            test_size=config.DEFAULT_TEST_SIZE,
            random_state=config.DEFAULT_RANDOM_STATE,
        )
        logger.info(
            f"Data split completed. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}"
        )
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return

    # Plot a count of classes
    # try:
    #     logger.info("Generating class distribution plot...")
    #     generate_plot(
    #         data=df,
    #         plot_type="count",
    #         x=args.label_column,
    #         title="Count of Classes",
    #         xlabel="Class",
    #         ylabel="Count",
    #     )
    #     logger.info("Class distribution plot generated.")
    # except Exception as e:
    #     logger.error(f"Error generating plot: {e}")

    # Balance classes if specified
    if args.balance:
        try:
            logger.info("Balancing classes using SMOTE...")
            X_train, y_train = balance_classes(X_train, y_train)
            logger.info(f"Classes balanced. New training samples: {X_train.shape[0]}")
        except Exception as e:
            logger.error(f"Error balancing classes: {e}")

    # Initialize a list to collect metrics
    all_metrics = []
    training_times = []

    # Iterate over models
    for model_name, model in config.MODELS.items():
        logger.info(f"Training model: {model_name}")

        param_grid = config.MODEL_PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            logger.warning(
                f"No parameter grid found for {model_name}. Skipping GridSearchCV."
            )
            continue

        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=config.SCORING_METRICS,
                refit="f1",
                cv=StratifiedKFold(n_splits=5),
                n_jobs=-1,
                verbose=2,
            )

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            duration = time.time() - start_time
            training_times.append({"model_name": model_name, "training_time": duration})
            logger.info(f"{model_name} training completed in {duration:.2f} seconds.")

            best_clf = grid_search.best_estimator_
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

            # Saving best parameters
            parameter_filename = os.path.join(
                config.DIRECTORIES["params_dir"],
                f"best_params_{model_name.replace(' ', '_')}.json",
            )
            best_params = grid_search.best_params_
            save_best_params(model_name, best_params, parameter_filename)
            logger.info(f"Best parameters saved to {parameter_filename}")

            # Evaluate
            metrics = evaluate_model(
                best_clf,
                X_train,
                y_train,
                X_test,
                y_test,
                label_encoder=le,
                model_name=model_name,
                plot_pr_curve=False,
            )
            logger.info(f"{model_name} Metrics: {metrics}")

            # Append metrics to the list
            all_metrics.append(metrics)

            # Save the model
            model_filename = os.path.join(
                config.DIRECTORIES["models_dir"],
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
                        config.DIRECTORIES["feature_importances_dir"],
                        f"feature_importances_{model_name.replace(' ', '_')}.csv",
                    ),
                )
                logger.info(f"Feature importances saved for {model_name}.")
            else:
                logger.info(f"{model_name} does not support feature importances.")
        except Exception as e:
            logger.error(f"Error training or evaluating model {model_name}: {e}")

    # After all models have been evaluated, save the metrics to a CSV file
    if all_metrics:
        try:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_filename = os.path.join(
                config.DIRECTORIES["metrics_dir"], "all_model_metrics.csv"
            )
            metrics_df.to_csv(metrics_filename, index=False)
            logger.info(f"All model metrics saved to {metrics_filename}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    else:
        logger.warning("No metrics to save.")

    # Save training time metrics
    try:
        if training_times:
            training_times_df = pd.DataFrame(training_times)
            training_times_filename = os.path.join(
                config.DIRECTORIES["metrics_dir"], "training_times.csv"
            )
            training_times_df.to_csv(training_times_filename, index=False)
            logger.info(f"Training times saved to {training_times_filename}")
    except Exception as e:
        logger.error(f"Error saving training times: {e}")


if __name__ == "__main__":
    main()
