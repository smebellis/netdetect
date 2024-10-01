import argparse
import json
import logging
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
    load_data,
    preprocess_data,
    split_data,
    generate_plot,
    balance_classes,
    GridSearchCV,
    StratifiedKFold,
    save_best_params,
    evaluate_model,
    export_feature_importances,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("file_load.log"), logging.StreamHandler(sys.stdout)],
)

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
        logging.error(f"Error parsing arguments: {e}")
        return

    # Ensure the directories exist for saving
    logging.info("Creating directories for saving...")
    directories = config.DIRECTORIES
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    logging.info("Directories created or already exist.")

    # Load data
    logging.info("Loading data...")
    df = load_data(args.data_path)
    logging.info("Data loaded successfully.")

    # Preprocess the data
    logging.info("Preprocessing data...")
    df, le, scaler = preprocess_data(
        df,
        label_column=args.label_column,
        scale=args.scale,
        feature_selection=args.feature_selection,
        k_features=args.k_features,
    )
    logging.info("Data preprocessing completed.")

    # Split the data
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(
        df,
        label_column=args.label_column,
        test_size=config.DEFAULT_TEST_SIZE,
        random_state=config.DEFAULT_RANDOM_STATE,
    )
    logging.info(
        f"Data split completed. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}"
    )

    # Plot a count of classes
    logging.info("Generating class distribution plot...")
    generate_plot(
        data=df,
        plot_type="count",
        x=args.label_column,
        title="Count of Classes",
        xlabel="Class",
        ylabel="Count",
        # save_path=os.path.join(
        #     config.DIRECTORIES["metrics_dir"], "class_distribution.png"
        # ),
    )
    logging.info("Class distribution plot generated.")

    # Balance classes if specified
    if args.balance:
        logging.info("Balancing classes using SMOTE...")
        X_train, y_train = balance_classes(X_train, y_train)
        logging.info(f"Classes balanced. New training samples: {X_train.shape[0]}")
    else:
        logging.info("Class balancing not performed.")

    # Initialize a list to collect metrics
    all_metrics = []

    # Iterate over models
    for model_name, model in config.MODELS.items():
        logging.info(f"Training model: {model_name}")

        param_grid = config.MODEL_PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            logging.warning(
                f"No parameter grid found for {model_name}. Skipping GridSearchCV."
            )
            continue

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
        logging.info(f"{model_name} training completed in {duration:.2f} seconds.")

        best_clf = grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

        # Saving best parameters
        parameter_filename = os.path.join(
            config.DIRECTORIES["params_dir"],
            f"best_params_{model_name.replace(' ', '_')}.json",
        )
        best_params = grid_search.best_params_
        save_best_params(model_name, best_params, parameter_filename)
        logging.info(f"Best parameters saved to {parameter_filename}")

        # Evaluate
        metrics = evaluate_model(
            best_clf,
            X_train,
            y_train,
            X_test,
            y_test,
            label_encoder=le,
            model_name=model_name,
            plot_pr_curve=True,  # Enable plotting PR curve
            # save_path=os.path.join(
            #     config.DIRECTORIES["metrics_dir"],
            #     f"pr_curve_{model_name.replace(' ', '_')}.png",
            # ),
        )
        logging.info(f"{model_name} Metrics: {metrics}")

        # Append metrics to the list
        all_metrics.append(metrics)

        # Save the model
        model_filename = os.path.join(
            config.DIRECTORIES["models_dir"],
            f"best_model_{model_name.replace(' ', '_')}.joblib",
        )
        joblib.dump(best_clf, model_filename)
        logging.info(f"Model saved to {model_filename}")

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
            logging.info(f"Feature importances saved for {model_name}.")
        else:
            logging.info(f"{model_name} does not support feature importances.")

    # After all models have been evaluated, save the metrics to a CSV file
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_filename = os.path.join(
            config.DIRECTORIES["metrics_dir"], "all_model_metrics.csv"
        )
        metrics_df.to_csv(metrics_filename, index=False)
        logging.info(f"All model metrics saved to {metrics_filename}")
    else:
        logging.warning("No metrics to save.")


if __name__ == "__main__":
    main()
