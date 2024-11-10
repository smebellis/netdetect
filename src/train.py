import os
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import joblib
from config import config
from helper import save_best_params, export_feature_importances, evaluate_model

from logger_singleton import get_logger

# Configure logger
logger = get_logger(__name__)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
) -> None:
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
                label_encoder=label_encoder,
                model_name=model_name,
                plot_pr_curve=True,
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

    return all_metrics, training_times
