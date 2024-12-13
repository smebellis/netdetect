import json
import os
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Optional, Union
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List, Dict
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from config import config
from LoggerSingleton import get_logger

# Configure logger
logger = get_logger(__name__)


# def file_load_large_csv(
#     file_path: Union[str, Path], chunksize: int = 100000
# ) -> pd.DataFrame:
#     chunks = []
#     for chunk in pd.read_csv(file_path, chunksize=chunksize):
#         cleaned_chunk = clean_data(chunk)
#         chunks.append(cleaned_chunk)
#     return pd.concat(chunks, ignore_index=True)


def parse_kwargs(kwargs_list: Optional[List[str]]) -> Dict:
    """
    Parses a list of key=value strings into a dictionary.

    Args:
        kwargs_list (List[str], optional): List of key=value strings.

    Returns:
        Dict: Parsed keyword arguments.
    """
    kwargs = {}
    if kwargs_list:
        for item in kwargs_list:
            key, value = item.split("=")
            # Attempt to convert to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            kwargs[key] = value
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the pickle data file"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=False,
        default="../data/raw/MachineLearningCSV/MachineLearningCVE/",
        help="Directory containing CSV files if pickle does not exist",
    )
    parser.add_argument(
        "--label_column", type=str, required=True, help="Name of the label column"
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
        "--balance_methods",
        type=str,
        nargs="+",
        choices=["smote", "weighted_loss", "vae"],
        default=[],
        help="Balancing methods to apply. Options: 'smote', 'weighted_loss', 'vae'.",
    )
    parser.add_argument(
        "--smote_kwargs",
        type=str,
        nargs="*",
        help="Additional keyword arguments for SMOTE in key=value format.",
    )
    parser.add_argument(
        "--vae_input_dim",
        type=int,
        help="Input dimension for VAE.",
    )
    parser.add_argument(
        "--vae_latent_dim",
        type=int,
        help="Latent dimension for VAE.",
    )
    parser.add_argument(
        "--vae_epochs",
        type=int,
        default=100,
        help="Number of epochs to train VAE.",
    )
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=64,
        help="Batch size for VAE training.",
    )
    parser.add_argument(
        "--vae_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for VAE optimizer.",
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


# Split the data
def split_data(
    df,
    label_column,
    test_size=config.DEFAULT_TEST_SIZE,
    random_state=config.DEFAULT_RANDOM_STATE,
):

    label_column = label_column.strip()
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def decode_labels(df, label_column, le):
    return le.inverse_transform(df[label_column])


def evaluate_model(
    clf,
    X_train,
    y_train,
    X_test,
    y_test,
    label_encoder,
    model_name=None,
    plot_pr_curve=True,
):
    """
    Evaluates the classifier on training and test data, logs metrics, plots confusion matrix
    and Precision-Recall curve, and returns evaluation metrics.

    Args:
        clf: Trained classifier.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        label_encoder: LabelEncoder fitted on the labels.
        model_name (str, optional): Name of the model for identification. Defaults to None.
        plot_pr_curve (bool, optional): Whether to plot the Precision-Recall curve. Defaults to False.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Evaluate on training data
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred, average="weighted")
    train_precision = precision_score(
        y_train, y_train_pred, average="weighted", zero_division=0
    )
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")

    # Evaluate on test data
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average="weighted")
    test_precision = precision_score(
        y_test, y_test_pred, average="weighted", zero_division=0
    )
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    # Initialize PR AUC
    pr_auc = None

    # Check if classifier has predict_proba or decision_function
    if hasattr(clf, "predict_proba"):
        y_scores = clf.predict_proba(X_test)
    elif hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)
    else:
        logger.warning(
            f"{model_name} does not have predict_proba or decision_function method. PR AUC cannot be computed."
        )
        y_scores = None

    # Compute PR AUC for each class and average
    if y_scores is not None:
        if len(label_encoder.classes_) == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_test, y_scores[:, 1])
            pr_auc = auc(recall, precision)
            if plot_pr_curve:
                plt.figure()
                plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve - {model_name}")
                plt.legend()
                plt.tight_layout()
                pr_curve_filename = (
                    os.path.join(
                        config.DIRECTORIES["plots_dir"],
                        f"pr_curve_{model_name.replace(' ', '_')}.png",
                    )
                    if model_name
                    else os.path.join(config.DIRECTORIES["plots_dir"], "pr_curve.png")
                )
                # Ensure the directory exists
                (
                    os.makedirs(os.path.dirname(pr_curve_filename), exist_ok=True)
                    if os.path.dirname(pr_curve_filename)
                    else None
                )
                plt.savefig(pr_curve_filename)
                plt.show(block=False)
                plt.pause(5)
                plt.close()
                logger.info(f"Precision-Recall curve saved to {pr_curve_filename}")
        else:
            # Multi-class classification: One-vs-Rest approach
            precision = dict()
            recall = dict()
            pr_auc = dict()
            for i, class_label in enumerate(label_encoder.classes_):
                precision[i], recall[i], _ = precision_recall_curve(
                    (y_test == i).astype(int), y_scores[:, i]
                )
                pr_auc[i] = auc(recall[i], precision[i])
                if plot_pr_curve:
                    plt.plot(
                        recall[i],
                        precision[i],
                        label=f"Class {class_label} PR AUC = {pr_auc[i]:.2f}",
                    )

            if plot_pr_curve:
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve - {model_name}")
                plt.legend()
                plt.tight_layout()
                pr_curve_filename = (
                    os.path.join(
                        config.DIRECTORIES["plots_dir"],
                        f"pr_curve_{model_name.replace(' ', '_')}.png",
                    )
                    if model_name
                    else os.path.join(config.DIRECTORIES["plots_dir"], "pr_curve.png")
                )
                # Ensure the directory exists
                (
                    os.makedirs(os.path.dirname(pr_curve_filename), exist_ok=True)
                    if os.path.dirname(pr_curve_filename)
                    else None
                )
                plt.savefig(pr_curve_filename)
                plt.show(block=False)
                plt.pause(5)
                plt.close()
                logger.info(f"Precision-Recall curves saved to {pr_curve_filename}")

    # Plot and save the confusion matrix
    plot_filename = (
        os.path.join(
            config.DIRECTORIES["plots_dir"],
            f"confusion_matrix_{model_name.replace(' ', '_')}.png",
        )
        if model_name
        else os.path.join(config.DIRECTORIES["plots_dir"], "confusion_matrix.png")
    )
    plot_confusion_matrix(
        y_test,
        y_test_pred,
        classes=label_encoder.classes_,
        title=f"Confusion Matrix - {model_name}" if model_name else "Confusion Matrix",
        save_path=plot_filename,
    )
    logger.info(f"Confusion matrix saved to {plot_filename}")

    # Log the results
    logger.info(f"Training Data Metrics for {model_name}:")
    logger.info(f"Accuracy: {train_accuracy}")
    logger.info(f"Recall: {train_recall}")
    logger.info(f"Precision: {train_precision}")
    logger.info(f"F1 Score: {train_f1}")

    logger.info(f"Test Data Metrics for {model_name}:")
    logger.info(f"Accuracy: {test_accuracy}")
    logger.info(f"Recall: {test_recall}")
    logger.info(f"Precision: {test_precision}")
    logger.info(f"F1 Score: {test_f1}")
    if pr_auc is not None:
        if isinstance(pr_auc, dict):
            for class_idx, auc_score in pr_auc.items():
                logger.info(
                    f"PR AUC for class {label_encoder.classes_[class_idx]}: {auc_score}"
                )
        else:
            logger.info(f"PR AUC: {pr_auc}")

    # Check for overfitting
    if train_accuracy > test_accuracy:
        logger.warning(f"The model {model_name} is likely overfitting.")
    else:
        logger.info(f"The model {model_name} is not overfitting.")

    # Prepare metrics dictionary
    metrics = {
        "model": model_name,
        "train_accuracy": train_accuracy,
        "train_recall": train_recall,
        "train_precision": train_precision,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_f1": test_f1,
    }

    # Add PR AUC to metrics
    if pr_auc is not None:
        if isinstance(pr_auc, dict):
            for class_idx, auc_score in pr_auc.items():
                metrics[f"pr_auc_class_{label_encoder.classes_[class_idx]}"] = auc_score
        else:
            metrics["pr_auc"] = pr_auc

    return metrics


def export_feature_importances(
    clf, feature_names, file_path="feature_importances.csv"
) -> pd.Series:
    """
    Extracts feature importances from a classifier, sorts them, and exports to a CSV file.

    Parameters:
    clf (estimator): Trained classifier with feature_importances_ attribute.
    feature_names (list): List of feature names.
    file_path (str): Path to save the CSV file. Defaults to 'feature_importances.csv'.

    Returns:
    pd.Series: Sorted feature importances.
    """

    # Check if classifier has feature_importances_
    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(
            "The classifier does not have feature_importances_ attribute."
        )

    # Validate feature_names length
    if len(feature_names) != len(clf.feature_importances_):
        raise ValueError(
            "Length of feature_names does not match number of features in the classifier."
        )

    # Extract and sort feature importances
    feature_importances = clf.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)

    # Export to CSV with exception handling
    try:
        importance_df.to_csv(file_path, index=False)
        logger.info(f"Feature importances saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save feature importances to CSV: {e}")
        raise

    # Return sorted series
    return importance_df.set_index("Feature")["Importance"]


def plot_feature_importances(
    sorted_importances,
    title="Feature Importances",
    figsize=(10, 6),
    color="skyblue",
    top_n=None,
    horizontal=False,
    save_path=None,
):

    plt.figure(figsize=figsize)
    if horizontal:
        sorted_importances.plot(kind="barh", color=color)
        plt.xlabel("Importance")
        plt.ylabel("Features")
    else:
        sorted_importances.plot(kind="bar", color=color)
        plt.ylabel("Importance")
        plt.xlabel("Features")

    plt.title(title)

    # Add value labels
    for index, value in enumerate(sorted_importances):
        if horizontal:
            plt.text(value, index, f"{value:.4f}", va="center")
        else:
            plt.text(index, value, f"{value:.4f}", ha="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importances plot saved to {save_path}.")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    logger.info(f"Plotted feature importances with title '{title}'.")


def print_best_params(grid_search):
    print("\nBest Parameters found by GridSearchCV:")
    print(grid_search.best_params_)


def remove_features(df, feature_importances, threshold=None):
    """
    Remove unimportant features from the DataFrame based on feature importances.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the features.
    feature_importances (np.array): The array of feature importances.
    threshold (float): The threshold below which features are considered unimportant. If None, use the mean importance.

    Returns:
    pd.DataFrame: The DataFrame with unimportant features removed.
    """

    if threshold is None:
        threshold = feature_importances.mean()

    importance_features = df.columns[feature_importances >= threshold]

    return df[importance_features]


def plot_confusion_matrix(
    y_true, y_pred, classes, title="Confusion Matrix", save_path=None
):
    """
    Plots and optionally saves the confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: List of class names.
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        save_path (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)

    plt.tight_layout()
    if save_path:
        # Ensure the directory exists
        (
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.dirname(save_path)
            else None
        )
        plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def save_best_params(model_name, best_params, parameter_filename):
    """
    Saves the best hyperparameters of a model to a JSON file.

    Args:
        model_name (str): The name of the model.
        best_params (dict): The best hyperparameters obtained from GridSearchCV.
    """

    try:
        with open(parameter_filename, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best parameters for {model_name} saved at {parameter_filename}")
    except Exception as e:
        logger.error(f"Failed to save best parameters for {model_name}: {e}")


# Function to generate plots
def generate_plot(
    data, plot_type="bar", x=None, y=None, title="", xlabel="", ylabel="", hue=None
):
    """
    Generate different types of plots using seaborn.

    Parameters:
    - data: DataFrame or array-like, the dataset to plot.
    - plot_type: str, the type of plot ('bar', 'hist', 'scatter', 'line', etc.).
    - x: str, the column name for the x-axis (if applicable).
    - y: str, the column name for the y-axis (if applicable).
    - title: str, the title of the plot.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    - hue: str, column to group data by color (used in scatter plots and barplots).

    Supported plot types: 'bar', 'hist', 'scatter', 'line', 'count'.
    """
    plt.figure(figsize=(8, 6))

    # Bar Plot
    if plot_type == "bar":
        sns.barplot(x=x, y=y, hue=hue, data=data)

    # Histogram
    elif plot_type == "hist":
        sns.histplot(data[x], kde=True)

    # Scatter Plot
    elif plot_type == "scatter":
        sns.scatterplot(x=x, y=y, hue=hue, data=data)

    # Line Plot
    elif plot_type == "line":
        sns.lineplot(x=x, y=y, hue=hue, data=data)

    # Count Plot (for categorical columns)
    elif plot_type == "count":
        sns.countplot(x=x, data=data, hue=hue)

    # Set plot details
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()
    # plt.pause(5)
    # plt.close()


def save_plot():
    pass
