import json
import os
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from logger_singleton import get_logger

# Configure logger
logger = get_logger(__name__)


def file_load(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a DataFrame from a pickle file if it exists. If not, it reads CSV files from a specified directory,
    cleans and concatenates them into a single DataFrame, saves it as a pickle file, and returns the DataFrame.

    Args:
        file_path (str or Path): The path to the pickle file.

    Returns:
        pd.DataFrame: The loaded or newly created DataFrame.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified directory.
        Exception: For any other exceptions that may occur during file operations.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.info(f"File does not exist: {file_path}")

        csv_dir = Path("../data/raw/MachineLearningCSV/MachineLearningCVE/")
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found in the specified directory.")
            raise FileNotFoundError("No CSV files found in the specified directory.")

        dataframes = []
        for file in tqdm(csv_files, desc="Processing CSV Files"):
            try:
                logger.info(f"Reading CSV file: {file}")
                df = pd.read_csv(file)
                cleaned = clean_data(df)
                dataframes.append(cleaned)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue  # Skip files that cause errors

        if not dataframes:
            logger.error("No dataframes were loaded successfully.")
            raise ValueError("No dataframes were loaded successfully.")

        try:
            concatenated_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Concatenated {len(dataframes)} dataframes.")
        except Exception as e:
            logger.error(f"Error concatenating dataframes: {e}")
            raise

        try:
            concatenated_df.to_pickle(file_path)
            logger.info(f"Saved concatenated DataFrame to {file_path}")
        except Exception as e:
            logger.error(f"Error saving pickle file: {e}")
            raise

        return concatenated_df
    else:
        logger.info(f"Loading DataFrame from pickle file: {file_path}")
        try:
            return pd.read_pickle(file_path)
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            raise


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a DataFrame from a pickle file if it exists. If not, it reads CSV files from a specified directory,
    cleans and concatenates them into a single DataFrame, saves it as a pickle file, and returns the DataFrame.

    This function wraps around `file_load` to provide additional logger and error handling.

    Args:
        file_path (str or Path): The path to the pickle file.

    Returns:
        pd.DataFrame: The loaded or newly created DataFrame.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified directory and the pickle file does not exist.
        pd.errors.EmptyDataError: If CSV files are empty.
        Exception: For any other exceptions that may occur during file operations.
    """
    try:
        df = file_load(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise
    except pd.errors.EmptyDataError as ede_error:
        logger.error(f"No data: {ede_error}")
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading data from {file_path}: {e}"
        )
        raise


def clean_data(
    df: pd.DataFrame,
    replace_inf: bool = True,
    drop_na: bool = True,
    drop_duplicates: bool = True,
    subset_duplicates: Optional[list] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling infinite values, missing data, and duplicates.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        replace_inf (bool, optional): Whether to replace infinite values with NaN. Defaults to True.
        drop_na (bool, optional): Whether to drop rows containing NaN values. Defaults to True.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
        subset_duplicates (list, optional): Columns to consider when identifying duplicates. Defaults to None.
        inplace (bool, optional): Whether to perform operations in place. Defaults to False.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")

    df_cleaned = df.copy() if not inplace else df

    # Replace infinite values with NaN
    if replace_inf:
        inf_count = np.isinf(df_cleaned).sum().sum()
        df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Replaced {inf_count} infinite values with NaN.")

    # Drop rows with NaN values
    if drop_na:
        initial_shape = df_cleaned.shape
        df_cleaned.dropna(inplace=True)
        final_shape = df_cleaned.shape
        dropped_na = initial_shape[0] - final_shape[0]
        logger.info(f"Dropped {dropped_na} rows containing NaN values.")

    # Drop duplicate rows
    if drop_duplicates:
        if subset_duplicates:
            duplicated = df_cleaned.duplicated(subset=subset_duplicates).sum()
            df_cleaned.drop_duplicates(subset=subset_duplicates, inplace=True)
            logger.info(
                f"Dropped {duplicated} duplicate rows based on subset {subset_duplicates}."
            )
        else:
            duplicated = df_cleaned.duplicated().sum()
            df_cleaned.drop_duplicates(inplace=True)
            logger.info(f"Dropped {duplicated} duplicate rows.")

    return df_cleaned


def clean_column_names(df):
    """Clean DataFrame column names by stripping whitespace"""
    df.columns = df.columns.str.strip()
    return df


def file_load_large_csv(
    file_path: Union[str, Path], chunksize: int = 100000
) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        cleaned_chunk = clean_data(chunk)
        chunks.append(cleaned_chunk)
    return pd.concat(chunks, ignore_index=True)


# Encode the labels
def encode_labels(df, label_column):
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])

    return df, le


# Scale the numerical data
def scale_data(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, scaler


def balance_classes(X_train, y_train):
    smote = SMOTE(random_state=42)
    try:
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"Successfully balanced classes. New shape: {X_res.shape}")
        return X_res, y_res
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        raise ValueError(f"SMOTE failed: {e}")


def select_features(df, label_column, k=10):
    """
    Select top k features while preserving the label column

    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of the label column
        k (int): Number of features to select

    Returns:
        pd.DataFrame: DataFrame with selected features and label column
    """
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    logger.info(f"Selected top {k} features: {list(selected_features)}")

    # Create a dataframe with the selected features
    X_selected_df = pd.DataFrame(X_new, columns=selected_features, index=df.index)

    # Add the label column back to the dataframe
    result_df = pd.concat([X_selected_df, df[label_column]], axis=1)

    return result_df


def preprocess_data(
    df, label_column, scale=False, feature_selection=False, k_features=10
):

    # Clean column names
    df = clean_column_names(df)
    label_column = label_column.strip()
    # Encode labels
    df, le = encode_labels(df, label_column)

    # Scale data if required
    if scale:
        numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
        df, scaler = scale_data(df, numerical_columns)
    else:
        scaler = None

    # Feature selection if required
    if feature_selection:
        df = select_features(df, label_column, k=k_features)

    return df, le, scaler


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


def calculate_class_weights(y_train):
    class_counts = Counter(y_train)

    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    class_weights = {
        cls: total_samples / (num_classes * count)
        for cls, count in class_counts.items()
    }

    return class_weights


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
    plt.show(block=False)
    plt.pause(5)
    plt.close()
