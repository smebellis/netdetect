import argparse
import json
import sys
import time
import numpy as np
from collections import Counter
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Union, Optional
from glob import glob
from tqdm import tqdm
import seaborn as sns

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("file_load.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Parameter grid specific to RandomForestClassifier
PARAM_GRID_RF = {
    "n_estimators": [50, 75, 100],
    "max_samples": [0.25, 0.5, 0.75],
    "max_depth": [2],
    "criterion": ["gini", "entropy"],
}

# Parameter grids for different classifiers
PARAM_GRID_DT = {
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

PARAM_GRID_GNB = {
    # GaussianNB has limited hyperparameters, but you can adjust var_smoothing
    "var_smoothing": np.logspace(-9, -7, num=3)
}

# Update MODELS with corresponding parameter grids
MODEL_PARAM_GRIDS = {
    "Random Forest": PARAM_GRID_RF,
    "Decision Tree": PARAM_GRID_DT,
    "Naïve Bayes": PARAM_GRID_GNB,
}

SCORING_METRICS = {
    "accuracy": 'accuracy',
    "recall": 'recall_weighted',
    "precision": 'precision_weighted',
    "f1": 'f1_weighted',
}

MODELS = {
    "Random Forest": RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE, verbose=2),
    "Decision Tree": DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
    "Naïve Bayes": GaussianNB(),
}

def clean_data(
    df: pd.DataFrame,
    replace_inf: bool = True,
    drop_na: bool = True,
    drop_duplicates: bool = True,
    subset_duplicates: Optional[list] = None,
    inplace: bool = False
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
        logging.error("Input is not a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")

    df_cleaned = df.copy() if not inplace else df

    # Replace infinite values with NaN
    if replace_inf:
        inf_count = np.isinf(df_cleaned).sum().sum()
        df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        logging.info(f"Replaced {inf_count} infinite values with NaN.")

    # Drop rows with NaN values
    if drop_na:
        initial_shape = df_cleaned.shape
        df_cleaned.dropna(inplace=True)
        final_shape = df_cleaned.shape
        dropped_na = initial_shape[0] - final_shape[0]
        logging.info(f"Dropped {dropped_na} rows containing NaN values.")

    # Drop duplicate rows
    if drop_duplicates:
        if subset_duplicates:
            duplicated = df_cleaned.duplicated(subset=subset_duplicates).sum()
            df_cleaned.drop_duplicates(subset=subset_duplicates, inplace=True)
            logging.info(f"Dropped {duplicated} duplicate rows based on subset {subset_duplicates}.")
        else:
            duplicated = df_cleaned.duplicated().sum()
            df_cleaned.drop_duplicates(inplace=True)
            logging.info(f"Dropped {duplicated} duplicate rows.")

    return df_cleaned

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
        logging.info(f"File does not exist: {file_path}")

        csv_dir = Path("../data/raw/MachineLearningCSV/MachineLearningCVE/")
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            logging.error("No CSV files found in the specified directory.")
            raise FileNotFoundError("No CSV files found in the specified directory.")

        dataframes = []
        for file in tqdm(csv_files, desc="Processing CSV Files"):
            try:
                logging.info(f"Reading CSV file: {file}")
                df = pd.read_csv(file)
                cleaned = clean_data(df)
                dataframes.append(cleaned)
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                continue  # Skip files that cause errors

        if not dataframes:
            logging.error("No dataframes were loaded successfully.")
            raise ValueError("No dataframes were loaded successfully.")

        try:
            concatenated_df = pd.concat(dataframes, ignore_index=True)
            logging.info(f"Concatenated {len(dataframes)} dataframes.")
        except Exception as e:
            logging.error(f"Error concatenating dataframes: {e}")
            raise

        try:
            concatenated_df.to_pickle(file_path)
            logging.info(f"Saved concatenated DataFrame to {file_path}")
        except Exception as e:
            logging.error(f"Error saving pickle file: {e}")
            raise

        return concatenated_df
    else:
        logging.info(f"Loading DataFrame from pickle file: {file_path}")
        try:
            return pd.read_pickle(file_path)
        except Exception as e:
            logging.error(f"Error loading pickle file: {e}")
            raise
    
def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a DataFrame from a pickle file if it exists. If not, it reads CSV files from a specified directory,
    cleans and concatenates them into a single DataFrame, saves it as a pickle file, and returns the DataFrame.
    
    This function wraps around `file_load` to provide additional logging and error handling.

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
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        raise
    except pd.errors.EmptyDataError as ede_error:
        logging.error(f"No data: {ede_error}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data from {file_path}: {e}")
        raise

def file_load_large_csv(file_path: Union[str, Path], chunksize: int = 100000) -> pd.DataFrame:
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

    X_res, y_res = smote.fit_resample(X_train, y_train)

    return X_res, y_res

def preprocess_data(df, label_column, scale=False, feature_selection=False):
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
        # Placeholder for feature selection logic
        pass
    
    return df, le, scaler

# Split the data
def split_data(df, label_column, test_size=0.3, random_state=DEFAULT_RANDOM_STATE):
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

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    # Evaluate on training data
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred, average="weighted")
    train_precision = precision_score(y_train, y_train_pred, average="weighted", zero_division=1)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")

    # Evaluate on test data
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average="weighted")
    test_precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    # Log the results
    logging.info("Training Data Metrics:")
    logging.info(f"Accuracy: {train_accuracy}")
    logging.info(f"Recall: {train_recall}")
    logging.info(f"Precision: {train_precision}")
    logging.info(f"F1 Score: {train_f1}")

    logging.info("Test Data Metrics:")
    logging.info(f"Accuracy: {test_accuracy}")
    logging.info(f"Recall: {test_recall}")
    logging.info(f"Precision: {test_precision}")
    logging.info(f"F1 Score: {test_f1}")
    
    # Check for overfitting
    if train_accuracy > test_accuracy:
        print("\nThe model is likely overfitting.")
    else:
        print("\nThe model is not overfitting.")

    # Return metrics
    return {
        "train_accuracy": train_accuracy,
        "train_recall": train_recall,
        "train_precision": train_precision,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_f1": test_f1,
    }

def export_feature_importances(clf, feature_names, file_path="feature_importances.csv")-> pd.Series:
    """
    Extracts feature importances from a classifier, sorts them, and exports to a CSV file.

    Parameters:
    clf (estimator): Trained classifier with feature_importances_ attribute.
    feature_names (list): List of feature names.
    file_path (str): Path to save the CSV file. Defaults to 'feature_importances.csv'.

    Returns:
    pd.Series: Sorted feature importances.
    """
    logger = logging.getLogger(__name__)

    # Check if classifier has feature_importances_
    if not hasattr(clf, "feature_importances_"):
        raise AttributeError("The classifier does not have feature_importances_ attribute.")

    # Validate feature_names length
    if len(feature_names) != len(clf.feature_importances_):
        raise ValueError("Length of feature_names does not match number of features in the classifier.")

    # Extract and sort feature importances
    feature_importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    })
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
    title='Feature Importances', 
    figsize=(10, 6), 
    color='skyblue', 
    top_n=None,
    horizontal=False,
    save_path=None
):
    logger = logging.getLogger(__name__)
    
    plt.figure(figsize=figsize)
    if horizontal:
        sorted_importances.plot(kind='barh', color=color)
        plt.xlabel('Importance')
        plt.ylabel('Features')
    else:
        sorted_importances.plot(kind='bar', color=color)
        plt.ylabel('Importance')
        plt.xlabel('Features')
    
    plt.title(title)
    
    # Add value labels
    for index, value in enumerate(sorted_importances):
        if horizontal:
            plt.text(value, index, f'{value:.4f}', va='center')
        else:
            plt.text(index, value, f'{value:.4f}', ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importances plot saved to {save_path}.")
    plt.show()
    
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

# Function to generate plots
def generate_plot(data, plot_type='bar', x=None, y=None, title='', xlabel='', ylabel='', hue=None):
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
    if plot_type == 'bar':
        sns.barplot(x=x, y=y, hue=hue, data=data)
        
    # Histogram
    elif plot_type == 'hist':
        sns.histplot(data[x], kde=True)
        
    # Scatter Plot
    elif plot_type == 'scatter':
        sns.scatterplot(x=x, y=y, hue=hue, data=data)
        
    # Line Plot
    elif plot_type == 'line':
        sns.lineplot(x=x, y=y, hue=hue, data=data)
        
    # Count Plot (for categorical columns)
    elif plot_type == 'count':
        sns.countplot(x=x, data=data, hue=hue)
    
    # Set plot details
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Show the plot
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--label_column", type=str, required=True, help="Name of the label column"
    )
    # Add more arguments as needed
    return parser.parse_args()

# Load the data from file
def main():
    
    args = parse_args()
     # Load and preprocess data
    df = load_data(args.data_path)
    
    df, le, scaler = preprocess_data(df, label_column=args.label_column, scale=False, feature_selection=False)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df, label_column=args.label_column)
    
    # Plot a count of classes
    generate_plot(data=df, plot_type='count', x=args.label_column, title='Count of Classes', xlabel='Class', ylabel='Count')
       
    # Balance classes
    # X_train, y_train = balance_classes(X_train, y_train)
    # logging.info("Classes balanced using SMOTE.")
    
    
    # Iterate over models
    for model_name, model in MODELS.items():
        logging.info(f"Training model: {model_name}")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=MODEL_PARAM_GRIDS.get(model_name, {}),
            scoring=SCORING_METRICS,
            refit="f1",
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        duration = time.time() - start_time
        logging.info(f"{model_name} training completed in {duration:.2f} seconds.")

        best_clf = grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

        # Evaluate
        metrics = evaluate_model(best_clf, X_train, y_train, X_test, y_test)
        logging.info(f"{model_name} Metrics: {metrics}")
        
        # Save the model
        model_filename = f"best_model_{model_name.replace(' ', '_')}.pkl"
        joblib.dump(best_clf, model_filename)
        logging.info(f"Model saved to {model_filename}")

        if hasattr(best_clf, 'feature_importances_'):
            export_feature_importances(best_clf, feature_names=X_train.columns, file_path=f"feature_importances_{model_name.replace(' ', '_')}.csv")
        else:
            logging.info(f"{model_name} does not support feature importances.")

if __name__ == "__main__":
    main()

