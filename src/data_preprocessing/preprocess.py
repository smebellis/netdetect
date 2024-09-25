import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import argparse


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    make_scorer,
    confusion_matrix,
)
import time
import joblib
import sys
import json

sys.path.append("/home/smebellis/ece579/final_project/network_anomaly_detection/")

from notebooks.exploratory import file_load

# Define the parameter grid
param_grid = {
    "n_estimators": [50, 75, 100],
    "max_samples": [0.25, 0.5, 0.75],
    "max_depth": [2],
    "criterion": ["gini", "entropy"],
}

# Define the scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "recall": make_scorer(recall_score, average="weighted"),
    "precision": make_scorer(precision_score, average="weighted"),
    "f1": make_scorer(f1_score, average="weighted"),
    # "confusion": make_scorer(confusion_matrix),
}

models = {
    "Random Forest": RandomForestClassifier(),
    "Random Tree": DecisionTreeClassifier(),
    "Naïve Bayes": GaussianNB(),
}

thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def load_data(file_path):
    return file_load(file_path)


# Encode the labels
def encode_labels(df, label_column):
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])

    return df, le


def balance_classes(X_train, y_train):
    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(X_train, y_train)

    return X_res, y_res


def load_balanced_classes(X_res_file_path, y_res_file_path):
    X_res = joblib.load(X_res_file_path)
    y_res = joblib.load(y_res_file_path)

    return X_res, y_res


# Scale the numerical data
def scale_data(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, scaler


# Split the data
def split_data(df, label_column, test_size=0.3, random_state=42):
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
    train_precision = precision_score(
        y_train, y_train_pred, average="weighted", zero_division=1
    )
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    # confusion_train = confusion_matrix(y_train, y_train_pred)

    # Evaluate on test data
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average="weighted")
    test_precision = precision_score(
        y_test, y_test_pred, average="weighted", zero_division=1
    )
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    # confusion_test = confusion_matrix(y_test, y_test_pred)

    # Print the results
    print("Training Data Metrics:")
    print(f"Accuracy: {train_accuracy}")
    print(f"Recall: {train_recall}")
    print(f"Precision: {train_precision}")
    print(f"F1 Score: {train_f1}")

    print("\nTest Data Metrics:")
    print(f"Accuracy: {test_accuracy}")
    print(f"Recall: {test_recall}")
    print(f"Precision: {test_precision}")
    print(f"F1 Score: {test_f1}")

    # Check for overfitting
    if train_accuracy > test_accuracy:
        print("\nThe model is likely overfitting.")
    else:
        print("\nThe model is not overfitting.")


def export_feature_importances(clf, feature_names, file_path="feature_importances.csv"):
    feature_importances = clf.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    importance_df.to_csv(file_path, index=False)
    print(f"\nFeature importances saved to {file_path}")

    return importance_df


def export_important_features(
    importance_df, threshold, file_path="important_features.json"
):
    important_features = importance_df[importance_df["Importance"] >= threshold][
        "Feature"
    ].to_list()
    with open(file_path, "w") as f:
        json.dump(important_features, f, indent=4)
    print(f"\nImportant features saved to {file_path}")


def load_important_features(file_path):
    with open(file_path, "r") as f:
        important_features = json.load(f)
    return important_features


def load_feature_importances(file_path):
    """
    Load feature importances from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing feature importances.

    Returns:
        pd.Series: A Series containing the feature importances.
    """
    importance_df = pd.read_csv(file_path)
    return pd.Series(importance_df.Importance.values, index=importance_df.Feature)


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
    # Ensure feature importance has the same length as df.columns
    if len(feature_importances) != len(df.columns):
        raise ValueError("Length of features does not match number of columns in df")
    if threshold is None:
        threshold = feature_importances.mean()

    importance_features = df.columns[feature_importances >= threshold]

    return df[importance_features]


def load_resampled_data(file_path):
    return joblib.load(file_path)


def print_best_params(grid_search):
    print("\nBest Parameters found by GridSearchCV:")
    print(grid_search.best_params_)


def parse_args():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file"
    )
    # Add more arguments as needed
    return parser.parse_args()


# Load the data from file
def main():
    args = parse_args()

    combined_df = file_load(file_path=args.data_path)
    # Encode the labels
    combined_df, le = encode_labels(combined_df, label_column=" Label")

    # Scale the numerical data
    numerical_columns = combined_df.select_dtypes(include=["number"]).columns.tolist()

    # combined_df, scaler = scale_data(combined_df, numerical_columns=numerical_columns)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(combined_df, label_column=" Label")

    # Print shapes of the splits
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # smote = SMOTE(random_state=42)
    class_weights = calculate_class_weights(y_train)

    # Balance the classes
    # print(f"Balancing the classes....\n{y_train.value_counts()}")
    # X_res, y_res = balance_classes(X_train, y_train)
    # print(f"Classes Balanced...\n{y_res.value_counts()}")

    clf = RandomForestClassifier(random_state=42, verbose=2)
    cv = StratifiedKFold(n_splits=5)
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=10,
    )

    # Record the start time
    start_time = time.time()

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Record the end time
    end_time = time.time()

    duration = end_time - start_time
    print(f"Model training completed in {duration:.2f} seconds.")

    # Get the best estimator
    best_clf = grid_search.best_estimator_
    print(best_clf)

    evaluate_model(best_clf, X_train, y_train, X_test, y_test)
    best_params = grid_search.best_params_
    print_best_params(grid_search)

    joblib.dump(best_clf, "best_model.pkl")
    print("\nModel saved to best_model.pkl")

    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    importance_df = export_feature_importances(best_clf, feature_names=X_train.columns)

    export_important_features(importance_df, 0.010)


if __name__ == "__main__":
    main()
