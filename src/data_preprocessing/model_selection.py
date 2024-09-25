import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import time
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    make_scorer,
)
import time
import joblib
import sys
import json
from preprocess import (
    split_data,
    remove_features,
    balance_classes,
    load_resampled_data,
    load_data,
)


models = {
    "Random Forest": RandomForestClassifier(verbose=2, n_jobs=8),
    "Random Tree": DecisionTreeClassifier(),
    "Naïve Bayes": GaussianNB(),
}


def main():
    results = []
    df = load_data(
        "/home/smebellis/ece579/final_project/network_anomaly_detection/data/processed/cleaned_df.pkl"
    )
    X_train, X_test, y_train, y_test = split_data(df, " Label")

    # X_res, y_res = balance_classes(X_train, y_train)
    # joblib.dump(X_res, "X_resampled.pkl")
    # joblib.dump(y_res, "y_resampled.pkl")
    # print("\nResampled data saved")

    X_res = load_resampled_data(
        "/home/smebellis/ece579/final_project/network_anomaly_detection/src/data_preprocessing/X_resampled.pkl"
    )
    y_res = load_resampled_data(
        "/home/smebellis/ece579/final_project/network_anomaly_detection/src/data_preprocessing/y_resampled.pkl"
    )

    # Check the new balanced classes
    class_distribution = Counter(y_res)
    print("\nNew class distribution after SMOTE:")
    for label, count in class_distribution.items():
        print(f"Class {label}: {count} samples")
    for model_name, model in models.items():
        start_time = time.time()
        print(f"Training {model_name}")
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )
        end_time = time.time()
        duration = end_time - start_time
        print(f"{model_name} trained in {duration} seconds")

    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
