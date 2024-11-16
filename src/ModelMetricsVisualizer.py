import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ModelMetricsVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metrics_df = pd.read_csv(file_path)

    def plot_model_metrics(self):
        # Set plot style
        sns.set(style="whitegrid")

        # Plot 1: Training and Test Accuracy for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_accuracy",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train Accuracy",
        )
        sns.barplot(
            x="model",
            y="test_accuracy",
            data=self.metrics_df,
            color="r",
            alpha=0.6,
            label="Test Accuracy",
        )
        plt.title("Training and Test Accuracy for All Models")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 2: Training and Test Recall for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_recall",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train Recall",
        )
        sns.barplot(
            x="model",
            y="test_recall",
            data=self.metrics_df,
            color="r",
            alpha=0.6,
            label="Test Recall",
        )
        plt.title("Training and Test Recall for All Models")
        plt.xlabel("Model")
        plt.ylabel("Recall")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 3: Training and Test Precision for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_precision",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train Precision",
        )
        sns.barplot(
            x="model",
            y="test_precision",
            data=self.metrics_df,
            color="r",
            alpha=0.6,
            label="Test Precision",
        )
        plt.title("Training and Test Precision for All Models")
        plt.xlabel("Model")
        plt.ylabel("Precision")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 4: Training and Test F1 Score for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_f1",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train F1 Score",
        )
        sns.barplot(
            x="model",
            y="test_f1",
            data=self.metrics_df,
            color="r",
            alpha=0.6,
            label="Test F1 Score",
        )
        plt.title("Training and Test F1 Score for All Models")
        plt.xlabel("Model")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 5: PR AUC Scores for Different Classes
        plt.figure(figsize=(14, 8))
        pr_auc_columns = [
            col for col in self.metrics_df.columns if "pr_auc_class_" in col
        ]
        for class_name in pr_auc_columns:
            sns.lineplot(
                x="model", y=class_name, data=self.metrics_df, label=class_name
            )
        plt.title("Precision-Recall AUC Scores for Different Classes")
        plt.xlabel("Model")
        plt.ylabel("PR AUC Score")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()

    def display_best_params(self):
        # Assuming the best parameters are provided in columns with prefix 'best_params_'
        best_params_columns = [
            col for col in self.metrics_df.columns if "best_params_" in col
        ]
        for index, row in self.metrics_df.iterrows():
            model_name = row["model"]
            best_params = {
                col.replace("best_params_", ""): row[col]
                for col in best_params_columns
                if pd.notna(row[col])
            }
            print(f"Best parameters for {model_name}: {best_params}")

    def display_feature_importances(self):
        # Assuming feature importances are provided in columns with prefix 'feature_importance_'
        feature_importance_columns = [
            col for col in self.metrics_df.columns if "feature_importance_" in col
        ]
        for index, row in self.metrics_df.iterrows():
            model_name = row["model"]
            feature_importances = {
                col.replace("feature_importance_", ""): row[col]
                for col in feature_importance_columns
                if pd.notna(row[col])
            }
            print(f"Feature importances for {model_name}: {feature_importances}")


if __name__ == "__main__":
    # Example usage
    file_path = "/home/smebellis/ece579/final_project/network_anomaly_detection/metrics/all_model_metrics_IMBALANCED.csv"
    visualizer = ModelMetricsVisualizer(file_path)
    visualizer.plot_model_metrics()
    # visualizer.display_best_params()
    # visualizer.display_feature_importances()
