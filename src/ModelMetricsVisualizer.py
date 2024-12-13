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

        # Plot 1: Training Accuracy for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_accuracy",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train Accuracy",
        )
        plt.title("Train Accuracy for All Models")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 2: Test Accuracy for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="test_accuracy",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Test Accuracy",
        )

        plt.title("Test Accuracy for All Models")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 3: Training Recall for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_recall",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train Recall",
        )

        plt.title("Training Recall for All Models")
        plt.xlabel("Model")
        plt.ylabel("Recall")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 4:  Test Recall for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="test_recall",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Test Recall",
        )

        plt.title("Test Recall for All Models")
        plt.xlabel("Model")
        plt.ylabel("Recall")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 5: Training Precision for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_precision",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train Precision",
        )

        plt.title("Training Precision for All Models")
        plt.xlabel("Model")
        plt.ylabel("Precision")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 6:  Test Precision for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="test_precision",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Test Precision",
        )

        plt.title("Test Precision for All Models")
        plt.xlabel("Model")
        plt.ylabel("Precision")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 7: Training F1 for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="train_f1",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Train F1",
        )

        plt.title("Training F1 for All Models")
        plt.xlabel("Model")
        plt.ylabel("F1")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Plot 8:  Test F1 for All Models
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="model",
            y="test_f1",
            data=self.metrics_df,
            color="b",
            alpha=0.6,
            label="Test F1",
        )

        plt.title("Test F1 for All Models")
        plt.xlabel("Model")
        plt.ylabel("F1")
        plt.xticks(rotation=45)
        plt.legend()
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
    file_path = "/home/smebellis/ece579/final_project/network_anomaly_detection/metrics/all_model_metrics_VAE.csv"
    visualizer = ModelMetricsVisualizer(file_path)
    visualizer.plot_model_metrics()
    # visualizer.display_best_params()
    # visualizer.display_feature_importances()
