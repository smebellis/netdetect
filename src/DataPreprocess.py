from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

from LoggerSingleton import get_logger

# Configure Logger
logger = get_logger(__name__)


class DataPreprocessor:
    def __init__(
        self,
        label_column: str,
        scale: bool = False,
        feature_selection: bool = False,
        k_features: int = 10,
    ):
        """
        Initializes the DataPreprocessor with specified configurations.

        Args:
            label_column (str): Name of the label column.
            scale (bool, optional): Whether to scale numerical features. Defaults to False.
            feature_selection (bool, optional): Whether to perform feature selection. Defaults to False.
            k_features (int, optional): Number of top features to select. Defaults to 10.
        """
        self.label_column = label_column.strip()
        self.scale = scale
        self.feature_selection = feature_selection
        self.k_features = k_features
        self.label_encoder = None
        self.scaler = None
        self.selected_features = None
        self.logger = logger

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame column names by stripping whitespace."""
        self.logger.debug("Cleaning column names.")
        df.columns = df.columns.str.strip()
        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode the labels using LabelEncoder."""
        self.logger.debug(f"Encoding labels in column: {self.label_column}")
        self.label_encoder = LabelEncoder()
        df[self.label_column] = self.label_encoder.fit_transform(df[self.label_column])
        return df

    def scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical data using StandardScaler."""
        self.logger.debug("Scaling numerical features.")
        numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
        self.scaler = StandardScaler()
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select top k features using SelectKBest while preserving the label column.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: DataFrame with selected features and label column.
        """
        self.logger.debug(f"Selecting top {self.k_features} features.")
        X = df.drop(self.label_column, axis=1)
        y = df[self.label_column]
        selector = SelectKBest(score_func=f_classif, k=self.k_features)
        X_new = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()]
        self.logger.info(
            f"Selected top {self.k_features} features: {list(self.selected_features)}"
        )

        # Create a dataframe with the selected features
        X_selected_df = pd.DataFrame(
            X_new, columns=self.selected_features, index=df.index
        )

        # Add the label column back to the dataframe
        result_df = pd.concat([X_selected_df, df[self.label_column]], axis=1)

        return result_df

    def preprocess(self, df: pd.DataFrame) -> tuple:
        """
        Execute the preprocessing steps on the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            tuple: Processed dataframe, label encoder, and scaler.
        """
        self.logger.info("Starting data preprocessing.")

        # Clean column names
        df = self.clean_column_names(df)

        # Encode labels
        df = self.encode_labels(df)

        # Scale data if required
        if self.scale:
            df = self.scale_data(df)

        # Feature selection if required
        if self.feature_selection:
            df = self.select_features(df)

        self.logger.info("Data preprocessing completed.")
        return df, self.label_encoder, self.scaler
