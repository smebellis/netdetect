# data_loader.py

import pandas as pd
from pathlib import Path
from typing import Union, List
from tqdm import tqdm

from DataCleaner import DataCleaner

from LoggerSingleton import get_logger

logger = get_logger(__name__)


class DataLoader:
    def __init__(
        self,
        pickle_path: Union[str, Path],
        csv_dir: Union[str, Path],
        data_cleaner: DataCleaner,
    ):
        """
        Initializes the DataLoader with specified configurations.

        Args:
            pickle_path (str or Path): The path to the pickle file.
            csv_dir (str or Path): The directory containing CSV files.
            data_cleaner (DataCleaner): An instance of DataCleaner to clean dataframes.
        """
        self.pickle_path = Path(pickle_path)
        self.csv_dir = Path(csv_dir)
        self.data_cleaner = data_cleaner
        self.logger = logger

    def load_from_pickle(self) -> pd.DataFrame:
        """
        Loads the DataFrame from a pickle file.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            Exception: If there is an error loading the pickle file.
        """
        self.logger.info(f"Loading DataFrame from pickle file: {self.pickle_path}")
        try:
            df = pd.read_pickle(self.pickle_path)
            self.logger.info(f"DataFrame loaded successfully from {self.pickle_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading pickle file: {e}")
            raise

    def load_from_csv(self) -> pd.DataFrame:
        """
        Loads and concatenates DataFrames from CSV files in the specified directory.

        Returns:
            pd.DataFrame: The concatenated DataFrame.

        Raises:
            FileNotFoundError: If no CSV files are found in the specified directory.
            ValueError: If no DataFrames are loaded successfully.
            Exception: For any other exceptions during file operations.
        """
        self.logger.info(f"Loading CSV files from directory: {self.csv_dir}")
        csv_files = list(self.csv_dir.glob("*.csv"))
        if not csv_files:
            self.logger.error("No CSV files found in the specified directory.")
            raise FileNotFoundError("No CSV files found in the specified directory.")

        dataframes = []
        for file in tqdm(csv_files, desc="Processing CSV Files"):
            try:
                self.logger.info(f"Reading CSV file: {file}")
                df = pd.read_csv(file)
                cleaned = self.data_cleaner.clean_data(df)
                dataframes.append(cleaned)
            except Exception as e:
                self.logger.error(f"Error reading {file}: {e}")
                continue  # Skip files that cause errors

        if not dataframes:
            self.logger.error("No DataFrames were loaded successfully.")
            raise ValueError("No DataFrames were loaded successfully.")

        try:
            concatenated_df = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Concatenated {len(dataframes)} DataFrames.")
        except Exception as e:
            self.logger.error(f"Error concatenating DataFrames: {e}")
            raise

        try:
            concatenated_df.to_pickle(self.pickle_path)
            self.logger.info(f"Saved concatenated DataFrame to {self.pickle_path}")
        except Exception as e:
            self.logger.error(f"Error saving pickle file: {e}")
            raise

        return concatenated_df

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from a pickle file if it exists; otherwise, loads from CSV files, cleans, concatenates,
        saves to pickle, and returns the DataFrame.

        Returns:
            pd.DataFrame: The loaded or newly created DataFrame.

        Raises:
            FileNotFoundError: If no CSV files are found and the pickle file does not exist.
            pd.errors.EmptyDataError: If CSV files are empty.
            Exception: For any other exceptions during file operations.
        """
        if not self.pickle_path.exists():
            self.logger.info(f"Pickle file does not exist: {self.pickle_path}")
            return self.load_from_csv()
        else:
            return self.load_from_pickle()
