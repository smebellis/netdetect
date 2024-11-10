# data_cleaner.py

import pandas as pd
import numpy as np
import logging
from typing import Optional, List


class DataCleaner:
    def __init__(
        self,
        replace_inf: bool = True,
        drop_na: bool = True,
        drop_duplicates: bool = True,
        subset_duplicates: Optional[List[str]] = None,
        inplace: bool = False,
    ):
        """
        Initializes the DataCleaner with specified configurations.

        Args:
            replace_inf (bool, optional): Whether to replace infinite values with NaN. Defaults to True.
            drop_na (bool, optional): Whether to drop rows containing NaN values. Defaults to True.
            drop_duplicates (bool, optional): Whether to drop duplicate rows. Defaults to True.
            subset_duplicates (List[str], optional): Columns to consider when identifying duplicates. Defaults to None.
            inplace (bool, optional): Whether to perform operations in place. Defaults to False.
        """
        self.replace_inf = replace_inf
        self.drop_na = drop_na
        self.drop_duplicates = drop_duplicates
        self.subset_duplicates = subset_duplicates
        self.inplace = inplace
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by handling infinite values, missing data, and duplicates.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error("Input is not a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        df_cleaned = df.copy() if not self.inplace else df

        # Replace infinite values with NaN
        if self.replace_inf:
            inf_count = np.isinf(df_cleaned).sum().sum()
            df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.logger.info(f"Replaced {inf_count} infinite values with NaN.")

        # Drop rows with NaN values
        if self.drop_na:
            initial_shape = df_cleaned.shape
            df_cleaned.dropna(inplace=True)
            final_shape = df_cleaned.shape
            dropped_na = initial_shape[0] - final_shape[0]
            self.logger.info(f"Dropped {dropped_na} rows containing NaN values.")

        # Drop duplicate rows
        if self.drop_duplicates:
            if self.subset_duplicates:
                duplicated = df_cleaned.duplicated(subset=self.subset_duplicates).sum()
                df_cleaned.drop_duplicates(subset=self.subset_duplicates, inplace=True)
                self.logger.info(
                    f"Dropped {duplicated} duplicate rows based on subset {self.subset_duplicates}."
                )
            else:
                duplicated = df_cleaned.duplicated().sum()
                df_cleaned.drop_duplicates(inplace=True)
                self.logger.info(f"Dropped {duplicated} duplicate rows.")

        return df_cleaned
