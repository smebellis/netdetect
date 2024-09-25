import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    f1_score,
    precision_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import seaborn as sns
import warnings

from src.data_preprocessing import file_load

# Check if the file exists, if it does exist then load it from the folder, if not then return an error


def clean_data(df):
    # Replace all the inf values with NaNs before replacing them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop NaN values
    df.dropna(inplace=True)

    # Check if there are duplicates in the dataframe.
    df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    return df


if __name__ == "__main__":
    combined_df = file_load(file_path="../data/processed/combined_df.pkl")
