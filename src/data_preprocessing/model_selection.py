import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

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
from preprocess import split_data, remove_features
sys.path.append("/home/smebellis/ece579/final_project/network_anomaly_detection/")

from notebooks.exploratory import file_load, balance_classes

models = {
    'Random Forest': RandomForestClassifier(),
    'Xgboost': XGBClassifier(), 
    'Random Tree': DecisionTreeClassifier(),
    'Naïve Bayes': GaussianNB(),
}

def select_model(models, X_train, y_train)

def main():
    pass