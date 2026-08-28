import pandas as pd
import numpy as np


def load_housing_data(path: str):
    data = pd.read_csv(path)
    data = data.dropna()
    return data


def prepare_features(data, feature_cols, target_col):
    X = data[feature_cols].to_numpy()
    y = data[target_col].to_numpy()
    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    m = X.shape[0]
    indices = np.random.permutation(m)
    test_count = int(m * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
