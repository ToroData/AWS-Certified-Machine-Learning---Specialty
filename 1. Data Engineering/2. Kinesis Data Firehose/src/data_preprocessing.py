"""
Module with the logic for loading and preparing data

Author: Ricard Santiago Raigada García
Date: 31-08-2024
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    """Load the Iris dataset.

    Returns:
        pd.DataFrame: Dataframe containing the Iris dataset
    """
    iris = load_iris(as_frame=True)
    data = pd.concat([iris['data'], iris['target']], axis=1)
    return data


def preprocess_data(data):
    """Function to preprocess data ingested

    Args:
        data (pd.DataFrame): Dataframe

    Returns:
        tuple: A tuple containing four elements:
            - X_train (numpy.ndarray): Training features after scaling
            - X_test (numpy.ndarray): Test features after scaling
            - y_train (pd.Series): Training labels
            - y_test (pd.Series): Test labels
    """
    # Split features
    X = data.drop('target', axis=1)
    y = data['target']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
