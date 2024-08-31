"""
Module with the basic logic to train a model

Author: Ricard Santiago Raigada García
Date: 31-08-2024
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib


def train_model(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning using GridSearchCV

    Args:
        X_train (numpy.ndarray or pd.DataFrame): Training features
        y_train (numpy.ndarray or pd.Series): Training labels

    Returns:
        RandomForestClassifier: The best model after hyperparameter tuning
    """
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and return the accuracy

    Args:
        model (RandomForestClassifier): The trained model to evaluate
        X_test (numpy.ndarray or pd.DataFrame): Test features
        y_test (numpy.ndarray or pd.Series): Test labels

    Returns:
        float: The accuracy of the model on the test set
    """
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    return accuracy


def save_model(model, filepath):
    """
    Save the trained model to a file

    Args:
        model (RandomForestClassifier): The model to save
        filepath (str): Path to the file where the model will be saved
    """
    joblib.dump(model, filepath)
