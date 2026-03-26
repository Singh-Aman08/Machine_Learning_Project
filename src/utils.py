# utils.py

"""
Utility module for the ML project.
Contains functions for saving objects and evaluating models
with hyperparameter tuning via RandomizedSearchCV.
"""

import os
import sys
import dill
from typing import Any, Dict, Tuple
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
import numpy as np

from exception import CustomException


def saveobject(file_path: str, obj: Any) -> str:
    """
    Saves a Python object to disk using dill serialization.

    Args:
        file_path (str): Path to save the object.
        obj (Any): Python object to serialize and save.

    Returns:
        str: Path where the object was saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        return file_path
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(
    Xtrain: np.ndarray,
    Xtest: np.ndarray,
    Ytrain: np.ndarray,
    Ytest: np.ndarray,
    Model: Dict[str, Any],
    params: Dict[str, Dict[str, list]]
) -> Dict[str, Tuple[float, float, Any]]:
    """
    Trains and evaluates multiple models using RandomizedSearchCV
    for hyperparameter tuning. Returns a report dictionary with
    test/train R² scores and the fitted model.

    Args:
        Xtrain (np.ndarray): Training features.
        Xtest (np.ndarray): Test features.
        Ytrain (np.ndarray): Training target.
        Ytest (np.ndarray): Test target.
        Model (Dict[str, Any]): Dictionary of model names and objects.
        params (Dict[str, Dict[str, list]]): Dictionary of hyperparameter grids.

    Returns:
        Dict[str, Tuple[float, float, Any]]: Dictionary with keys as model names
        and values as a tuple of (test_score, train_score, trained_model)
    """
    try:
        report: Dict[str, Tuple[float, float, Any]] = {}

        for name in Model:
            mod = Model.get(name)
            param_grid = params.get(name, {})

            # Hyperparameter tuning with RandomizedSearchCV
            rs = RandomizedSearchCV(mod, param_grid, cv=5, n_jobs=-1)
            rs.fit(Xtrain, Ytrain)

            # Set the best parameters to the model and refit
            mod.set_params(**rs.best_params_)
            mod.fit(Xtrain, Ytrain)

            # Predictions
            Ytrain_pred = mod.predict(Xtrain)
            Ytest_pred = mod.predict(Xtest)

            # Compute R² scores
            train_score = r2_score(Ytrain, Ytrain_pred)
            test_score = r2_score(Ytest, Ytest_pred)

            report[name] = (test_score, train_score, mod)

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
        

