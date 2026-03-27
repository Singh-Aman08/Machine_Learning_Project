# model_trainer.py

"""
Module for training and evaluating multiple regression models,
selecting the best-performing model based on R² score,
and saving it to disk.
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from src.rootlogger import logger
from src.exception import CustomException
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainingConfig:
    """
    Configuration for model training.
    Holds the path where the trained model will be saved.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Class to handle training, hyperparameter tuning, evaluation,
    and saving of multiple regression models.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> Tuple[Any, float, Dict[str, Tuple[float, float, Any]]]:
        """
        Train multiple regression models, evaluate them, and save the best model.

        Args:
            train_array (np.ndarray): Training data including features and target as last column.
            test_array (np.ndarray): Test data including features and target as last column.

        Returns:
            best_model (Any): Trained model object with the highest test R² score.
            best_model_score (float): R² score of the best model on test data.
            model_report (Dict[str, Tuple[float, float, Any]]): Dictionary with all models, 
                their train/test scores, and model objects.

        Raises:
            CustomException: If any error occurs during model training or evaluation.
        """
        try:
            # Split features and target
            x_train: np.ndarray = train_array[:, :-1]
            x_test: np.ndarray = test_array[:, :-1]
            y_train: np.ndarray = train_array[:, -1]
            y_test: np.ndarray = test_array[:, -1]

            logger.info("Test/train split completed.")

            # Define models
            model: Dict[str, Any] = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbours": KNeighborsRegressor(),
                "Xgboost": XGBRegressor(),
                "Catboost": CatBoostRegressor(verbose=0),
                "Adaboost": AdaBoostRegressor(),
                "Support Vector Machine": SVR()
            }

            # Define hyperparameter grids
            param_grids: Dict[str, Dict[str, list]] = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2']
                },
                "Decision Tree": {
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2']
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300, 400],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 10, 20],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },
                "K-Nearest Neighbours": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "Xgboost": {
                    'n_estimators': [100, 200, 300, 400],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 10, 20],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.3, 0.5]
                },
                "Catboost": {
                    'iterations': [500, 1000, 1500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'depth': [3, 5, 7, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]
                },
                "Adaboost": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
                },
                "Support Vector Machine": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]
                }
            }

            logger.info("Model evaluation started.")

            # Evaluate all models
            model_report: Dict[str, Tuple[float, float, Any]] = evaluate_model(
                Xtrain=x_train,
                Xtest=x_test,
                Ytrain=y_train,
                Ytest=y_test,
                Model=model,
                params=param_grids
            )

            logger.info("Model evaluation completed.")

            # Select the best model
            best_model_score: float = max([score for score, _, _ in model_report.values()])
            best_model: Any = None
            for name in model_report:
                if model_report[name][0] == best_model_score:
                    best_model = model_report[name][2]

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logger.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            return best_model, best_model_score, model_report

        except Exception as e:
            logger.error("Model training failed.")
            raise CustomException(e, sys)
        
    
    
    




