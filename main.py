# main.py

"""
Main script to run the full ML pipeline:
1. Data ingestion
2. Data transformation
3. Model training and evaluation
4. Reporting the best model and R² scores
"""

from src.exception import CustomException
from src.rootlogger import logger
from src.data_transformation import DataTransformation
from src.data_ingestion import DataIngestion
from src.model_trainer import ModelTrainer
import warnings
from typing import Tuple, Dict, Any
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def model_score() -> Tuple[str, float, Dict[str, Tuple[float, float, Any]]]:
    """
    Executes the ML pipeline to ingest data, transform it, train models, 
    and return the best-performing model along with its score and detailed report.
    
    Returns:
        model_name (str): Name of the best performing model
        score (float): R² score of the best performing model on test data
        mods_report (dict): Dictionary of all models with their train/test scores and object
    """
    try:
        logger.info("Starting data ingestion...")
        data_ing = DataIngestion()
        train_path, test_path = data_ing.initiate_data_ingestion()
        logger.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")

        logger.info("Starting data transformation...")
        data_trans = DataTransformation()
        train_arr, test_arr, _ = data_trans.initiate_data_transformation(train_path, test_path)
        logger.info("Data transformation completed.")

        logger.info("Starting model training and evaluation...")
        model_train = ModelTrainer()
        model_name, score, mods_report = model_train.initiate_model_trainer(train_arr, test_arr)
        logger.info(f"Model training completed. Best model: {model_name}, Test R²: {score:.4f}")

        return model_name, score, mods_report

    except Exception as e:
        logger.error("Error occurred in model_score function.")
        raise CustomException(e,sys)


def report():
    """
    Generates a console report of all models' train and test R² scores 
    and highlights the best-performing model.
    """
    try:
        best_model, score, models_report = model_score()

        logger.info("Preparing model evaluation report...")
        print("*" * 60)
        print(f"{'MODEL':<25}{'R2_Train':<15}{'R2_Test':<15}")
        print("*" * 60)

        # Loop through all models and print their scores
        for name, (test_score, train_score, mod) in models_report.items():
            print(f"{name:<25}{train_score:<15.4f}{test_score:<15.4f}")

        print("-" * 100)
        print(f"The best-performing model is {best_model} with a test R² score of {score:.4f}.")
        print("-" * 100)

        logger.info("Model evaluation report displayed successfully.")

    except Exception as e:
        logger.error("Error occurred in report function.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    report()
        
        
        
    



    
    
