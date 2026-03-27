# data_ingestion.py

"""
Module for data ingestion.
Handles reading raw data, splitting into train/test sets, 
saving processed datasets, and returning paths for downstream use.
"""

import sys
import os
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.rootlogger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    Holds paths for raw, train, and test datasets.
    """
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """
    Class to handle the ingestion of raw data and splitting into training/testing sets.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Reads raw CSV data, splits into train/test sets, saves them, 
        and returns the paths for further processing.

        Returns:
            Tuple[str, str]: Paths to the training and testing CSV files
        """
        logger.info("Entered the data ingestion method.")
        try:
            # Read raw data
            df: pd.DataFrame = pd.read_csv(
                r"C:\Users\Aman Kumar Singh\Desktop\ML_Project\Machine_Learning_Project\data\cleaned_data.csv"
            )
            logger.info("Completed reading raw data.")

            # Ensure artifacts directory exists
            os.makedirs("artifacts", exist_ok=True)

            # Split into train/test sets
            train_set: pd.DataFrame
            test_set: pd.DataFrame
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save datasets to disk
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Data ingestion completed successfully.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise CustomException(e, sys)




    
    
    
    
        
        
        
        
        
    
