# data_transformation.py

"""
Module for data transformation and preprocessing.
Handles numerical and categorical feature preprocessing, 
and returns transformed train/test arrays along with the preprocessor object path.
"""

import sys
import os
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from exception import CustomException
from rootlogger import logger
from utils import saveobject


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    Holds the path where the preprocessing object will be saved.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Class to handle data preprocessing:
    - Categorical features: imputation + one-hot encoding
    - Numerical features: median imputation + standard scaling
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a ColumnTransformer for preprocessing.

        Categorical columns:
            - Impute missing values using most frequent
            - OneHotEncode (drop first to avoid dummy variable trap)
        Numerical columns:
            - Impute missing values using median
            - StandardScaler for normalization

        Returns:
            ColumnTransformer: Preprocessing object ready for fit/transform
        """
        try:
            # Define columns
            categorical_columns: list[str] = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            numerical_columns: list[str] = ["math_score", "reading_score"]

            # Pipelines
            cat_pipeline: Pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
            ])

            num_pipeline: Pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor: ColumnTransformer = ColumnTransformer(transformers=[
                ("categorical_transformation", cat_pipeline, categorical_columns),
                ("numerical_transformation", num_pipeline, numerical_columns)
            ])

            return preprocessor

        except Exception as e:
            logger.error("Error occurred in get_data_transformer_object.")
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Performs full data transformation on train and test datasets.

        Steps:
            1. Read CSV files
            2. Drop target column from features
            3. Apply preprocessing transformations
            4. Combine processed features with target
            5. Save preprocessor object to disk

        Args:
            train_path (str): Path to training CSV file
            test_path (str): Path to testing CSV file

        Returns:
            train_arr (np.ndarray): Transformed training data (features + target)
            test_arr (np.ndarray): Transformed testing data (features + target)
            preprocessor_path (str): Path to saved preprocessor object
        """
        try:
            logger.info("Initiating data transformation...")

            # Read datasets
            train_df: pd.DataFrame = pd.read_csv(train_path)
            logger.info(f"Train CSV read successfully: {train_path}")

            test_df: pd.DataFrame = pd.read_csv(test_path)
            logger.info(f"Test CSV read successfully: {test_path}")

            # Create preprocessor object
            processing_obj: ColumnTransformer = self.get_data_transformer_object()
            logger.info("Preprocessor object created successfully.")

            # Split features and target
            input_feature_train_df: pd.DataFrame = train_df.drop(columns=["writing_score"])
            target_feature_train_ser: pd.Series = train_df["writing_score"]

            input_feature_test_df: pd.DataFrame = test_df.drop(columns=["writing_score"])
            target_feature_test_ser: pd.Series = test_df["writing_score"]

            # Apply transformations
            processed_train_feature_arr: np.ndarray = processing_obj.fit_transform(input_feature_train_df)
            processed_test_feature_arr: np.ndarray = processing_obj.transform(input_feature_test_df)
            logger.info("Data transformation completed successfully.")

            # Combine features and target
            train_arr: np.ndarray = np.c_[processed_train_feature_arr, target_feature_train_ser]
            test_arr: np.ndarray = np.c_[processed_test_feature_arr, target_feature_test_ser]

            # Save preprocessor object
            saveobject(self.data_transformation_config.preprocessor_obj_file_path, processing_obj)
            logger.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logger.error("Error occurred in initiate_data_transformation.")
            raise CustomException(e, sys)
    
