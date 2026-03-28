import sys
import dill
import pandas as pd
import os
from src.rootlogger import logger
from src.exception import CustomException


class CustomData:
    """
    This class is used to take user input data and convert it into
    a pandas DataFrame format suitable for prediction.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        math_score: float,
        reading_score: float
    ):
        """
        Initialize input features for prediction.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Converts input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing one row of input features.
        """
        try:
            logger.info("Converting custom data into DataFrame")

            data_df = pd.DataFrame({
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "math_score": [self.math_score],
                "reading_score": [self.reading_score]
            })

            logger.info("DataFrame creation successful")
            return data_df

        except Exception as e:
            logger.error("Error occurred while creating DataFrame")
            raise CustomException(e, sys)


class PredictionPipeline:
    """
    This class handles loading the trained model and preprocessor,
    transforming the input data, and generating predictions.
    """

    def __init__(self):
        logger.info("PredictionPipeline initialized")

    def predict(self, data: pd.DataFrame) -> float:
        """
        Generates predictions using the trained model.

        Args:
            data (pd.DataFrame): Input features in DataFrame format

        Returns:
            float: Predicted value(s)
        """
        try:
            logger.info("Starting prediction pipeline")

            model_path = os.path.join("artifacts", "model.pkl")
            preprss_path = os.path.join("artifacts", "preprocessor.pkl")

            logger.info("Loading model and preprocessor")

            with open(model_path, "rb") as f:
                model = dill.load(f)

            with open(preprss_path, "rb") as f:
                preprocessor = dill.load(f)

            logger.info("Model and preprocessor loaded successfully")

            logger.info("Transforming input data")
            processed_data = preprocessor.transform(data)

            logger.info("Making predictions")
            preds = model.predict(processed_data)

            logger.info("Prediction successful")
            return preds

        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)
    
    
    
        
        
        
        
        
        