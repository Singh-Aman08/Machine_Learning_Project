import sys
import os 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from exception import CustomException
from rootlogger import logger
from utils import saveobject

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
    
    def get_data_transformer_object(self):
        
        try:
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            
            cat_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                                             ("Encoder", OneHotEncoder(handle_unknown = "ignore",  drop="first"))])
            
            preprocessor = ColumnTransformer([("Categorical Transformation", cat_pipeline, categorical_columns)])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            logger.info(f"Initiating data Transformation")
            
            train_df = pd.read_csv(train_path)
            logger.info(f"Reading train.csv completed : {train_path}")
            
            test_df = pd.read_csv(test_path)
            logger.info(f"Reading test.csv completed : {test_path}")
            
            
            processing_obj = self.get_data_transformer_object()
            logger.info(f"Processor Object Created")
            
            input_feature_train_df = train_df.drop(columns=["average"])
            target_feature_train_ser = train_df["average"]
            
            input_feature_test_df = test_df.drop(columns=["average"])
            target_feature_test_ser = test_df["average"]
            
            processed_train_feature_arr = processing_obj.fit_transform(input_feature_train_df)
            processed_test_feature_arr = processing_obj.transform(input_feature_test_df)
            logger.info("Data Transformation Completed")
            
            train_arr = np.c_[processed_train_feature_arr, target_feature_train_ser]
            test_arr = np.c_[processed_test_feature_arr, target_feature_test_ser]
            
            saveobject(self.data_transformation_config.preprocessor_obj_file_path, processing_obj)
            
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
       
        except Exception as e:
            raise CustomException(e, sys)
        
    
    
