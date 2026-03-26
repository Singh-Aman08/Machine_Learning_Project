import sys
import os 
from rootlogger import logger
from exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformation
from model_trainer import ModelTrainer

@dataclass
class DataIngestionCongig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionCongig()
        
    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method")
        try:
            df = pd.read_csv(r"C:\Users\Aman Kumar Singh\Desktop\ML_Project\Machine_Learning_Project\data\cleaned_data.csv")
            logger.info("Completed data reading")
            
            os.makedirs("artifacts", exist_ok=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header=True)
            
            logger.info("Ingestion of the data is completed")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
            
        except Exception as e:
            logger.error(f"Data ingestion failed : {e}")
            raise CustomException(e, sys)




if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    
    obj1 = DataTransformation()
    train_arr, test_arr, _ = obj1.initiate_data_transformation(train_path, test_path)
    
    obj3 = ModelTrainer()
    model_name, score =  obj3.initiate_model_trainer( train_arr, test_arr)
    
    print (model_name, score)
    
    
    
    
        
        
        
        
        
    
