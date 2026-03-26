from exception import CustomException
from rootlogger import logger
from data_transformation import DataTransformation
from data_ingestion import DataIngestion
from model_trainer import ModelTrainer
import warnings
warnings.filterwarnings("ignore")

def model_score():
    data_ing = DataIngestion()
    data_trans = DataTransformation()
    model_train =  ModelTrainer()
    
    train_path, test_path = data_ing.initiate_data_ingestion()
    train_arr, test_arr, _ = data_trans.initiate_data_transformation(train_path, test_path)
    model_name, score, mods_report =  model_train.initiate_model_trainer( train_arr, test_arr)
    
    return model_name, score, mods_report

def report():
    best_model, score, models_report = model_score()
    
    print("*" * 60)
    print(f"{'MODEL':<25}{'R2_Train':<15}{'R2_Test':<15}")
    print("*" * 60)

    for name, (test_score, train_score, mod) in models_report.items():
        print(f"{name:<25}{train_score:<15.4f}{test_score:<15.4f}")
        
    print("-" * 100)
    print(f"The best-performing model is {best_model} with a test R² score of {score:.4f}.")
    print("-" * 100)
        
        
        
if __name__ == "__main__":
    report()
        
        
        
        
    



    
    
