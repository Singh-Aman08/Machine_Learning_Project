import os 
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from rootlogger import logger
from exception import CustomException
from utils import saveobject, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            x_train, x_test, y_train, y_test = (train_array[:, :-1], test_array[:,:-1], train_array[:,-1], test_array[:,-1])
            logger.info(f"Test Train Split Completed")
            
            model = {"Random Forest": RandomForestRegressor(),
                     "Decision Tree": DecisionTreeRegressor(),
                     "Gradient Boosting": GradientBoostingRegressor(),
                     "Linear Regression" : LinearRegression(),
                     "K-Nearest Neighbours": KNeighborsRegressor(),
                     "Xgboost":XGBRegressor(),
                     "Catboost": CatBoostRegressor(),
                     "Adaboost": AdaBoostRegressor(),
                     "Support Vector Machine": SVR()
                     }
            
            logger.info(f"Model Evaluation Started")
            model_report : dict = evaluate_model(Xtrain = x_train, Xtest = x_test, Ytrain = y_train, Ytest = y_test, Model = model)
            logger.info(f"Model Evaluation Completed")
            
            best_model_score = max([score for score, mod in model_report.values()])
            
            for i in model_report:
                if model_report[i][0] == best_model_score:
                    best_model = model_report[i][1] 
            
            saveobject(self.model_trainer_config.trained_model_file_path, best_model)
            logger.info(f"Model is saved in {self.model_trainer_config.trained_model_file_path}")
            
            return best_model, best_model_score
                
        except Exception as e:
            logger.error(f"Model Training Failed")
            raise CustomException(e, sys)
        
        
    
    
    




