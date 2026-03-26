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
                     "Catboost": CatBoostRegressor(verbose=0),
                     "Adaboost": AdaBoostRegressor(),
                     "Support Vector Machine": SVR()
                     }
            
            param_grids = {
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
                # LinearRegression has very few hyperparameters; could tune 'fit_intercept' and 'normalize' (if old sklearn)
                'fit_intercept': [True, False],
                'positive': [True, False]
            },

            "K-Nearest Neighbours": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1=Manhattan, 2=Euclidean
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
            
            logger.info(f"Model Evaluation Started")
            model_report : dict = evaluate_model(Xtrain = x_train, Xtest = x_test, Ytrain = y_train, Ytest = y_test, Model = model, params = param_grids)
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
        
        
    
    
    




