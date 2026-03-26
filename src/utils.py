import os
from exception import CustomException
import dill
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def saveobject(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
            
        return file_path
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(Xtrain, Xtest, Ytrain, Ytest, Model, params):
    try:
        report = {}
        
        for i in Model:
            mod = Model.get(i)
            param = params.get(i)
            
            rs = RandomizedSearchCV(mod, param, cv = 5, n_jobs=-1)
            rs.fit(Xtrain, Ytrain)
            mod.set_params(**rs.best_params_)
            mod.fit(Xtrain, Ytrain)
            
            
            Ytrain_pred = mod.predict(Xtrain)
            
            Ytest_pred = mod.predict(Xtest)
            
            train_model_score = r2_score(Ytrain, Ytrain_pred)
            test_model_score = r2_score(Ytest, Ytest_pred)
            report[i] = (test_model_score, train_model_score, mod)
        return report
            
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
        

