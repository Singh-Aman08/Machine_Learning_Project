import os
from exception import CustomException
import dill
import sys

def saveobject(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
            
        return file_path
    
    except Exception as e:
        raise CustomException(e,sys)


